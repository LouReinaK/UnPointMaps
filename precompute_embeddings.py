import sys
import os
import logging
import multiprocessing
import torch
from tqdm import tqdm

# Add current directory to path so we can import src
sys.path.append(os.getcwd())

try:
    from src.processing.dataset_filtering import convert_to_dict_filtered
    from src.processing.remove_nonsignificative_words import remove_nonsignificant_words_multilang
    from src.processing.embedding_service import EmbeddingService
except ImportError as e:
    print(f"Error importing modules: {e}")
    sys.exit(1)

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(processName)s - %(message)s')


def clean_text_worker(texts_chunk):
    """
    Worker function to clean texts in parallel.
    We import and use the cleaning function directly for efficiency.
    """
    cleaned = []
    # Avoid re-import loops by importing inside if needed, but here we passed
    # function or imported it top-level
    for t in texts_chunk:
        c = remove_nonsignificant_words_multilang(t)
        if c and c.strip():
            cleaned.append(c)
    return cleaned


def compute_embeddings_worker(texts_chunk, gpu_id=0):
    """
    Worker function to compute embeddings.
    Each worker initializes its own model/service instance.
    """
    # Force specific GPU if needed, but usually 'cuda' finds best or all.
    # If we want to split GPUs across workers, we can set CUDA_VISIBLE_DEVICES via environment
    # os.environ["CUDA_VISIBLE_DEVICES"] = str(gpu_id)

    service = EmbeddingService.get_instance()
    # Explicitly load model in this process
    service.load_model()

    # Process the chunk
    # Processing in internal batches is handled by the loop below or inside encode?
    # Actually service.encode handles a list.
    # To be safe with memory, let's process this chunk in sub-batches if it is
    # huge

    sub_batch_size = 128
    results_count = 0

    for i in range(0, len(texts_chunk), sub_batch_size):
        batch = texts_chunk[i:i + sub_batch_size]
        try:
            service.encode(batch)  # This handles caching (read/write)
            results_count += len(batch)
        except Exception as e:
            logging.error(f"Error processing batch: {e}")

    return results_count


def chunk_list(data, num_chunks):
    """Yield successive n-sized chunks from data."""
    avg = len(data) / float(num_chunks)
    out = []
    last = 0.0
    while last < len(data):
        out.append(data[int(last):int(last + avg)])
        last += avg
    return out


def precompute_embeddings():
    # Set start method to spawn for better CUDA compatibility if using Linux,
    # but on Windows 'spawn' is default/only.
    # multiprocessing.set_start_method('spawn', force=True)

    print("==== Starting Parallel Embeddings Precomputation ====")

    # Check GPU
    if torch.cuda.is_available():
        print(f"--> CUDA available. GPU Count: {torch.cuda.device_count()}")
        print(f"--> Current Device: {torch.cuda.get_device_name(0)}")
    else:
        print("--> CUDA NOT available. Running on CPU.")

    print("--> Loading and filtering dataset...")
    df = convert_to_dict_filtered()

    if df is None:
        print("Error: No data returned.")
        return

    print(f"--> Dataset loaded. Rows: {len(df)}")

    # Extract texts
    raw_texts = []
    print("--> Extracting texts from 'title' and 'tags'...")

    if 'title' in df.columns:
        titles = df['title'].dropna().astype(str).tolist()
        raw_texts.extend(titles)

    if 'tags' in df.columns:
        tags = df['tags'].dropna().astype(str).tolist()
        raw_texts.extend(tags)

    # Deduplicate raw texts first
    unique_raw_texts = list(set(raw_texts))
    print(f"    Unique raw texts: {len(unique_raw_texts)}")

    # ---------------------------------------------------------
    # PART 1: Parallel Text Cleaning (CPU Heavy)
    # ---------------------------------------------------------
    print("--> Cleaning texts (Parallel CPU)...")

    num_cpu = multiprocessing.cpu_count()
    print(f"    Using {num_cpu} CPU processes for cleaning.")

    # Split for cleaning
    text_chunks = chunk_list(unique_raw_texts, num_cpu)

    cleaned_texts = []
    with multiprocessing.Pool(processes=num_cpu) as pool:
        for result in tqdm(
            pool.imap_unordered(clean_text_worker, text_chunks),
            total=num_cpu,
            desc="Cleaning"
        ):
            cleaned_texts.extend(result)

    # Filter short/empty
    valid_texts = [t for t in cleaned_texts if len(t) > 3]
    unique_valid_texts = list(set(valid_texts))

    print(f"--> Total unique valid texts to embed: {len(unique_valid_texts)}")

    if not unique_valid_texts:
        print("No texts to process.")
        return

    # ---------------------------------------------------------
    # PART 2: Parallel Embeddings (GPU/CPU)
    # ---------------------------------------------------------

    # Strategy: Spawn N processes. Each process loads the model (shared GPU memory if mostly read, or VRAM splits).
    # Small model (MiniLM) takes ~100MB VRAM. We can easily spawn multiple workers on one GPU.
    # Too many workers might slow down due to context switching on GPU, but
    # CPU-side preprocessing (tokenization) helps.

    # Let's target 4 workers per GPU or just CPU count if CPU only.
    if torch.cuda.is_available():
        # e.g. 4 workers if 1 GPU
        num_workers = min(num_cpu, 4 * torch.cuda.device_count())
    else:
        num_workers = max(1, num_cpu - 1)  # Leave one for OS

    print(f"--> Computing embeddings with {num_workers} workers...")

    embedding_chunks = chunk_list(unique_valid_texts, num_workers)

    # Create tuples of (chunk, gpu_id)
    # Simple round-robin for GPU ID if multiple
    worker_args = []
    gpu_count = torch.cuda.device_count() if torch.cuda.is_available() else 0

    for i, chunk in enumerate(embedding_chunks):
        g_id = i % gpu_count if gpu_count > 0 else 0
        worker_args.append((chunk, g_id))

    total_processed = 0
    # Use starmap to pass arguments
    with multiprocessing.Pool(processes=num_workers) as pool:
        # imap isn't easy with starmap, so use starmap_async or just starmap
        # To get progress bar, we can wrap starmap or use apply_async

        results = []
        for args in worker_args:
            results.append(pool.apply_async(compute_embeddings_worker, args))

        for r in tqdm(results, desc="Embedding Processes"):
            try:
                total_processed += r.get()
            except Exception as e:
                print(f"Worker failed: {e}")

    print("\n==== Precomputation Complete ====")
    print(f"Processed {total_processed} texts.")


if __name__ == "__main__":
    # Windows support for multiprocessing
    multiprocessing.freeze_support()
    precompute_embeddings()
