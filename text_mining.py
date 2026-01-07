import pandas as pd
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from collections import Counter
import re
from wordcloud import WordCloud
import matplotlib.pyplot as plt
from mlxtend.frequent_patterns import apriori, association_rules
from mlxtend.preprocessing import TransactionEncoder

# Download NLTK data if not present
nltk.download('punkt')
nltk.download('punkt_tab')
nltk.download('stopwords')

def load_dataset(file_path):
    """
    Load the dataset from a tab-separated file.
    Assumes columns: lat, lon, tags, title
    """
    df = pd.read_csv(file_path, sep='\t', header=None, names=['lat', 'lon', 'tags', 'title'])
    return df

def preprocess_text(text):
    """
    Preprocess a single text string: tokenize, remove stop words, lowercase.
    """
    if pd.isna(text):
        return []
    # Lowercase
    text = text.lower()
    # Tokenize
    tokens = word_tokenize(text)
    # Remove non-alphabetic tokens
    tokens = [word for word in tokens if word.isalpha()]
    # Remove stop words (English and French)
    stop_words = set(stopwords.words('english') + stopwords.words('french'))
    tokens = [word for word in tokens if word not in stop_words]
    return tokens

def combine_and_preprocess(df):
    """
    Combine tags and title, preprocess.
    Tags are assumed comma-separated.
    """
    df['combined_text'] = df.apply(lambda row: (row['tags'] if not pd.isna(row['tags']) else '') + ' ' + (row['title'] if not pd.isna(row['title']) else ''), axis=1)
    df['tokens'] = df['combined_text'].apply(preprocess_text)
    return df

def remove_frequent_words(tokens_list, top_n=10):
    """
    Remove the top_n most frequent words from the token lists.
    """
    all_tokens = [token for sublist in tokens_list for token in sublist]
    freq = Counter(all_tokens)
    most_common = [word for word, _ in freq.most_common(top_n)]
    # Assume these are non-meaningful; in practice, might need manual inspection
    filtered_tokens = [[token for token in tokens if token not in most_common] for tokens in tokens_list]
    return filtered_tokens

def create_binary_features(tokens_list):
    """
    Create binary features using CountVectorizer.
    """
    # Join tokens back to strings
    texts = [' '.join(tokens) for tokens in tokens_list]
    vectorizer = CountVectorizer(binary=True)
    X = vectorizer.fit_transform(texts)
    return X, vectorizer.get_feature_names_out()

def visualize_word_cloud(tokens_list):
    """
    Visualize the most frequent words using a word cloud.
    """
    all_tokens = [token for sublist in tokens_list for token in sublist]
    text = ' '.join(all_tokens)
    wordcloud = WordCloud(width=800, height=400, background_color='white').generate(text)
    plt.figure(figsize=(10, 5))
    plt.imshow(wordcloud, interpolation='bilinear')
    plt.axis('off')
    plt.show()

def compute_tfidf_for_dataset(df):
    """
    Compute TF-IDF scores for the entire dataset.
    Returns the TF-IDF vectorizer and matrix.
    """
    # Combine tokens into strings
    texts = [' '.join(tokens) for tokens in df['tokens'].tolist()]
    
    # Initialize TF-IDF vectorizer
    vectorizer = TfidfVectorizer()
    tfidf_matrix = vectorizer.fit_transform(texts)
    
    return vectorizer, tfidf_matrix


def get_cluster_tfidf_words(df, cluster_image_ids, vectorizer, tfidf_matrix, top_n=5):
    """
    Extract top TF-IDF words distinctive to a cluster compared to the whole dataset.
    
    Args:
        df: DataFrame containing the dataset
        cluster_image_ids: List of image IDs in the cluster
        vectorizer: Fitted TF-IDF vectorizer
        tfidf_matrix: TF-IDF matrix for the entire dataset
        top_n: Number of top words to return
        
    Returns:
        List of top distinctive words for the cluster
    """
    # Get indices of cluster items in the dataframe
    cluster_indices = [i for i, img_id in enumerate(df.index) if img_id in cluster_image_ids]
    
    # Compute average TF-IDF scores for the cluster
    cluster_tfidf = tfidf_matrix[cluster_indices].mean(axis=0)
    
    # Compute average TF-IDF scores for the entire dataset
    dataset_tfidf = tfidf_matrix.mean(axis=0)
    
    # Calculate distinctive scores (cluster score - dataset score)
    distinctive_scores = cluster_tfidf - dataset_tfidf
    
    # Get feature names and their distinctive scores
    feature_names = vectorizer.get_feature_names_out()
    word_scores = list(zip(feature_names, distinctive_scores.A1))
    
    # Sort by distinctive score and get top words
    word_scores.sort(key=lambda x: x[1], reverse=True)
    top_words = [word for word, score in word_scores[:top_n] if score > 0]
    
    return top_words


def label_cluster_with_tfidf(df, cluster_image_ids, top_n=5):
    """
    Label a cluster using TF-IDF distinctive words.
    
    Args:
        df: DataFrame containing the dataset
        cluster_image_ids: List of image IDs in the cluster
        top_n: Number of top words to use for labeling
        
    Returns:
        String label combining top distinctive words
    """
    # Compute TF-IDF for the entire dataset
    vectorizer, tfidf_matrix = compute_tfidf_for_dataset(df)
    
    # Get top distinctive words for the cluster
    top_words = get_cluster_tfidf_words(df, cluster_image_ids, vectorizer, tfidf_matrix, top_n)
    
    # Create label from top words
    label = " ".join(top_words)
    return label


def preprocess_for_association_rules(df, cluster_image_ids):
    """
    Preprocess data for association rule mining.
    
    Args:
        df: DataFrame containing the dataset
        cluster_image_ids: List of image IDs in the cluster
        
    Returns:
        Transaction data (list of lists) for association rule mining
    """
    # Get the tokens for items in the cluster
    cluster_tokens = df.loc[cluster_image_ids, 'tokens'].tolist()
    
    # Convert tokens to transaction format (list of lists)
    transactions = [tokens for tokens in cluster_tokens if len(tokens) > 0]
    
    return transactions


def find_frequent_itemsets(transactions, min_support=0.1):
    """
    Find frequent itemsets using Apriori algorithm.
    
    Args:
        transactions: List of transactions (list of lists)
        min_support: Minimum support threshold
        
    Returns:
        DataFrame of frequent itemsets
    """
    # Convert transactions to one-hot encoded format
    te = TransactionEncoder()
    te_ary = te.fit(transactions).transform(transactions)
    df_encoded = pd.DataFrame(te_ary, columns=te.columns_)
    
    # Find frequent itemsets
    frequent_itemsets = apriori(df_encoded, min_support=min_support, use_colnames=True)
    
    return frequent_itemsets


def generate_association_rules(frequent_itemsets, num_transactions, min_threshold=0.5):
    """
    Generate association rules from frequent itemsets.
    
    Args:
        frequent_itemsets: DataFrame of frequent itemsets
        num_transactions: Number of transactions in original data
        min_threshold: Minimum confidence threshold
        
    Returns:
        DataFrame of association rules
    """
    # Generate association rules
    rules = association_rules(frequent_itemsets, num_itemsets=num_transactions, metric="confidence", min_threshold=min_threshold, support_only=False)
    
    # Sort by lift (strength of association)
    rules = rules.sort_values('lift', ascending=False)
    
    return rules


def label_cluster_with_association_rules(df, cluster_image_ids, min_support=0.1, min_confidence=0.5, top_n=3):
    """
    Label a cluster using association rule mining.
    
    Args:
        df: DataFrame containing the dataset
        cluster_image_ids: List of image IDs in the cluster
        min_support: Minimum support threshold for frequent itemsets
        min_confidence: Minimum confidence threshold for association rules
        top_n: Number of top rules to use for labeling
        
    Returns:
        String label combining top association rules
    """
    # Preprocess data for association rules
    transactions = preprocess_for_association_rules(df, cluster_image_ids)
    
    if len(transactions) == 0:
        return "No valid transactions for association rules"
    
    # Find frequent itemsets
    frequent_itemsets = find_frequent_itemsets(transactions, min_support)
    
    if len(frequent_itemsets) == 0:
        return "No frequent itemsets found"
    
    # Generate association rules
    rules = generate_association_rules(frequent_itemsets, len(transactions), min_confidence)
    
    if len(rules) == 0:
        return "No association rules found"
    
    # Extract top rules for labeling
    top_rules = rules.head(top_n)
    
    # Create label from antecedents and consequents
    label_parts = []
    for _, rule in top_rules.iterrows():
        antecedents = ' '.join(list(rule['antecedents']))
        consequents = ' '.join(list(rule['consequents']))
        label_parts.append(f"{antecedents} -> {consequents}")
    
    label = " | ".join(label_parts)
    return label


def integrate_text_mining_results(tfidf_words, association_rules, max_words=8):
    """
    Integrate results from TF-IDF and association rules to create a comprehensive label.
    
    Args:
        tfidf_words: List of top TF-IDF words
        association_rules: List of association rule strings
        max_words: Maximum number of words to include in final label
        
    Returns:
        Comprehensive label string and supporting information dict
    """
    # Combine all words from both approaches
    all_words = []
    
    # Add TF-IDF words
    all_words.extend(tfidf_words)
    
    # Extract words from association rules
    for rule in association_rules:
        # Split rule like "paris tower -> eiffel" into individual words
        rule_words = rule.replace('->', '').replace('|', ' ').split()
        all_words.extend(rule_words)
    
    # Count word frequencies
    word_counts = Counter(all_words)
    
    # Get most common words for the label
    top_words = [word for word, count in word_counts.most_common(max_words)]
    
    # Create comprehensive label
    label = " ".join(top_words)
    
    # Prepare supporting information
    supporting_info = {
        'tfidf_words': tfidf_words,
        'association_rules': association_rules,
        'word_frequencies': dict(word_counts),
        'top_integrated_words': top_words
    }
    
    return label, supporting_info


def main_text_mining_workflow(file_path, cluster_image_ids,
                             tfidf_top_n=5,
                             assoc_min_support=0.1,
                             assoc_min_confidence=0.5,
                             assoc_top_rules=3,
                             max_label_words=8):
    """
    Main text mining function that integrates TF-IDF and association rule approaches.
    
    This function performs the complete text mining workflow:
    1. Load the dataset
    2. Preprocess the text data
    3. Compute TF-IDF scores for the entire dataset
    4. Generate association rules for the cluster
    5. Combine results from both methods to create a comprehensive label
    6. Return the final label along with supporting information
    
    Args:
        file_path: Path to the dataset file
        cluster_image_ids: List of image IDs in the cluster
        tfidf_top_n: Number of top TF-IDF words to extract
        assoc_min_support: Minimum support for association rules
        assoc_min_confidence: Minimum confidence for association rules
        assoc_top_rules: Number of top association rules to use
        max_label_words: Maximum number of words in final label
        
    Returns:
        Dictionary containing:
        - 'label': Final descriptive label
        - 'supporting_info': Detailed information about the analysis
        - 'tfidf_results': TF-IDF specific results
        - 'association_results': Association rule specific results
    """
    try:
        # Step 1: Load the dataset
        print(f"Loading dataset from {file_path}")
        df = load_dataset(file_path)
        
        # Step 2: Preprocess the text data
        print("Preprocessing text data...")
        df = combine_and_preprocess(df)
        
        # Step 3: Compute TF-IDF scores for the entire dataset
        print("Computing TF-IDF scores...")
        vectorizer, tfidf_matrix = compute_tfidf_for_dataset(df)
        
        # Get top TF-IDF words for the cluster
        tfidf_words = get_cluster_tfidf_words(df, cluster_image_ids, vectorizer, tfidf_matrix, tfidf_top_n)
        
        # Step 4: Generate association rules for the cluster
        print("Generating association rules...")
        
        # Preprocess for association rules
        transactions = preprocess_for_association_rules(df, cluster_image_ids)
        
        association_results = {
            'transactions': transactions,
            'frequent_itemsets': None,
            'rules': None,
            'label': None
        }
        
        if len(transactions) > 0:
            # Find frequent itemsets
            frequent_itemsets = find_frequent_itemsets(transactions, assoc_min_support)
            association_results['frequent_itemsets'] = frequent_itemsets
            
            if len(frequent_itemsets) > 0:
                # Generate association rules
                rules = generate_association_rules(frequent_itemsets, len(transactions), assoc_min_confidence)
                association_results['rules'] = rules
                
                if len(rules) > 0:
                    # Extract top rules for labeling
                    top_rules = rules.head(assoc_top_rules)
                    rule_labels = []
                    for _, rule in top_rules.iterrows():
                        antecedents = ' '.join(list(rule['antecedents']))
                        consequents = ' '.join(list(rule['consequents']))
                        rule_labels.append(f"{antecedents} -> {consequents}")
                    
                    association_results['label'] = rule_labels
                else:
                    association_results['label'] = []
            else:
                association_results['label'] = []
        else:
            association_results['label'] = []
        
        # Step 5: Combine results from both methods
        print("Integrating results...")
        final_label, integration_info = integrate_text_mining_results(
            tfidf_words,
            association_results['label']
        )
        
        # Step 6: Return final results
        result = {
            'label': final_label,
            'supporting_info': {
                'tfidf_words': tfidf_words,
                'association_rules': association_results['label'],
                'integration_details': integration_info,
                'cluster_size': len(cluster_image_ids),
                'dataset_size': len(df)
            },
            'tfidf_results': {
                'vectorizer': vectorizer,
                'tfidf_matrix_shape': tfidf_matrix.shape,
                'top_words': tfidf_words
            },
            'association_results': {
                'transactions_count': len(transactions),
                'frequent_itemsets_count': len(association_results['frequent_itemsets']) if association_results['frequent_itemsets'] is not None else 0,
                'rules_count': len(association_results['rules']) if association_results['rules'] is not None else 0,
                'top_rules': association_results['label']
            }
        }
        
        print(f"Text mining completed. Final label: {final_label}")
        return result
        
    except Exception as e:
        print(f"Error in text mining workflow: {str(e)}")
        return {
            'error': str(e),
            'label': 'Error in processing',
            'supporting_info': {}
        }
