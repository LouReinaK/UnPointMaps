import pandas as pd
try:
    import matplotlib.pyplot as plt
except ImportError:
    plt = None
import os
from datetime import datetime
from dotenv import load_dotenv
from typing import Optional
from src.processing.remove_nonsignificative_words import clean_text_list

# Import LLM Labeling components
try:
    from src.processing.llm_labelling import LLMLabelingService, ConfigManager
    LLM_AVAILABLE = True
except ImportError:
    LLM_AVAILABLE = False
    print("Warning: LLM Labeling modules not found. Labeling will be skipped.")


def analyze_timestamps(df: Optional[pd.DataFrame] = None):
    """
    Main function to analyze temporal distribution of the dataset.
    If df is provided, uses it. Only expects 'date' and optionally 'hour', 'date_taken_hour' columns.
    If df is None, tries to load from file.
    """
    if plt is None:
        print("Error: 'matplotlib' library is missing. Cannot generate temporal analysis plots. Please install it with 'pip install matplotlib'.")
        return

    if df is None:
        load_dotenv()
        file_path = 'flickr_data2.csv'
        if not os.path.exists(file_path):
            print(f"Error: {file_path} not found.")
            return

        print("Loading dataset...")
        # Load specific columns including hour and text metadata for labeling.
        try:
            # Note: Original columns have spaces
            cols = [
                ' date_taken_year',
                ' date_taken_month',
                ' date_taken_day',
                ' date_taken_hour',
                ' id',
                ' tags',
                ' title',
                ' latitude',
                ' longitude']
            df = pd.read_csv(
                file_path,
                usecols=lambda c: c in cols,
                low_memory=False)
        except ValueError:
            print("Specific columns not found, loading header to inspect...")
            df = pd.read_csv(file_path, low_memory=False)

        print(f"Loaded {len(df)} rows.")

        # Clean column names
        df.columns = df.columns.str.strip()

        # Clean data (remove non-numeric)
        # Note: 'date_taken_hour' also needs cleaning.
        date_columns = [
            'date_taken_year',
            'date_taken_month',
            'date_taken_day']
        if 'date_taken_hour' in df.columns:
            date_columns.append('date_taken_hour')

        for col in date_columns:
            if col in df.columns:
                df[col] = df[col].astype(str).str.replace(
                    r'[^0-9]', '', regex=True)

        # Construct datetime if not present
        if 'date' not in df.columns:
            print("Converting timestamps...")
            # Map to names expected by pd.to_datetime: year, month, day
            df_dates = df[['date_taken_year', 'date_taken_month', 'date_taken_day']].rename(columns={
                'date_taken_year': 'year',
                'date_taken_month': 'month',
                'date_taken_day': 'day'
            })

            # coerce to numeric, dropping failures
            for c in ['year', 'month', 'day']:
                df_dates[c] = pd.to_numeric(df_dates[c], errors='coerce')

            df['date'] = pd.to_datetime(df_dates, errors='coerce')

    # Ensure 'date' column is datetime
    if 'date' in df.columns and not pd.api.types.is_datetime64_any_dtype(
            df['date']):
        df['date'] = pd.to_datetime(df['date'], errors='coerce')

    # Ensure hour column
    if 'hour' not in df.columns:
        if 'date_taken_hour' in df.columns:
            df['hour'] = pd.to_numeric(df['date_taken_hour'], errors='coerce')
        elif 'date' in df.columns:
            df['hour'] = df['date'].dt.hour
        else:
            df['hour'] = pd.Series([None] * len(df))

    # Drop invalid dates

    initial_count = len(df)
    df = df.dropna(subset=['date'])
    dropped_count = initial_count - len(df)
    if dropped_count > 0:
        print(
            f"Dropped {dropped_count} rows with invalid dates. Remaining: {len(df)}")

    if len(df) == 0:
        print("No valid dates found to plot.")
        return

    # --- PLOTTING ---
    print("Generating combined plots...")

    # Create a 2x2 grid of subplots

    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    fig.suptitle('Temporal Analysis of Flickr Dataset', fontsize=16)

    # 1. Overall Histogram (Log Scale) - Top Left
    ax1 = axes[0, 0]
    ax1.hist(df['date'], bins=50, edgecolor='black', alpha=0.7, log=True)
    ax1.set_title('Distribution of Timestamps (Log Scale)')
    ax1.set_xlabel('Date')
    ax1.set_ylabel('Frequency (Log Scale)')
    ax1.grid(axis='y', alpha=0.5, which='both')
    ax1.tick_params(axis='x', rotation=45)

    # 2. Time of Day Histogram - Top Right
    ax2 = axes[0, 1]
    # Filter valid hours (0-23) just in case
    valid_hours = df['hour'].dropna()
    valid_hours = valid_hours[(valid_hours >= 0) & (valid_hours <= 23)]

    ax2.hist(valid_hours, bins=24, range=(0, 24), edgecolor='black', alpha=0.7)
    ax2.set_title('Distribution by Hour of Day')
    ax2.set_xlabel('Hour (0-23)')
    ax2.set_ylabel('Frequency')
    ax2.set_xticks(range(0, 25, 2))  # Ticks every 2 hours
    ax2.grid(axis='y', alpha=0.5)

    # 3. Day of Year Histogram - Bottom Left
    ax3 = axes[1, 0]
    df['day_of_year'] = df['date'].dt.dayofyear
    ax3.hist(
        df['day_of_year'],
        bins=366,
        range=(
            1,
            366),
        edgecolor='none',
        alpha=0.7)
    ax3.set_title('Distribution by Day of Year')
    ax3.set_xlabel('Day of Year')
    ax3.set_ylabel('Frequency')

    month_starts = [1, 32, 60, 91, 121, 152, 182, 213, 244, 274, 305, 335]
    month_names = [
        'Jan',
        'Feb',
        'Mar',
        'Apr',
        'May',
        'Jun',
        'Jul',
        'Aug',
        'Sep',
        'Oct',
        'Nov',
        'Dec']
    ax3.set_xticks(month_starts)
    ax3.set_xticklabels(month_names)
    ax3.grid(axis='y', alpha=0.5)

    # 4. Week of Year Histogram - Bottom Right
    ax4 = axes[1, 1]
    df['week_of_year'] = df['date'].dt.isocalendar().week
    ax4.hist(
        df['week_of_year'],
        bins=53,
        range=(
            1,
            53),
        edgecolor='black',
        alpha=0.7)
    ax4.set_title('Distribution by Week of Year')
    ax4.set_xlabel('Week Number')
    ax4.set_ylabel('Frequency')
    ax4.grid(axis='y', alpha=0.5)

    # Adjust layout
    plt.tight_layout(rect=[0, 0.03, 1, 0.95])  # Adjust for suptitle

    output_file = 'combined_timestamps_analysis.png'
    # Use global plt to save
    plt.savefig(output_file)
    print(f"Combined analysis saved to {output_file}")

    # Detect and print top events
    detect_events_by_entries(df)

    # Try to show
    try:
        pass
        # plt.show() # Can block execution, better to just save
    except Exception:
        pass


def detect_events_by_entries(df):
    print("\n--- Top 10 Events Detection & Labeling ---")

    # Initialize Labeler if available
    labeler = None
    if LLM_AVAILABLE:
        try:
            config_manager = ConfigManager()
            labeler = LLMLabelingService(config_manager)
            print("LLM Labeler initialized.")
        except Exception as e:
            print(f"Failed to initialize LLM Labeler: {e}")

    # 1. Count entries per day of year
    # Ensure all days 1-366 range present
    daily_counts = df['date'].dt.dayofyear.value_counts().sort_index()
    daily_counts = daily_counts.reindex(range(1, 367), fill_value=0)

    # 2. Define threshold (Mean + 1.0 STD)
    threshold = daily_counts.mean() + 1.0 * daily_counts.std()
    print(
        f"Threshold for event days: {threshold:.2f} entries (Mean: {daily_counts.mean():.2f}, Std: {daily_counts.std():.2f})")

    # 3. Identify days above threshold
    high_activity_days = daily_counts[daily_counts > threshold].index.tolist()

    # 4. Group consecutive days
    events = []
    if not high_activity_days:
        print("No events found above threshold.")
        return

    current_event_days = [high_activity_days[0]]

    for day in high_activity_days[1:]:
        if day == current_event_days[-1] + 1:
            current_event_days.append(day)
        else:
            # End of current event
            events.append(current_event_days)
            current_event_days = [day]
    events.append(current_event_days)  # Append last one

    # 5. Measure events
    event_stats = []
    dummy_year = 2024  # Leap year to handle day 366

    for days in events:
        total_entries = daily_counts.loc[days].sum()
        peak_day = daily_counts.loc[days].idxmax()
        peak_entries = daily_counts.loc[peak_day]

        # Format dates
        start_date = datetime(dummy_year, 1, 1) + \
            pd.Timedelta(days=days[0] - 1)
        end_date = datetime(dummy_year, 1, 1) + pd.Timedelta(days=days[-1] - 1)

        event_stats.append({
            'start_day': days[0],
            'end_day': days[-1],
            'days': days,  # store list of day numbers
            'start_date_str': start_date.strftime('%B %d'),
            'end_date_str': end_date.strftime('%B %d'),
            'total_entries': total_entries,
            'peak_entries': peak_entries,
            'duration': len(days)
        })

    # 6. Sort by total entries
    sorted_events = sorted(
        event_stats,
        key=lambda x: x['total_entries'],
        reverse=True)

    # 7. Print Top 10 and Label
    for i, event in enumerate(sorted_events[:10], 1):
        date_str = event['start_date_str']
        if event['start_day'] != event['end_day']:
            date_str += f" - {event['end_date_str']}"

        print(f"#{i}: {date_str} ({event['duration']} days)")
        print(f"    Total Entries: {event['total_entries']}")
        print(f"    Peak: {event['peak_entries']} entries")

        # --- LLM Labeling ---
        if labeler is not None:
            try:
                # Extract text data for this event
                # Filter df for these days of year
                # Note: this ignores year, as requested (aggregated recurring
                # events)
                event_mask = df['date'].dt.dayofyear.isin(event['days'])
                event_df = df[event_mask]

                # Sample text metadata to avoid token limits
                sample_size = 30
                if len(event_df) > sample_size:
                    sample_df = event_df.sample(sample_size)
                else:
                    sample_df = event_df

                # Combine title and tags
                text_metadata = []
                image_ids = []
                for _, row in sample_df.iterrows():
                    parts = []
                    if pd.notna(
                            row.get('title')) and str(
                            row['title']).strip():
                        parts.append(f"Title: {str(row['title']).strip()}")
                    if pd.notna(row.get('tags')) and str(row['tags']).strip():
                        parts.append(f"Tags: {str(row['tags']).strip()}")

                    if parts:
                        text_metadata.append(" | ".join(parts))
                        image_ids.append(str(row.get('id', 'unknown')))

                # Clean text
                text_metadata = clean_text_list(text_metadata)

                if not text_metadata:
                    print("    Label: [No text metadata available]")
                    continue

                cluster_metadata = {
                    "cluster_id": f"event_{i}_{event['start_day']}",
                    "image_ids": image_ids,
                    "text_metadata": text_metadata,
                    "cluster_size": len(event_df)
                }

                # Generate label
                print("    Generating label...", end="", flush=True)
                result = labeler.generate_cluster_label(cluster_metadata)
                print(
                    f"\r    Label: {result.label} (Conf: {result.confidence:.2f})")

            except Exception as e:
                print(f"\r    Labeling failed: {e}")
                import traceback
                traceback.print_exc()


if __name__ == "__main__":
    analyze_timestamps()
