import pandas as pd
import numpy as np
import os
import json
import logging
from datetime import datetime
from typing import List, Dict, Optional, Tuple
from src.processing.llm_labelling import LLMLabelingService, ConfigManager
from src.database.manager import DatabaseManager

# Configure logger
logger = logging.getLogger('time_filtering')
if not logger.handlers:
    logger.setLevel(logging.INFO)
    handler = logging.StreamHandler()
    formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    handler.setFormatter(formatter)
    logger.addHandler(handler)

class TimeFilter:
    def __init__(self):
        self.db = DatabaseManager()
        
        # Initialize LLM Labeler if possible
        self.labeler = None
        try:
            config_manager = ConfigManager()
            self.labeler = LLMLabelingService(config_manager)
            logger.info("LLM Labeler initialized for event detection.")
        except Exception as e:
            logger.warning(f"LLM Labeling disabled: {e}")

    def detect_events(self, df: pd.DataFrame, force_refresh: bool = False) -> List[Dict]:
        """
        Detects recurring events (ignoring year) that have high activity.
        Uses caching to avoid re-labeling.
        """
        # Check cache first
        if not force_refresh:
            cached_events = self.db.get_cached_events(len(df))
            if cached_events:
                logger.info(f"Loaded {len(cached_events)} detected events from DB cache.")
                return cached_events

        logger.info("Detecting events from scratch...")
        
        # 1. Count entries per day of year
        # Ensure 'date' column is datetime
        if not pd.api.types.is_datetime64_any_dtype(df['date']):
             df['date'] = pd.to_datetime(df['date'], errors='coerce')

        daily_counts = df['date'].dt.dayofyear.value_counts().sort_index()
        daily_counts = daily_counts.reindex(range(1, 367), fill_value=0)

        # 2. Define threshold (Mean + 1.0 STD)
        threshold = daily_counts.mean() + 1.0 * daily_counts.std()
        
        # 3. Identify days above threshold
        high_activity_days = daily_counts[daily_counts > threshold].index.tolist()
        
        # 4. Group consecutive days
        raw_events = []
        if high_activity_days:
            current_event_days = [high_activity_days[0]]
            for day in high_activity_days[1:]:
                if day == current_event_days[-1] + 1:
                    current_event_days.append(day)
                else:
                    raw_events.append(current_event_days)
                    current_event_days = [day]
            raw_events.append(current_event_days)

        # 5. Measure and Label events
        events = []
        dummy_year = 2024 # Leap year
        
        for i, days in enumerate(raw_events, 1):
            total_entries = int(daily_counts.loc[days].sum()) # Convert to int for JSON serializable
            
            # Format dates
            start_date = datetime(dummy_year, 1, 1) + pd.Timedelta(days=days[0]-1)
            end_date = datetime(dummy_year, 1, 1) + pd.Timedelta(days=days[-1]-1)
            
            event_info = {
                'id': i,
                'start_day': int(days[0]),
                'end_day': int(days[-1]),
                'days': [int(d) for d in days],
                'start_date_str': start_date.strftime('%B %d'),
                'end_date_str': end_date.strftime('%B %d'),
                'total_entries': total_entries,
                'label': "Unknown Event"
            }
            
            # Label with LLM
            if self.labeler:
                label = self._generate_label_for_days(df, days, i)
                if label:
                    event_info['label'] = label
            
            events.append(event_info)

        # Sort by total entries
        events = sorted(events, key=lambda x: x['total_entries'], reverse=True)
        
        # Save cache
        try:
            self.db.save_events(len(df), events)
            logger.info(f"Saved {len(events)} events to cache.")
        except Exception as e:
            logger.error(f"Failed to save cache: {e}")

        return events

    def _generate_label_for_days(self, df: pd.DataFrame, days: List[int], event_index: int) -> Optional[str]:
        """Helper to sample data and call LLM"""
        try:
            # Filter df for these days of year
            event_mask = df['date'].dt.dayofyear.isin(days)
            event_df = df[event_mask]
            
            if len(event_df) == 0:
                return None

            # Sample text metadata
            sample_size = 25
            if len(event_df) > sample_size:
                sample_df = event_df.sample(sample_size)
            else:
                sample_df = event_df
            
            text_metadata = []
            image_ids = []
            for _, row in sample_df.iterrows():
                parts = []
                if pd.notna(row.get('title')) and str(row['title']).strip():
                    parts.append(f"Title: {str(row['title']).strip()}")
                if pd.notna(row.get('tags')) and str(row['tags']).strip():
                    parts.append(f"Tags: {str(row['tags']).strip()}")
                
                if parts:
                    text_metadata.append(" | ".join(parts))
                    image_ids.append(str(row.get('id', 'unknown')))

            if not text_metadata:
                return "Unlabeled Cluster (No Text)"

            cluster_metadata = {
                "cluster_id": f"event_{event_index}_{days[0]}",
                "image_ids": image_ids,
                "text_metadata": text_metadata,
                "cluster_size": len(event_df)
            }
            
            result = self.labeler.generate_cluster_label(cluster_metadata)
            return result.label
            
        except Exception as e:
            logger.error(f"Label generation error: {e}")
            return None

    def filter_by_event(self, df: pd.DataFrame, event: Dict) -> pd.DataFrame:
        """Keep only data corresponding to the event days (recurring every year)"""
        logger.info(f"Filtering for event: {event['label']} (Days {event['start_day']}-{event['end_day']})")
        return df[df['date'].dt.dayofyear.isin(event['days'])]

    def exclude_events(self, df: pd.DataFrame, events_to_exclude: List[Dict]) -> pd.DataFrame:
        """Remove data corresponding to the specified events"""
        if not events_to_exclude:
            return df
            
        days_to_exclude = []
        labels = []
        for e in events_to_exclude:
            days_to_exclude.extend(e['days'])
            labels.append(e['label'])
            
        logger.info(f"Excluding events: {', '.join(labels)}")
        return df[~df['date'].dt.dayofyear.isin(days_to_exclude)]

    def filter_by_time_range(self, df: pd.DataFrame, start_year: int, end_year: int) -> pd.DataFrame:
        """Filter by year range (inclusive)"""
        logger.info(f"Filtering years {start_year}-{end_year}")
        return df[(df['date'].dt.year >= start_year) & (df['date'].dt.year <= end_year)]

    def filter_by_hour(self, df: pd.DataFrame, start_hour: int, end_hour: int) -> pd.DataFrame:
        """Filter by hour range (inclusive)"""
        logger.info(f"Filtering hours {start_hour}-{end_hour}")
        if 'hour' not in df.columns:
            logger.warning("'hour' column missing, attempting to create it.")
            df['hour'] = df['date_taken_hour'].astype(float)
            
        return df[(df['hour'] >= start_hour) & (df['hour'] <= end_hour)]
