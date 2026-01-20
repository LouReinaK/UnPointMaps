
from src.processing.time_filtering import TimeFilter
import unittest
import pandas as pd
import sys
import os
from unittest.mock import MagicMock

# Add src to path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))


class TestTimeFiltering(unittest.TestCase):
    def setUp(self):
        self.time_filter = TimeFilter()
        # Mock DB manager and LLM service to avoid external calls
        self.time_filter.db_manager = MagicMock()
        self.time_filter.llm_service = MagicMock()

    def test_exclude_events(self):
        """Test excluding events from dataframe"""
        # Create a dataframe with data crossing even time ranges
        data = {
            'date': pd.to_datetime(['2023-01-01 10:00', '2023-01-01 11:00', '2023-01-02 10:00']),
            'latitude': [45.0, 45.0, 45.0],
            'longitude': [4.0, 4.0, 4.0],
            'description': ['A', 'B', 'C']
        }
        df = pd.DataFrame(data)

        event_to_exclude = {
            'start_date': pd.to_datetime('2023-01-01 00:00'),
            'end_date': pd.to_datetime('2023-01-01 23:59'),
            'label': 'Event 1',
            'days': [1]  # Jan 1 is day-of-year 1
        }

        filtered_df = self.time_filter.exclude_events(df, [event_to_exclude])

        # Should exclude first 2 rows
        self.assertEqual(len(filtered_df), 1)
        self.assertEqual(
            filtered_df.iloc[0]['date'],
            pd.to_datetime('2023-01-02 10:00'))


if __name__ == '__main__':
    unittest.main()
