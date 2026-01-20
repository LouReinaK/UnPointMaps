
from src.processing.dataset_filtering import filter_dataset
import unittest
import pandas as pd
import datetime
import sys
import os

# Add src to path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))


class TestDatasetFiltering(unittest.TestCase):
    def setUp(self):
        # Create a sample DataFrame
        self.data = {
            'id': ['1', '2', '3', '4'],
            # 3 is outside Lyon, 4 is NaN
            'latitude': [45.75, 45.76, 48.85, None],
            'longitude': [4.85, 4.86, 2.35, 4.85],
            'date': ['2023-01-01', '2023-01-02', '2023-01-03', '2023-01-04'],
            'title': ['T1', 'T2', 'T3', 'T4'],
            'tags': ['tag1', 'tag2', 'tag3', 'tag4'],
            'user': ['u1', 'u1', 'u2', 'u3']
        }
        self.df = pd.DataFrame(self.data)

    def test_filter_dataset_basics(self):
        """Test basic filtering: cleaning dates, coordinates, NaN"""
        df_filtered, df_grouped = filter_dataset(self.df)

        # Expecting point 3 (Paris) and 4 (NaN) to be removed
        # Lyon coords roughly 45.67-45.82 lat, 4.75-4.98 lon

        self.assertEqual(len(df_filtered), 2)
        self.assertTrue('1' in df_filtered['id'].values)
        self.assertTrue('2' in df_filtered['id'].values)
        self.assertFalse('3' in df_filtered['id'].values)  # Outside Lyon

        # Check grouping
        self.assertGreaterEqual(len(df_grouped), 1)

    def test_filter_dataset_missing_columns(self):
        """Test robust handling if optional columns missing"""
        # Remove 'user'
        df_missing = self.df.drop(columns=['user'])
        df_filtered, df_grouped = filter_dataset(df_missing)
        self.assertEqual(len(df_filtered), 2)

    def test_filter_dataset_future_dates(self):
        """Test filtering of future dates"""
        future_date = datetime.datetime.now() + datetime.timedelta(days=365)
        self.df.loc[0, 'date'] = future_date.strftime('%Y-%m-%d')

        df_filtered, df_grouped = filter_dataset(self.df)
        # Point 1 should be removed
        self.assertFalse('1' in df_filtered['id'].values)

    def test_filter_duplicates(self):
        """Test duplicate removal"""
        # Duplicate row 2
        new_row = self.df.iloc[1:2].copy()
        df_duped = pd.concat([self.df, new_row], ignore_index=True)

        df_filtered, df_grouped = filter_dataset(df_duped)
        # Should still be 2 (assuming valid ones)
        # Points 1, 2 are valid. 3 is outside. 4 is NaN.
        # Duplicate of 2 should be removed or collapsed.

        # filter_dataset removes duplicates
        self.assertEqual(len(df_filtered), 2)


if __name__ == '__main__':
    unittest.main()
