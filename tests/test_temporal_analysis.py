import pytest
import pandas as pd
import numpy as np
from unittest.mock import MagicMock, patch

from src.visualization import temporal_analysis


class TestTemporalAnalysis:

    @pytest.fixture
    def sample_df(self):
        # Create a sample dataframe with various dates
        dates = pd.to_datetime([
            '2023-01-01 10:00:00',
            '2023-01-01 11:00:00',
            '2023-01-02 10:00:00',
            '2023-06-15 15:30:00',
            '2023-12-31 23:59:59'
        ])

        data = {
            'date': dates,
            'hour': dates.hour,
            'date_taken_year': dates.year,
            'date_taken_month': dates.month,
            'date_taken_day': dates.day,
            'date_taken_hour': dates.hour,
            'title': ['Title1', 'Title2', 'Title3', 'Title4', 'Title5'],
            'description': ['Desc1', 'Desc2', 'Desc3', 'Desc4', 'Desc5'],
            'tags': ['tag1', 'tag2', 'tag3', 'tag4', 'tag5']
        }
        return pd.DataFrame(data)

    @patch('src.visualization.temporal_analysis.plt')
    @patch('src.visualization.temporal_analysis.LLMLabelingService')
    @patch('src.visualization.temporal_analysis.ConfigManager')
    def test_analyze_timestamps(
            self,
            mock_config,
            mock_llm,
            mock_plt,
            sample_df):
        # Setup mocks
        mock_fig = MagicMock()
        mock_ax = MagicMock()

        # Create a proper numpy array of objects for axes
        axes = np.empty((2, 2), dtype=object)
        axes[0, 0] = mock_ax
        axes[0, 1] = mock_ax
        axes[1, 0] = mock_ax
        axes[1, 1] = mock_ax

        mock_plt.subplots.return_value = (mock_fig, axes)

        # Test basic execution
        temporal_analysis.analyze_timestamps(sample_df)

        # Verify plotting calls
        assert mock_plt.subplots.called
        assert mock_plt.savefig.called

    @patch('src.visualization.temporal_analysis.plt')
    def test_analyze_timestamps_invalid_df(self, mock_plt):
        # Test with empty dataframe or invalid dates
        df = pd.DataFrame({'date': ['invalid']})
        temporal_analysis.analyze_timestamps(df)
        # Should handle gracefully, maybe print error but not crash

    def test_detect_events_by_entries(self, sample_df):
        # Create a df with a "event" spike
        dates = []
        # Normal traffic
        for i in range(10):
            dates.append(pd.Timestamp('2023-01-01') + pd.Timedelta(days=i))

        # Spike on day 20
        for i in range(10):
            dates.append(pd.Timestamp('2023-01-20'))

        df = pd.DataFrame({'date': dates,
                           'title': ['t'] * 20,
                           'description': ['d'] * 20,
                           'tags': ['t'] * 20})

        with patch('src.visualization.temporal_analysis.LLMLabelingService'):
            # We don't care about the output much, just that it runs and
            # detects something
            temporal_analysis.detect_events_by_entries(df)

    def test_missing_llm(self, sample_df):
        # Simulate LLM missing by patching
        with patch('src.visualization.temporal_analysis.LLM_AVAILABLE', False):
            temporal_analysis.detect_events_by_entries(sample_df)
