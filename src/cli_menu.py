from typing import Optional
import sys
import pandas as pd
from src.processing.time_filtering import TimeFilter
from src.visualization.temporal_analysis import analyze_timestamps


class CLIMenu:
    def __init__(self):
        self.time_filter = TimeFilter()
        self.events = []

    def run(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Main entry point for the interactive menu.
        """
        while True:
            print("\n=== UnPointMaps Filter Selection ===")
            print(f"Current Dataset Size: {len(df)} entries")
            print("1. Standard Run (Full Dataset / Custom Exclusions)")
            print("2. Isolate a Top Event")
            print("3. Filter by Year Range")
            print("4. Filter by Time of Day")
            print("5. Generate Temporal Analysis Report (Plots)")
            print("0. Exit Application")

            choice = input("> Select option: ").strip()

            if choice == '0':
                print("Exiting...")
                sys.exit(0)

            elif choice == '1':
                # Standard Run with optional exclusion
                return self._handle_standard_run(df)

            elif choice == '2':
                # Isolate Event
                result_df = self._handle_isolate_event(df)
                if result_df is not None:
                    return result_df

            elif choice == '3':
                # Year Range
                result_df = self._handle_year_filter(df)
                if result_df is not None:
                    return result_df

            elif choice == '4':
                # Time of Day
                result_df = self._handle_hour_filter(df)
                if result_df is not None:
                    return result_df

            elif choice == '5':
                # Temporal Analysis
                print("\nRunning Temporal Analysis on current dataset...")
                analyze_timestamps(df)
                input("\nPress Enter to return to menu...")

            else:
                print("Invalid option. Please try again.")

    def _ensure_events_loaded(self, df: pd.DataFrame):
        if not self.events:
            print("\n--- Detecting Events (this may take a moment first time) ---")
            self.events = self.time_filter.detect_events(df)

    def _ask_exclude_big_events(self, df: pd.DataFrame) -> pd.DataFrame:
        """Helper to ask user if they want to exclude big events."""
        self._ensure_events_loaded(df)

        # Display top 5 just so they know what "Big Events" are
        print("\nTop detected events:")
        for i, event in enumerate(self.events[:5], 1):
            print(
                f"  #{i}: {event['label']} ({event['start_date_str']} - {event['end_date_str']}) - {event['total_entries']} entries")

        choice = input(
            "\nDo you want to exclude major events to avoid skewing? [y/N]: ").strip().lower()
        if choice == 'y':
            print(
                "Select which events to exclude (comma separated numbers, e.g. '1, 2'), or 'all' for top 5:")
            selection = input("> ").strip().lower()

            events_to_exclude = []
            if selection == 'all':
                events_to_exclude = self.events[:5]
            else:
                try:
                    indices = [int(x.strip())
                               for x in selection.split(',') if x.strip()]
                    for idx in indices:
                        if 1 <= idx <= len(self.events):
                            events_to_exclude.append(self.events[idx - 1])
                except ValueError:
                    print("Invalid input, no events excluded.")

            if events_to_exclude:
                df = self.time_filter.exclude_events(df, events_to_exclude)
                print(f"Events excluded. New size: {len(df)}")

        return df

    def _handle_standard_run(self, df: pd.DataFrame) -> pd.DataFrame:
        # Just ask for exclusions
        return self._ask_exclude_big_events(df)

    def _handle_isolate_event(self,
                              df: pd.DataFrame) -> Optional[pd.DataFrame]:
        self._ensure_events_loaded(df)

        print("\n--- Top Detected Events ---")
        for i, event in enumerate(self.events[:10], 1):
            print(f"#{i}: {event['label']}")
            print(
                f"    {event['start_date_str']} - {event['end_date_str']} ({event['total_entries']} entries)")

        choice = input(
            "\nSelect event number to isolate (or 'c' to cancel): ").strip()
        if choice.lower() == 'c':
            return None

        try:
            idx = int(choice)
            if 1 <= idx <= len(self.events):
                event = self.events[idx - 1]
                return self.time_filter.filter_by_event(df, event)
            else:
                print("Invalid number.")
        except ValueError:
            print("Invalid input.")
        return None

    def _handle_year_filter(self, df: pd.DataFrame) -> Optional[pd.DataFrame]:
        print("\n--- Filter by Year ---")
        min_year = int(df['date'].dt.year.min())
        max_year = int(df['date'].dt.year.max())
        print(f"Dataset range: {min_year} - {max_year}")

        try:
            start = input(f"Start Year [{min_year}]: ").strip()
            end = input(f"End Year [{max_year}]: ").strip()

            start_year = int(start) if start else min_year
            end_year = int(end) if end else max_year

            df_filtered = self.time_filter.filter_by_time_range(
                df, start_year, end_year)
            print(f"Filtered to {len(df_filtered)} entries.")

            # Optional: Ask to exclude big events from this subsets
            return self._ask_exclude_big_events(df_filtered)

        except ValueError:
            print("Invalid year input.")
            return None

    def _handle_hour_filter(self, df: pd.DataFrame) -> Optional[pd.DataFrame]:
        print("\n--- Filter by Hour of Day ---")
        print("Format: 0-23")

        try:
            start = input("Start Hour [0]: ").strip()
            end = input("End Hour [23]: ").strip()

            start_hour = int(start) if start else 0
            end_hour = int(end) if end else 23

            df_filtered = self.time_filter.filter_by_hour(
                df, start_hour, end_hour)
            print(f"Filtered to {len(df_filtered)} entries.")

            return self._ask_exclude_big_events(df_filtered)

        except ValueError:
            print("Invalid hour input.")
            return None
