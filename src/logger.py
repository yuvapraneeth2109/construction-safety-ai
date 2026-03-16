import pandas as pd
from datetime import datetime
import os
import time


class SafetyLogger:
    def __init__(self, log_interval=1.0):
        """
        log_interval: seconds between logs (default = 1 second)
        """
        self.records = []
        self.last_log_time = 0
        self.log_interval = log_interval

    def log(self, detections, violations):
        """
        Logs one clean record per interval.
        """

        current_time = time.time()

        # Throttle logging to once per interval
        if current_time - self.last_log_time < self.log_interval:
            return

        self.last_log_time = current_time

        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

        record = {
            "timestamp": timestamp,
            "violations": ", ".join(violations) if violations else "None",
            "objects_detected": len(detections)
        }

        self.records.append(record)

    def export_excel(self, filepath="logs/safety_report.xlsx"):
        """
        Exports Excel file with:
        - Summary sheet
        - Detailed log sheet
        """

        if not os.path.exists("logs"):
            os.makedirs("logs")

        df = pd.DataFrame(self.records)

        if df.empty:
            # Create empty structure
            df = pd.DataFrame(columns=["timestamp", "violations", "objects_detected"])

        total_seconds = len(df)

        violation_flags = df["violations"] != "None"

        true_positive_seconds = 0
        false_positive_seconds = 0

        consecutive_count = 0

        for flag in violation_flags:
            if flag:
                consecutive_count += 1
            else:
                if consecutive_count == 1:
                    false_positive_seconds += 1
                elif consecutive_count >= 2:
                    true_positive_seconds += consecutive_count
                consecutive_count = 0

        # Handle last streak
        if consecutive_count == 1:
            false_positive_seconds += 1
        elif consecutive_count >= 2:
            true_positive_seconds += consecutive_count

        safe_seconds = total_seconds - (true_positive_seconds + false_positive_seconds)

        summary_data = {
            "Metric": [
                "Total Logged Seconds",
                "True Positive Seconds",
                "False Positive Seconds",
                "Safe Seconds"
            ],
            "Value": [
                total_seconds,
                true_positive_seconds,
                false_positive_seconds,
                safe_seconds
            ]
        }

        summary_df = pd.DataFrame(summary_data)

        with pd.ExcelWriter(filepath, engine="openpyxl") as writer:
            summary_df.to_excel(writer, index=False, sheet_name="Summary")
            df.to_excel(writer, index=False, sheet_name="Detailed_Log")

        return filepath

    def get_dataframe(self):
        return pd.DataFrame(self.records)
