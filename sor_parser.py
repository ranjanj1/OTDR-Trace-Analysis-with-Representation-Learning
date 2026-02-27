import os
import json
import numpy as np
from typing import List
from pyotdr import sorparse

class Parser:
    def __init__(self, input_folder: str, output_folder: str):
        self.input_folder = input_folder
        self.output_folder = output_folder

        os.makedirs(output_folder, exist_ok=True)
        self.parsed_folder = os.path.join(output_folder, "parsed_folder")
        os.makedirs(self.parsed_folder, exist_ok=True)

    def parse_sor_file(self, file_path: str) -> dict:
        """Parse a single SOR file and return its data as a dictionary."""
        try:
            result = sorparse(file_path)
            return result
        except Exception as e:
            print(f"Error parsing {file_path}: {e}")
            return None

    def process_folder(self):
        """Process all SOR files in the input folder and save results."""
        sor_files = [
            f for f in os.listdir(self.input_folder)
            if f.lower().endswith(".sor")
        ]

        if not sor_files:
            print(f"No SOR files found in {self.input_folder}")
            return

        print(f"Found {len(sor_files)} SOR file(s). Processing...")

        for filename in sor_files:
            input_path = os.path.join(self.input_folder, filename)
            output_filename = os.path.splitext(filename)[0] + ".json"
            output_path = os.path.join(self.parsed_folder, output_filename)

            print(f"Processing: {filename}")
            parsed_data = self.parse_sor_file(input_path)

            if parsed_data is not None:
                self.save_to_json(parsed_data, output_path)
                print(f"  Saved -> {output_path}")
            else:
                print(f"  Skipped (parse error): {filename}")

        print("Done.")

    def save_to_json(self, data: dict, output_path: str):
        """Serialize and save parsed data to a JSON file."""
        def convert(obj):
            if isinstance(obj, np.ndarray):
                return obj.tolist()
            if isinstance(obj, (np.integer,)):
                return int(obj)
            if isinstance(obj, (np.floating,)):
                return float(obj)
            raise TypeError(f"Object of type {type(obj)} is not JSON serializable")

        with open(output_path, "w", encoding="utf-8") as f:
            json.dump(data, f, indent=2, default=convert)


if __name__ == "__main__":
    input_folder = "data/otdr_event_classification_training/2022-06-01_otdr_measurements"      # Folder containing .sor files
    output_folder = "./output"        # Folder to save parsed results

    parser = Parser(input_folder, output_folder)
    parser.process_folder()