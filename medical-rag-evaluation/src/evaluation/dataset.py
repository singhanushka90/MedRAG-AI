import json
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[2]

DATASET_PATH = PROJECT_ROOT / "datasets" / "golden_dataset.json"


def load_golden_dataset():

    with open(DATASET_PATH, "r", encoding="utf-8") as file:

        dataset = json.load(file)

    print(f"Total test cases loaded: {len(dataset)}")

    return dataset