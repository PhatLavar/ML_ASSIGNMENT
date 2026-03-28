from pathlib import Path
from eda_helper import *


def main():
    # STEP 0
    project_root = Path(__file__).resolve().parent.parent
    dataset_info = zip_extraction(project_root)

    if dataset_info is None:
        return

    print("[STEP 0] DONE: Dataset is ready for EDA.")
    print(f"[INFO] Dataset extracted to: {dataset_info['dataset_dir']}")
    
    # STEP 1
    generate_basic_overview(dataset_info)
    print("[STEP 1] DONE: basic_overview.txt generated in ./results")


if __name__ == "__main__":
    main()