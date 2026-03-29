from pathlib import Path
from eda_helper import *


def main():
    print("\n\nEDA PROCESS STARTED")
    print("\n======================================================================\n\n")

    # STEP 0
    print("[STEP 0] STARTING.")
    project_root = Path(__file__).resolve().parent.parent
    dataset_info = zip_extraction(project_root)

    if dataset_info is None:
        return

    print("[STEP 0] DONE: Dataset is ready for EDA")
    print(f"[INFO] Dataset extracted to: {dataset_info['dataset_dir']}")
    print("\n\n======================================================================\n\n")
    
    # STEP 1
    print("[STEP 1] STARTING.")
    generate_basic_overview(dataset_info)
    print("[STEP 1] DONE: basic_overview.txt generated in ./results")
    print("\n\n======================================================================\n\n")

    # STEP 2
    print("[STEP 2] STARTING.")
    generate_integrity_check(dataset_info)
    print("[STEP 2] DONE: integrity_check.txt generated in ./results")
    print("\n\n======================================================================\n\n")


if __name__ == "__main__":
    main()