from pathlib import Path
from EDA.eda_helper import *


def pre_text():
    print("\nEDA PROCESS STARTED !!!")
    print("\n======================================================================")


def eda_step_0():
    print("\n[STEP 0] STARTING.")
    project_root = Path(__file__).resolve().parent.parent
    dataset_info = zip_extraction(project_root)

    if dataset_info is None:
        print("[ERROR] Dataset extraction failed. No such zip file found in the project root.")
        return None

    print(f"[INFO] Dataset extracted to: {dataset_info['dataset_dir']}")
    print("[STEP 0] DONE: Dataset is ready for EDA")
    print("\n======================================================================")
    return dataset_info


def eda_step_1(dataset_info=None):
    print("\n[STEP 1] STARTING.")

    if dataset_info is None:
        print("[INFO] No dataset_info provided, calling get_dataset_info to retrieve dataset details.")
        dataset_info = get_dataset_info(Path(__file__).resolve().parent.parent)

    generate_basic_overview(dataset_info)
    print("[STEP 1] DONE: basic_overview.txt generated in ./results")
    print("\n======================================================================")


def eda_step_2(dataset_info=None):
    print("\n[STEP 2] STARTING.")

    if dataset_info is None:
        print("[INFO] No dataset_info provided, calling get_dataset_info to retrieve dataset details.")
        dataset_info = get_dataset_info(Path(__file__).resolve().parent.parent)

    generate_integrity_check(dataset_info)
    print("[STEP 2] DONE: integrity_check.txt generated in ./results")
    print("\n======================================================================")


def eda_step_3(dataset_info=None):
    print("\n[STEP 3] STARTING.")

    if dataset_info is None:
        print("[INFO] No dataset_info provided, calling get_dataset_info to retrieve dataset details.")
        dataset_info = get_dataset_info(Path(__file__).resolve().parent.parent)

    generate_image_properties(dataset_info)
    print("[STEP 3] DONE: image_properties.txt generated in ./results")
    print("\n======================================================================")


def eda_step_4(dataset_info=None):
    print("\n[STEP 4] STARTING.")

    if dataset_info is None:
        print("[INFO] No dataset_info provided, calling get_dataset_info to retrieve dataset details.")
        dataset_info = get_dataset_info(Path(__file__).resolve().parent.parent)

    generate_similarity_leakage_check(dataset_info)
    print("[STEP 4] DONE: similarity_leakage_check.txt generated in ./results")
    print("\n======================================================================")


def main(step=None):
    pre_text()
    dataset_info = None  

    if step == "eda_full":
        dataset_info = eda_step_0()
        if dataset_info:
            eda_step_1(dataset_info)
            eda_step_2(dataset_info)
            eda_step_3(dataset_info)
            eda_step_4(dataset_info)

    elif step == "eda_step_0":
        eda_step_0()

    elif step == "eda_step_1":
        eda_step_1(dataset_info)

    elif step == "eda_step_2":
        eda_step_2(dataset_info)

    elif step == "eda_step_3":
        eda_step_3(dataset_info)

    elif step == "eda_step_4":
        eda_step_4(dataset_info)

    else:
        print("\n[ERROR] Invalid step specified.\n")