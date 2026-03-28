from pathlib import Path
import zipfile
import os


"""
Step 0:
- Check whether ./dataset/archive.zip exists
- If not, cancel EDA
- If yes, extract it into ./dataset
- Extraction will happen regardless of whether folders exist or not

Returns:
    dict | None
"""
def zip_extraction(project_root: Path):
    dataset_dir = project_root / "dataset"
    zip_path = dataset_dir / "archive.zip"

    # 1. Check zip file
    if not zip_path.exists():
        print(f"[ERROR] archive.zip not found: {zip_path}")
        print("[EDA] canceled.")
        return None

    # 2. Extract zip
    try:
        print(f"[INFO] Extracting: {zip_path}")
        with zipfile.ZipFile(zip_path, "r") as zip_ref:
            zip_ref.extractall(dataset_dir)
        print(f"[INFO] Extraction completed into: {dataset_dir}")

    except zipfile.BadZipFile:
        print(f"[ERROR] '{zip_path.name}' is not a valid ZIP file or is corrupted.")
        print("[EDA CANCELLED] Extraction failed.")
        return None
    
    except Exception as e:
        print(f"[ERROR] Failed to extract zip file: {e}")
        print("[EDA CANCELLED] Extraction failed.")
        return None

    seg_pred_dir = dataset_dir / "seg_pred" / "seg_pred"
    seg_test_dir = dataset_dir / "seg_test" / "seg_test"
    seg_train_dir = dataset_dir / "seg_train" / "seg_train"

    return {
        "dataset_dir": dataset_dir,
        "zip_path": zip_path,
        "seg_pred_dir": seg_pred_dir,
        "seg_test_dir": seg_test_dir,
        "seg_train_dir": seg_train_dir
    }



"""
Step 1:
- Generate an overview of the dataset.
- Save it to basic_overview.txt inside ./EDA/results.

Returns:
    None
"""
def generate_basic_overview(dataset_info: dict):
    dataset_dir = dataset_info["dataset_dir"]
    seg_pred_dir = dataset_info["seg_pred_dir"]
    seg_test_dir = dataset_info["seg_test_dir"]
    seg_train_dir = dataset_info["seg_train_dir"]

    # Check if results directory exists, if not, create it
    results_dir = dataset_info["dataset_dir"].parent / "EDA" / "results"
    if not results_dir.exists():
        os.makedirs(results_dir)
        print(f"[INFO] Created results directory: {results_dir}")

    # Dynamically discover class names from subfolders in seg_train/seg_train and seg_test/seg_test
    def get_class_names(folder_path):
        return [folder.name for folder in folder_path.iterdir() if folder.is_dir()]

    class_names = sorted(get_class_names(seg_train_dir))
    class_mapping = {class_name: index for index, class_name in enumerate(class_names)}
    
    # Initialize image counters and class counts
    total_images = 0
    split_counts = {"train": 0, "test": 0, "prediction": 0}
    class_counts_split = {split: {class_name: 0 for class_name in class_names} for split in split_counts}

    def count_images_in_folder(folder_path, split, is_classified=True):
        nonlocal total_images
        image_count = 0

        # Count images for each class in classified folders (train and test)
        if is_classified:
            for class_name in class_names:
                class_folder = folder_path / class_name
                if class_folder.exists():
                    class_image_count = len(list(class_folder.glob("*.jpg")))  # Assuming images are .jpg
                    image_count += class_image_count
                    class_counts_split[split][class_name] = class_image_count
        else:
            # For prediction folder, simply count all the images, no classification
            image_count = len(list(folder_path.glob("*.jpg")))  # Assuming images are .jpg
            split_counts[split] = image_count

        split_counts[split] = image_count
        total_images += image_count

    # Count images in each split
    count_images_in_folder(seg_train_dir, "train")
    count_images_in_folder(seg_test_dir, "test")
    count_images_in_folder(seg_pred_dir, "prediction", is_classified=False)

    # Prepare content for the basic_overview.txt
    overview_content = (
        f"Total number of images: {total_images}\n"
        f"Number of classes: {len(class_names)}\n"
        f"Class names: {class_names}\n\n"
    )

    overview_content += "Number of images in each split:\n"
    overview_content += f"  Train: {split_counts['train']}\n"
    overview_content += f"  Test: {split_counts['test']}\n"
    overview_content += f"  Prediction: {split_counts['prediction']}\n\n"

    overview_content += "Number of images in each class inside each split:\n"
    for split in split_counts:
        if (split == "train"):
            overview_content += f"  {split.capitalize()} Split:\n"
        else: 
            overview_content += f"\n  {split.capitalize()} Split:\n"
            
        for class_name in class_names:
            overview_content += f"    {class_name}: {class_counts_split[split][class_name]}\n"

    # Write the overview to basic_overview.txt
    basic_overview_path = results_dir / "basic_overview.txt"
    with open(basic_overview_path, "w") as f:
        f.write(overview_content)

    print(f"[INFO] Basic overview written to: {basic_overview_path}")