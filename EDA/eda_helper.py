from pathlib import Path
import zipfile
import hashlib
from PIL import Image
import os

def get_dataset_info(project_root: Path):
    dataset_dir = project_root / "dataset"
    zip_path = dataset_dir / "archive.zip"
    seg_pred_dir = dataset_dir / "seg_pred" / "seg_pred"
    seg_test_dir = dataset_dir / "seg_test" / "seg_test"
    seg_train_dir = dataset_dir / "seg_train" / "seg_train"

    # Check if the dataset is already extracted, if not, extract it
    if not zip_path.exists():
        print(f"[ERROR] {zip_path} not found. Please make sure the archive.zip exists.")
        return None
    
    # If the extracted folders don't exist, proceed to extract the zip
    if not (seg_pred_dir.exists() and seg_test_dir.exists() and seg_train_dir.exists()):
        print("[INFO] Dataset directories not found. Extracting the ZIP file...")
        dataset_info = zip_extraction(project_root)  # This function will handle extraction
        if dataset_info is None:
            return None

    return {
        "dataset_dir": dataset_dir,
        "zip_path": zip_path,
        "seg_pred_dir": seg_pred_dir,
        "seg_test_dir": seg_test_dir,
        "seg_train_dir": seg_train_dir
    }



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
    print("[INFO] Getting numbers of images and classes in the full dataset")
    overview_content = (
        f"Total number of images: {total_images}\n"
        f"Number of classes: {len(class_names)}\n"
        f"Class names: {class_names}\n\n"
    )

    print("[INFO] Getting numbers of images in each split")
    overview_content += "Number of images in each split:\n"
    overview_content += f"  Train: {split_counts['train']} ({(split_counts['train'] / total_images) * 100:.2f}%)\n"
    overview_content += f"  Test: {split_counts['test']} ({(split_counts['test'] / total_images) * 100:.2f}%)\n"
    overview_content += f"  Prediction: {split_counts['prediction']} ({(split_counts['prediction'] / total_images) * 100:.2f}%)\n\n"

    overview_content += "Number of images in each class inside each split:\n"
    for split in split_counts:
        if (split == "train"):
            overview_content += f"  {split.capitalize()} Split:\n"
        else: 
            overview_content += f"\n  {split.capitalize()} Split:\n"
            
        for class_name in class_names:
            class_count = class_counts_split[split][class_name]
            percentage = (class_count / total_images) * 100
            overview_content += f"    {class_name}: {class_count} ({percentage:.2f}%)\n"

    # Write the overview to basic_overview.txt
    basic_overview_path = results_dir / "basic_overview.txt"
    with open(basic_overview_path, "w") as f:
        f.write(overview_content)

    print(f"[INFO] Basic overview written to: {basic_overview_path}")



"""
Step 2:
- Check the integrity of the dataset.
- Detect corrupted images, unreadable files, empty files, wrong file extensions, duplicate images, duplicated filenames, and non-image files.
- Save the results to integrity_check.txt inside ./EDA/results.

Returns:
    None
"""
def generate_integrity_check(dataset_info: dict):
    from PIL import Image
    import hashlib

    seg_pred_dir = dataset_info["seg_pred_dir"]
    seg_test_dir = dataset_info["seg_test_dir"]
    seg_train_dir = dataset_info["seg_train_dir"]

    results_dir = dataset_info["dataset_dir"].parent / "EDA" / "results"
    if not results_dir.exists():
        os.makedirs(results_dir)

    integrity_check_path = results_dir / "integrity_check.txt"

    corrupted_files = []
    unreadable_files = []
    empty_files = []
    wrong_extension_files = []
    non_image_files = []

    duplicate_images = {}
    duplicated_filenames = {}

    valid_image_extensions = {".jpg", ".jpeg", ".png", ".bmp", ".gif", ".tiff", ".webp"}
    expected_extensions = {".jpg", ".jpeg"}

    def get_file_hash(file_path):
        sha256_hash = hashlib.sha256()
        with open(file_path, "rb") as f:
            for byte_block in iter(lambda: f.read(4096), b""):
                sha256_hash.update(byte_block)
        return sha256_hash.hexdigest()

    def is_image_readable(file_path):
        try:
            with Image.open(file_path) as img:
                img.load()
            return True, None
        except Exception as e:
            return False, str(e)

    def format_short_path(file_path):
        file_path = Path(file_path)

        if seg_train_dir in file_path.parents:
            rel = file_path.relative_to(seg_train_dir)
            if len(rel.parts) >= 2:
                class_name = rel.parts[0]
                filename = rel.name
                return f"Train ({class_name}): {filename}"
            return f"Train: {file_path.name}"

        elif seg_test_dir in file_path.parents:
            rel = file_path.relative_to(seg_test_dir)
            if len(rel.parts) >= 2:
                class_name = rel.parts[0]
                filename = rel.name
                return f"Test ({class_name}): {filename}"
            return f"Test: {file_path.name}"

        elif seg_pred_dir in file_path.parents:
            return f"Prediction: {file_path.name}"

        return str(file_path)

    def write_section(f, title, items, formatter=str):
        if not items:
            f.write(f"{title}: None\n\n")
        else:
            f.write(f"{title}:\n")
            for item in items:
                f.write(f"  {formatter(item)}\n")
            f.write("\n")

    def scan_folder(folder_path):
        for file_path in folder_path.rglob("*"):
            if not file_path.is_file():
                continue

            # duplicated filenames
            filename = file_path.name
            if filename not in duplicated_filenames:
                duplicated_filenames[filename] = []
            duplicated_filenames[filename].append(file_path)

            # empty files
            try:
                if file_path.stat().st_size == 0:
                    empty_files.append(file_path)
                    continue
            except Exception:
                unreadable_files.append(file_path)
                continue

            ext = file_path.suffix.lower()

            # non-image files
            if ext not in valid_image_extensions:
                non_image_files.append(file_path)
                continue

            # wrong file extensions
            if ext not in expected_extensions:
                wrong_extension_files.append(file_path)

            # corrupted / unreadable image
            is_ok, _ = is_image_readable(file_path)
            if not is_ok:
                corrupted_files.append(file_path)
                unreadable_files.append(file_path)
                continue

            # duplicate images by hash
            try:
                file_hash = get_file_hash(file_path)
                if file_hash not in duplicate_images:
                    duplicate_images[file_hash] = []
                duplicate_images[file_hash].append(file_path)
            except Exception:
                unreadable_files.append(file_path)

    print("[INFO] Scanning dataset folders for integrity check")
    scan_folder(seg_train_dir)
    scan_folder(seg_test_dir)
    scan_folder(seg_pred_dir)

    duplicate_image_groups = {
        file_hash: paths
        for file_hash, paths in duplicate_images.items()
        if len(paths) > 1
    }

    duplicated_filename_groups = {
        filename: paths
        for filename, paths in duplicated_filenames.items()
        if len(paths) > 1
    }

    with open(integrity_check_path, "w", encoding="utf-8") as f:
        write_section(f, "Corrupted Files", corrupted_files, format_short_path)
        write_section(f, "Unreadable Files", unreadable_files, format_short_path)
        write_section(f, "Empty Files", empty_files, format_short_path)
        write_section(f, "Wrong File Extensions", wrong_extension_files, format_short_path)

        if not duplicate_image_groups:
            f.write("Duplicate Images: None\n\n")
        else:
            f.write("Duplicate Images:\n")
            for file_hash, paths in duplicate_image_groups.items():
                f.write(f"  Duplicate Hash: {file_hash}\n")
                for path in paths:
                    f.write(f"  {format_short_path(path)}\n")
            f.write("\n")

        if not duplicated_filename_groups:
            f.write("Duplicated Filenames: None\n\n")
        else:
            f.write("Duplicated Filenames:\n")
            for filename, paths in duplicated_filename_groups.items():
                f.write(f"  Filename: {filename}\n")
                for path in paths:
                    f.write(f"  {format_short_path(path)}\n")
            f.write("\n")

        write_section(f, "Non-Image Files", non_image_files, format_short_path)

    print(f"[INFO] Integrity check results written to: {integrity_check_path}")