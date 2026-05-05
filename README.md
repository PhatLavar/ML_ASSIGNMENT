# CO3117 - Machine Learning Project (Semester 252, AY 2025-2026)

## 📌 Course Information
- **Course Name:** Machine Learning
- **Course Code:** CO3117
- **Semester:** 252
- **Academic Year:** 2025-2026

## 👨‍🏫 Instructor
- **Dr. Trương Vĩnh Lân**

## 👥 Team Members

| Name | Student ID | Role |
| :--- | :--- | :--- |
| **Nguyễn Văn An** | 2352013 | Leader |
| **Huỳnh Vương Khang** | 2350011 | Member |
| **Nguyễn Tấn Phát** | 2352889 | Member |
| **Lê Đặng Khánh Quỳnh** | 2353037 | Member |

## 🎯 Project Objectives
This project aims to build a comprehensive end-to-end image classification pipeline using the **Intel Image Classification** dataset. The key objectives are:
- **Automated Data Processing:** Implementing automated download, extraction, data auditing, and cleaning (including corrupt file removal, resolution validation, and duplicate detection via perceptual hashing).
- **Dataset Standardization:** Establishing a robust and standardized train/validation/test split alongside comprehensive metadata tracking.
- **Traditional Machine Learning:** Experimenting with handcrafted feature extraction techniques (HOG, SIFT with Bag-of-Visual-Words) coupled with traditional classifiers (Linear SVM, Logistic Regression).
- **Deep Learning Feature Extraction:** Leveraging transfer learning using a pre-trained ResNet18 model for advanced feature embedding, followed by classification using traditional algorithms.
- **Model Evaluation:** Conducting quantitative performance evaluations utilizing accuracy metrics, weighted F1-scores, detailed classification reports, and confusion matrices.

## 🚀 Instructions for Running the Notebook

### Required Libraries
Ensure the following libraries are installed (detailed in `requirements.txt`):
- `numpy`, `pandas`, `scikit-learn`
- `matplotlib`, `seaborn`
- `torch`, `torchvision`
- `Pillow`, `ImageHash`

### How to Download Data and Run the Code

**Option A: Import this repository to Google Colab**
1. Open up a new Google Colab notebook and clone the repository directly inside the notebook:
   ```bash
   !git clone https://github.com/PhatLavar/ML_ASSIGNMENT.git
   ```
2. Navigate to `notebooks/main.ipynb` or execute the core modules.
The code is designed to automatically install dependencies, download the dataset using `gdown`, and run the entire pipeline correctly within the Colab environment.

**Option B: Use the Standalone Notebook**
1. Click the [Colab Notebook link](#🔗-links) provided below.
2. Execute the notebook cells sequentially from top to bottom. This standalone notebook has everything configured to download the dataset and run the pipeline independently.

## 📂 Project Folder Structure

```text
dataset_metadata.json       # Metadata containing data splits and normalization parameters
requirements.txt            # Required Python packages
README.md                   # Project documentation
features/                   # Extracted deep features (NumPy arrays)
   X_train_resnet18.npy
   X_test_resnet18.npy
   y_train_resnet18.npy
   y_test_resnet18.npy
modules/                    # Core Python modules
   dataset_helper.py
   deep_learning_helper.py
   eda_helper.py
   traditional_helper.py
notebooks/
   main.ipynb              # Main execution notebook
```

## 🔗 Links
- **GitHub Repository:** [PhatLavar/ML_ASSIGNMENT](https://github.com/PhatLavar/ML_ASSIGNMENT)
- **Colab Notebook:** [View on Google Colab](https://drive.google.com/file/d/1f7_Z47M5agd68b4Ej4XMUKScW-ZBTIsP/view?usp=sharing)