import os
import json
import numpy as np
import torch
import torch.nn as nn
import torchvision.transforms as transforms
import torchvision.models as models
from torchvision.datasets import ImageFolder
from torch.utils.data import DataLoader
from sklearn.svm import LinearSVC
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, accuracy_score

def load_metadata(metadata_path="dataset_metadata.json"):
    """Loads the metadata JSON generated from Phase 1."""
    with open(metadata_path, "r") as f:
        metadata = json.load(f)
    return metadata

def get_transforms(metadata):
    """Sets up PyTorch transforms for train and test splits based on metadata."""
    norm_mean = metadata["normalization"]["mean"]
    norm_std = metadata["normalization"]["std"]
    input_size = tuple(metadata["input_size"])

    train_transform = transforms.Compose([
        transforms.Resize(input_size),
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.RandomRotation(degrees=15),
        # Modest jitter to add lighting robustness
        transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.05),
        transforms.ToTensor(),
        transforms.Normalize(mean=norm_mean, std=norm_std)
    ])

    test_transform = transforms.Compose([
        transforms.Resize(input_size),
        transforms.ToTensor(),
        transforms.Normalize(mean=norm_mean, std=norm_std)
    ])

    return train_transform, test_transform

def get_dataloaders(metadata, train_transform, test_transform, batch_size=32):
    """Creates PyTorch DataLoaders for train and test splits."""
    dataset_dir = "dataset"
    train_dir = os.path.join(dataset_dir, metadata["splits"]["train"])
    test_dir = os.path.join(dataset_dir, metadata["splits"]["test"])

    train_dataset = ImageFolder(root=train_dir, transform=train_transform)
    test_dataset = ImageFolder(root=test_dir, transform=test_transform)

    # shuffle=False is critical to keep extracted features aligned with their labels
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=False) 
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

    return train_loader, test_loader

def get_feature_extractor(device):
    """Initializes a pre-trained ResNet18 model as a feature extractor."""
    weights = models.ResNet18_Weights.DEFAULT
    model = models.resnet18(weights=weights)
    
    # Remove the final classification layer (fc)
    model = nn.Sequential(*list(model.children())[:-1])
    model = model.to(device)
    model.eval() # Freeze dropout and batchnorm layers
    return model

def batch_extract_features(dataloader, model, device):
    """Helper function to extract features in batches using the feature extractor."""
    features_list = []
    labels_list = []
    with torch.no_grad():
        for images, labels in dataloader:
            images = images.to(device)
            features = model(images)
            # Flatten features from (batch_size, 512, 1, 1) to (batch_size, 512)
            features = features.view(features.size(0), -1)
            
            features_list.append(features.cpu().numpy())
            labels_list.append(labels.numpy())
            
    return np.vstack(features_list), np.concatenate(labels_list)

def run_deep_learning_pipeline(metadata_path="dataset_metadata.json", batch_size=32, save_features=True):
    """
    Main function to run Phase 3:
    - Loads Metadata & datasets.
    - Extracts ResNet18 features.
    - Trans & Evaluates SVM and Logistic Regression classifiers.
    """
    metadata = load_metadata(metadata_path)
    print(f"Loaded Normalization Mean: {metadata['normalization']['mean']}")
    print(f"Loaded Normalization Std: {metadata['normalization']['std']}")
    print(f"Input Size: {metadata['input_size']}")
    
    train_transform, test_transform = get_transforms(metadata)
    train_loader, test_loader = get_dataloaders(metadata, train_transform, test_transform, batch_size)
    
    print(f"Training batches: {len(train_loader)}")
    print(f"Testing batches: {len(test_loader)}")
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    model = get_feature_extractor(device)
    print("ResNet18 initialized as a feature extractor.")
    
    print("Extracting features for train split... (This may take a minute)")
    X_train_dl, y_train_dl = batch_extract_features(train_loader, model, device)

    print("Extracting features for test split...")
    X_test_dl, y_test_dl = batch_extract_features(test_loader, model, device)

    if save_features:
        np.save('X_train_resnet18.npy', X_train_dl)
        np.save('y_train_resnet18.npy', y_train_dl)
        np.save('X_test_resnet18.npy', X_test_dl)
        np.save('y_test_resnet18.npy', y_test_dl)
        print("-> Extracted features saved to .npy files successfully.")

    target_names = metadata["classes"]
    
    print("\n-> Running ResNet18 Features + SVM...")
    # dual=False is optimal when n_samples > n_features
    svm_clf = LinearSVC(max_iter=2000, dual=False) 
    svm_clf.fit(X_train_dl, y_train_dl)
    y_pred_svm = svm_clf.predict(X_test_dl)
    
    print("Detailed Report for ResNet18 + SVM:")
    print(classification_report(y_test_dl, y_pred_svm, target_names=target_names))
    svm_acc = accuracy_score(y_test_dl, y_pred_svm)
    print(f"Accuracy: {svm_acc:.2f}")

    print("\n-> Running ResNet18 Features + LR...")
    lr_clf = LogisticRegression(max_iter=2000)
    lr_clf.fit(X_train_dl, y_train_dl)
    y_pred_lr = lr_clf.predict(X_test_dl)

    print("Detailed Report for ResNet18 + LR:")
    print(classification_report(y_test_dl, y_pred_lr, target_names=target_names))
    lr_acc = accuracy_score(y_test_dl, y_pred_lr)
    print(f"Accuracy: {lr_acc:.2f}")
    
    return {
        "svm": {
            "model": svm_clf,
            "accuracy": svm_acc,
            "predictions": y_pred_svm
        },
        "lr": {
            "model": lr_clf,
            "accuracy": lr_acc,
            "predictions": y_pred_lr
        },
        "true_labels": y_test_dl,
        "target_names": target_names
    }
