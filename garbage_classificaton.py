"""
# Garbage Classification System

## Usage

1. Run the script: `python garbage_classification.py`
2. Select operation mode:
   - Option 1: Run baseline models (default parameters)
   - Option 2: Run optimized SVM model
   - Option 3: Run optimized Random Forest model
   - Option 4: Run optimized KNN model
   - Option 5: Test previously trained models on complex dataset

3. Provide paths to your datasets when prompted:
   - Main dataset: Should contain 5 subdirectories named 'battery', 'clothes', 'glass', 'paper', 'shoes'
   - Complex dataset (optional): Contains challenging real-world images for model testing

## Requirements

- Python 3.7+
- OpenCV (cv2)
- NumPy
- Matplotlib
- scikit-learn
- scikit-image
- seaborn
- joblib
- tqdm
"""

import os
import cv2
import numpy as np
import matplotlib.pyplot as plt
from skimage.feature import hog, local_binary_pattern
from sklearn.model_selection import train_test_split, GridSearchCV, StratifiedKFold, StratifiedShuffleSplit
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.decomposition import PCA
from sklearn.base import clone
import seaborn as sns
import time
import joblib
from tqdm import tqdm
from collections import Counter
import random

# Function to perform data augmentation on a single image
def augment_image(image):
    # Choose a random augmentation method
    aug_type = np.random.choice(['flip', 'rotate', 'brightness', 'shift', 'noise', 'all'])

    if aug_type == 'flip' or aug_type == 'all':
        # Horizontal flip
        if np.random.random() > 0.5:
            image = cv2.flip(image, 1)

        # Vertical flip (less common for real objects, so lower probability)
        if np.random.random() > 0.8:
            image = cv2.flip(image, 0)

    if aug_type == 'rotate' or aug_type == 'all':
        # Random rotation
        angle = np.random.uniform(-45, 45)
        rows, cols = image.shape[0], image.shape[1]
        M = cv2.getRotationMatrix2D((cols / 2, rows / 2), angle, 1)
        image = cv2.warpAffine(image, M, (cols, rows))

    if aug_type == 'brightness' or aug_type == 'all':
        # Random brightness and contrast adjustment
        alpha = np.random.uniform(0.8, 1.2)  # Contrast control
        beta = np.random.uniform(-30, 30)  # Brightness control
        image = cv2.convertScaleAbs(image, alpha=alpha, beta=beta)

    if aug_type == 'shift' or aug_type == 'all':
        # Random shift (translation)
        tx = np.random.uniform(-15, 15)
        ty = np.random.uniform(-15, 15)
        M = np.float32([[1, 0, tx], [0, 1, ty]])
        rows, cols = image.shape[0], image.shape[1]
        image = cv2.warpAffine(image, M, (cols, rows))

    if aug_type == 'noise' or aug_type == 'all':
        # Add random noise
        if np.random.random() > 0.5:
            noise = np.random.normal(0, 15, image.shape).astype(np.uint8)
            image = cv2.add(image, noise)

    return image


# Function to create augmented images
def create_augmentations(images, labels, target_count=2000, categories=None):
    # Count images per category
    label_counts = Counter(labels)
    print("Original class distribution:")

    for label_idx, count in label_counts.items():
        category_name = categories[label_idx] if categories else f"Class {label_idx}"
        print(f"{category_name}: {count} images")

    # Plot original distribution
    plt.figure(figsize=(12, 6))
    if categories:
        plt.bar(categories, [label_counts[i] for i in range(len(categories))])
    else:
        plt.bar(range(len(label_counts)), list(label_counts.values()))
    plt.title('Original Class Distribution')
    plt.ylabel('Number of Images')
    plt.tight_layout()
    plt.savefig('original_class_distribution.png')

    augmented_images = []
    augmented_labels = []

    # Process each class
    for label_idx in range(max(labels) + 1):
        # Get images of current class
        class_indices = np.where(labels == label_idx)[0]
        class_images = images[class_indices]

        # If we have more than target_count images, randomly select target_count
        if len(class_images) > target_count:
            selected_indices = np.random.choice(len(class_images), target_count, replace=False)
            selected_images = class_images[selected_indices]
            augmented_images.extend(selected_images)
            augmented_labels.extend([label_idx] * target_count)
            print(f"Randomly selected {target_count} images from {len(class_images)} for class {label_idx}")

        # If we have fewer than target_count images, augment until target_count
        elif len(class_images) < target_count:
            # Add all original images
            augmented_images.extend(class_images)
            augmented_labels.extend([label_idx] * len(class_images))

            # Calculate how many more images we need
            num_to_generate = target_count - len(class_images)
            print(f"Generating {num_to_generate} augmented images for class {label_idx}")

            # Generate augmented images
            for _ in tqdm(range(num_to_generate), desc=f"Augmenting class {label_idx}"):
                # Randomly select an image to augment
                img_to_augment = class_images[np.random.randint(0, len(class_images))]
                # Apply augmentation
                augmented_img = augment_image(img_to_augment)
                augmented_images.append(augmented_img)
                augmented_labels.append(label_idx)
        else:
            # If we have exactly target_count images, use all of them
            augmented_images.extend(class_images)
            augmented_labels.extend([label_idx] * len(class_images))

    # Convert to numpy arrays
    augmented_images = np.array(augmented_images)
    augmented_labels = np.array(augmented_labels)

    # Display new distribution
    new_label_counts = Counter(augmented_labels)
    print("\nBalanced class distribution:")
    for label_idx, count in new_label_counts.items():
        category_name = categories[label_idx] if categories else f"Class {label_idx}"
        print(f"{category_name}: {count} images")

    # Plot new distribution
    plt.figure(figsize=(12, 6))
    if categories:
        plt.bar(categories, [new_label_counts[i] for i in range(len(categories))])
    else:
        plt.bar(range(len(new_label_counts)), list(new_label_counts.values()))
    plt.title('Balanced Class Distribution After Augmentation')
    plt.ylabel('Number of Images')
    plt.tight_layout()
    plt.savefig('balanced_class_distribution.png')

    return augmented_images, augmented_labels


# Function to load all images and labels with data balancing
def load_dataset(base_path, categories, balance_classes=True, target_count=2000):
    images = []
    labels = []
    category_counts = {}

    # First pass: load all images and count per category
    for category_idx, category in enumerate(categories):
        category_path = os.path.join(base_path, category)
        if os.path.exists(category_path):
            # Get all image files in the directory
            files = [f for f in os.listdir(category_path) if f.lower().endswith(('.png', '.jpg', '.jpeg'))]
            category_counts[category] = len(files)
            print(f"Found {len(files)} images in category: {category}")

    # Plot class distribution before balancing
    plt.figure(figsize=(12, 6))
    plt.bar(categories, [category_counts[cat] for cat in categories])
    plt.title('Original Class Distribution')
    plt.ylabel('Number of Images')
    plt.tight_layout()
    plt.savefig('original_dataset_distribution.png')

    # Second pass: load with class balancing
    for category_idx, category in enumerate(categories):
        category_path = os.path.join(base_path, category)
        if os.path.exists(category_path):
            files = [f for f in os.listdir(category_path) if f.lower().endswith(('.png', '.jpg', '.jpeg'))]

            # If balancing classes and we have more than target_count, randomly select target_count images
            if balance_classes and len(files) > target_count:
                print(f"Randomly selecting {target_count} images from {len(files)} for category: {category}")
                files = random.sample(files, target_count)

            for file in tqdm(files, desc=f"Loading {category}"):
                img_path = os.path.join(category_path, file)
                img = cv2.imread(img_path)
                if img is not None:
                    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)  # Convert from BGR to RGB
                    img = cv2.resize(img, (128, 128))  # Resize for consistency
                    images.append(img)
                    labels.append(category_idx)

    images = np.array(images)
    labels = np.array(labels)

    # Perform data augmentation if needed
    if balance_classes:
        for category_idx, category in enumerate(categories):
            if category_counts[category] < target_count:
                print(f"Need to augment {target_count - category_counts[category]} images for category: {category}")

        # Apply data augmentation to balance classes
        images, labels = create_augmentations(images, labels, target_count=target_count, categories=categories)

    return images, labels


# Function to load complex dataset without balancing
def load_complex_dataset(base_path, categories):
    images = []
    labels = []
    category_counts = {}

    # Load all images and count per category
    for category_idx, category in enumerate(categories):
        category_path = os.path.join(base_path, category)
        if os.path.exists(category_path):
            # Get all image files in the directory
            files = [f for f in os.listdir(category_path) if f.lower().endswith(('.png', '.jpg', '.jpeg'))]
            category_counts[category] = len(files)
            print(f"Found {len(files)} images in complex dataset category: {category}")

            # Load each image
            for file in tqdm(files, desc=f"Loading complex {category}"):
                img_path = os.path.join(category_path, file)
                img = cv2.imread(img_path)
                if img is not None:
                    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)  # Convert from BGR to RGB
                    img = cv2.resize(img, (128, 128))  # Resize for consistency
                    images.append(img)
                    labels.append(category_idx)

    # Plot complex dataset distribution
    plt.figure(figsize=(12, 6))
    plt.bar(categories, [category_counts.get(cat, 0) for cat in categories])
    plt.title('Complex Dataset Class Distribution')
    plt.ylabel('Number of Images')
    plt.tight_layout()
    plt.savefig('complex_dataset_distribution.png')

    # Convert to numpy arrays
    images = np.array(images)
    labels = np.array(labels)

    print(f"Loaded complex dataset: {len(images)} images without balancing")

    return images, labels


# Function to extract HOG features from an image
def extract_hog_features(image):
    # Convert to grayscale if it's a color image
    if len(image.shape) == 3:
        gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
    else:
        gray = image

    # Extract HOG features
    fd, _ = hog(
        gray,
        orientations=9,  # Increased from 8 to 9 for more directional sensitivity
        pixels_per_cell=(16, 16),
        cells_per_block=(2, 2),  # Increased from 1x1 to 2x2 for better normalization
        visualize=True,
        block_norm='L2-Hys'
    )

    return fd


# Function to extract LBP features from an image
def extract_lbp_features(image):
    # Convert to grayscale if it's a color image
    if len(image.shape) == 3:
        gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
    else:
        gray = image

    # Parameters for LBP
    radius = 3
    n_points = 8 * radius

    # Compute LBP
    lbp = local_binary_pattern(gray, n_points, radius, method='uniform')

    # Compute histogram of LBP
    n_bins = int(lbp.max() + 1)
    hist, _ = np.histogram(lbp.ravel(), bins=n_bins, range=(0, n_bins), density=True)
    return hist


# Function to extract all features from a single image
def extract_all_features(image):
    # Extract different feature types
    hog_features = extract_hog_features(image)
    lbp_features = extract_lbp_features(image)

    # Combine all features
    combined_features = np.concatenate([hog_features, lbp_features])
    return combined_features


# Function to extract features from all images
def extract_features_from_dataset(images):
    all_features = []

    for image in tqdm(images, desc="Extracting features"):
        all_features.append(extract_all_features(image))

    return np.array(all_features)


# Function to perform hyperparameter tuning for SVM
def tune_svm(X_train, y_train, cv=5):
    print("Tuning SVM hyperparameters...")

    # Define parameter grid
    param_grid = {
        'svm__C': [1, 5, 10, 20],
        'svm__gamma': ['scale', 'auto', 0.01, 0.1],
        'svm__kernel': ['rbf', 'linear']
    }

    # Use a smaller portion of data for tuning to speed up the process
    if len(X_train) > 1000:
        print(f"Original training set size: {X_train.shape[0]}")
        sample_size = min(1000, int(0.4 * len(X_train)))
        print(f"Using a subset of {sample_size} samples for hyperparameter tuning")

        splitter = StratifiedShuffleSplit(n_splits=1, train_size=sample_size, random_state=42)
        for train_idx, _ in splitter.split(X_train, y_train):
            X_tune = X_train[train_idx]
            y_tune = y_train[train_idx]

        print(f"Tuning subset size: {X_tune.shape[0]}")
    else:
        X_tune = X_train
        y_tune = y_train
        print("Using entire training set for tuning")

    # Define pipeline
    pipeline = Pipeline([
        ('scaler', StandardScaler()),
        ('pca', PCA(n_components=0.85)),
        ('svm', SVC(probability=True, random_state=42))
    ])

    # Create grid search
    grid_search = GridSearchCV(
        pipeline, param_grid, cv=cv, scoring='accuracy', verbose=1, n_jobs=-1
    )

    # Fit grid search
    print("Starting grid search on the subset...")
    start_time = time.time()
    grid_search.fit(X_tune, y_tune)
    end_time = time.time()
    print(f"Grid search completed in {end_time - start_time:.2f} seconds")

    # Print best parameters
    print(f"Best SVM parameters: {grid_search.best_params_}")
    print(f"Best SVM CV accuracy: {grid_search.best_score_:.4f}")

    # Get best parameters
    best_params = grid_search.best_params_

    # Train final model on the full training set
    print("Training final model with best parameters on full training set...")
    final_model = Pipeline([
        ('scaler', StandardScaler()),
        ('pca', PCA(n_components=0.85)),
        ('svm', SVC(
            C=best_params.get('svm__C', 1.0),
            gamma=best_params.get('svm__gamma', 'scale'),
            kernel=best_params.get('svm__kernel', 'rbf'),
            probability=True,
            random_state=42
        ))
    ])

    final_model.fit(X_train, y_train)
    print("Final model training completed")

    return final_model


# Function to perform hyperparameter tuning for Random Forest
def tune_rf(X_train, y_train, cv=5):
    print("Tuning Random Forest hyperparameters...")

    # Define parameter grid
    param_grid = {
        'rf__n_estimators': [100, 200, 300],
        'rf__max_depth': [None, 30, 50, 70],
        'rf__min_samples_split': [2, 5],
        'rf__min_samples_leaf': [1, 2]
    }

    # Use a smaller portion of data for tuning to speed up the process
    if len(X_train) > 1000:
        print(f"Original training set size: {X_train.shape[0]}")
        sample_size = min(1000, int(0.4 * len(X_train)))
        print(f"Using a subset of {sample_size} samples for hyperparameter tuning")

        splitter = StratifiedShuffleSplit(n_splits=1, train_size=sample_size, random_state=42)
        for train_idx, _ in splitter.split(X_train, y_train):
            X_tune = X_train[train_idx]
            y_tune = y_train[train_idx]

        print(f"Tuning subset size: {X_tune.shape[0]}")
    else:
        X_tune = X_train
        y_tune = y_train
        print("Using entire training set for tuning")

    # Define pipeline
    pipeline = Pipeline([
        ('scaler', StandardScaler()),
        ('pca', PCA(n_components=0.85)),
        ('rf', RandomForestClassifier(random_state=42))
    ])

    # Create grid search
    grid_search = GridSearchCV(
        pipeline, param_grid, cv=cv, scoring='accuracy', verbose=1, n_jobs=-1
    )

    # Fit grid search
    print("Starting grid search on the subset...")
    start_time = time.time()
    grid_search.fit(X_tune, y_tune)
    end_time = time.time()
    print(f"Grid search completed in {end_time - start_time:.2f} seconds")

    # Print best parameters
    print(f"Best RF parameters: {grid_search.best_params_}")
    print(f"Best RF CV accuracy: {grid_search.best_score_:.4f}")

    # Get best parameters
    best_params = grid_search.best_params_

    # Train final model on the full training set
    print("Training final model with best parameters on full training set...")
    final_model = Pipeline([
        ('scaler', StandardScaler()),
        ('pca', PCA(n_components=0.85)),
        ('rf', RandomForestClassifier(
            n_estimators=best_params.get('rf__n_estimators', 100),
            max_depth=best_params.get('rf__max_depth', None),
            min_samples_split=best_params.get('rf__min_samples_split', 2),
            min_samples_leaf=best_params.get('rf__min_samples_leaf', 1),
            random_state=42
        ))
    ])

    final_model.fit(X_train, y_train)
    print("Final model training completed")

    return final_model


# Function to perform hyperparameter tuning for KNN
def tune_knn(X_train, y_train, cv=5):
    print("Tuning KNN hyperparameters...")

    # Define parameter grid
    param_grid = {
        'knn__n_neighbors': [3, 5, 7, 9, 11],
        'knn__weights': ['uniform', 'distance'],
        'knn__metric': ['euclidean', 'manhattan', 'minkowski']
    }

    # Use a smaller portion of data for tuning to speed up the process
    if len(X_train) > 1000:
        print(f"Original training set size: {X_train.shape[0]}")
        sample_size = min(1000, int(0.4 * len(X_train)))
        print(f"Using a subset of {sample_size} samples for hyperparameter tuning")

        splitter = StratifiedShuffleSplit(n_splits=1, train_size=sample_size, random_state=42)
        for train_idx, _ in splitter.split(X_train, y_train):
            X_tune = X_train[train_idx]
            y_tune = y_train[train_idx]

        print(f"Tuning subset size: {X_tune.shape[0]}")
    else:
        X_tune = X_train
        y_tune = y_train
        print("Using entire training set for tuning")

    # Define pipeline
    pipeline = Pipeline([
        ('scaler', StandardScaler()),
        ('pca', PCA(n_components=0.85)),
        ('knn', KNeighborsClassifier())
    ])

    # Create grid search
    grid_search = GridSearchCV(
        pipeline, param_grid, cv=cv, scoring='accuracy', verbose=1, n_jobs=-1
    )

    # Fit grid search
    print("Starting grid search on the subset...")
    start_time = time.time()
    grid_search.fit(X_tune, y_tune)
    end_time = time.time()
    print(f"Grid search completed in {end_time - start_time:.2f} seconds")

    # Print best parameters
    print(f"Best KNN parameters: {grid_search.best_params_}")
    print(f"Best KNN CV accuracy: {grid_search.best_score_:.4f}")

    # Get best parameters
    best_params = grid_search.best_params_

    # Train final model on the full training set
    print("Training final model with best parameters on full training set...")
    final_model = Pipeline([
        ('scaler', StandardScaler()),
        ('pca', PCA(n_components=0.85)),
        ('knn', KNeighborsClassifier(
            n_neighbors=best_params.get('knn__n_neighbors', 5),
            weights=best_params.get('knn__weights', 'uniform'),
            metric=best_params.get('knn__metric', 'minkowski')
        ))
    ])

    final_model.fit(X_train, y_train)
    print("Final model training completed")

    return final_model


# Function to run baseline models with simple features
def run_baseline(base_path, categories):
    print("\n===== Running Baseline Models =====")

    # Load the dataset without balancing for baseline comparison
    print("Loading dataset for baseline...")
    images, labels = load_dataset(base_path, categories, balance_classes=False)
    print(f"Loaded {len(images)} images for baseline")

    # Define feature types
    feature_types = ['hog', 'lbp']

    # Initialize models
    models = {
        'SVM': SVC(kernel='linear', random_state=42),
        'Random Forest': RandomForestClassifier(n_estimators=100, random_state=42),
        'KNN': KNeighborsClassifier(n_neighbors=5)
    }

    # Results storage
    results = {
        'Model': [],
        'Feature': [],
        'Accuracy': [],
        'Training Time': []
    }

    # For each feature type
    for feature_type in feature_types:
        print(f"\nExtracting {feature_type} features...")

        # Extract features based on feature type
        features = []
        for image in tqdm(images, desc=f"Extracting {feature_type}"):
            if feature_type == 'hog':
                feature = extract_hog_features(image)
            elif feature_type == 'lbp':
                feature = extract_lbp_features(image)
            features.append(feature)

        features = np.array(features)
        print(f"Extracted {feature_type} features with shape {features.shape}")

        # Split the dataset
        X_train, X_test, y_train, y_test = train_test_split(
            features, labels, test_size=0.3, random_state=42, stratify=labels
        )
        print(f"Training set: {X_train.shape}, Test set: {X_test.shape}")

        # Train and evaluate each model with the current feature type
        for model_name, model_instance in models.items():
            print(f"\nEvaluating baseline {model_name} with {feature_type} features...")

            # Train and evaluate
            start_time = time.time()
            model_instance.fit(X_train, y_train)
            y_pred = model_instance.predict(X_test)
            training_time = time.time() - start_time

            # Calculate accuracy
            accuracy = accuracy_score(y_test, y_pred)

            print(f"Baseline {model_name} with {feature_type} - Accuracy: {accuracy:.4f}")
            print(f"Baseline {model_name} with {feature_type} - Training Time: {training_time:.2f} seconds")

            # Display classification report
            report = classification_report(y_test, y_pred, target_names=categories)
            print(f"Baseline {model_name} with {feature_type} - Classification Report:")
            print(report)

            # Create confusion matrix
            cm = confusion_matrix(y_test, y_pred)
            plt.figure(figsize=(10, 8))
            sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=categories, yticklabels=categories)
            plt.title(f'Confusion Matrix - Baseline {model_name} with {feature_type}')
            plt.ylabel('True Label')
            plt.xlabel('Predicted Label')
            plt.tight_layout()
            plt.savefig(f'baseline_confusion_matrix_{model_name}_{feature_type}.png')

            # Store results
            results['Model'].append(model_name)
            results['Feature'].append(feature_type)
            results['Accuracy'].append(accuracy)
            results['Training Time'].append(training_time)

    # Create a bar chart to compare results
    plt.figure(figsize=(15, 8))

    # Group bars by feature type
    feature_positions = np.arange(len(feature_types))
    bar_width = 0.25

    # Plot bars for each model
    for i, model_name in enumerate(models.keys()):
        model_accuracies = []
        for feature_type in feature_types:
            # Find the accuracy for this model and feature type
            indices = [j for j, (m, f) in enumerate(zip(results['Model'], results['Feature']))
                       if m == model_name and f == feature_type]
            if indices:
                model_accuracies.append(results['Accuracy'][indices[0]])
            else:
                model_accuracies.append(0)

        plt.bar(feature_positions + i * bar_width, model_accuracies,
                width=bar_width, label=model_name)

    plt.xlabel('Feature Type')
    plt.ylabel('Accuracy')
    plt.title('Baseline Model Accuracy by Feature Type')
    plt.xticks(feature_positions + bar_width, feature_types)
    plt.legend()
    plt.tight_layout()
    plt.savefig('baseline_model_feature_comparison.png')

    # Print summary table
    print("\nBaseline Model Performance Summary:")
    print("-------------------------")
    print("Feature Type | Model | Accuracy | Training Time (s)")
    print("-------------------------")
    for i in range(len(results['Model'])):
        print(
            f"{results['Feature'][i]:12} | {results['Model'][i]:14} | {results['Accuracy'][i]:.4f} | {results['Training Time'][i]:.2f}")


# Function to evaluate model
def evaluate_model(model, X_test, y_test, categories, model_name="Optimized Model"):
    start_time = time.time()

    # Make predictions
    y_pred = model.predict(X_test)

    # Calculate accuracy
    accuracy = accuracy_score(y_test, y_pred)

    # Calculate inference time
    inference_time = time.time() - start_time

    print(f"{model_name} Accuracy: {accuracy:.4f}")
    print(f"{model_name} Inference Time: {inference_time:.2f} seconds")

    # Display classification report
    report = classification_report(y_test, y_pred, target_names=categories)
    print(f"{model_name} Classification Report:")
    print(report)

    # Create confusion matrix
    cm = confusion_matrix(y_test, y_pred)
    plt.figure(figsize=(10, 8))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=categories, yticklabels=categories)
    plt.title(f'Confusion Matrix - {model_name}')
    plt.ylabel('True Label')
    plt.xlabel('Predicted Label')
    plt.tight_layout()
    plt.savefig(f'confusion_matrix_{model_name}.png'.replace(' ', '_'))

    return accuracy, report, cm


# Function to visualize model performance with cross-validation
def cross_val_visualization(model, X, y, cv, categories):
    print("Performing cross-validation visualization...")

    # Initialize arrays for storing results
    accuracies = []
    all_cms = []

    # Perform cross-validation
    for train_idx, test_idx in tqdm(cv.split(X, y), total=cv.get_n_splits(), desc="Cross-validation"):
        X_train, X_test = X[train_idx], X[test_idx]
        y_train, y_test = y[train_idx], y[test_idx]

        # Clone the model and fit
        model_clone = clone(model)
        model_clone.fit(X_train, y_train)

        # Predict and calculate accuracy
        y_pred = model_clone.predict(X_test)
        acc = accuracy_score(y_test, y_pred)
        accuracies.append(acc)

        # Calculate confusion matrix
        cm = confusion_matrix(y_test, y_pred, labels=range(len(categories)))
        all_cms.append(cm)

    # Plot accuracy distribution
    plt.figure(figsize=(10, 6))
    plt.boxplot(accuracies)
    plt.title('Cross-Validation Accuracy Distribution')
    plt.ylabel('Accuracy')
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.tight_layout()
    plt.savefig('cross_val_accuracy.png')

    # Calculate and plot average confusion matrix
    avg_cm = np.mean(all_cms, axis=0).astype(int)
    plt.figure(figsize=(10, 8))
    sns.heatmap(avg_cm, annot=True, fmt='d', cmap='Blues', xticklabels=categories, yticklabels=categories)
    plt.title('Average Confusion Matrix Across Cross-Validation')
    plt.ylabel('True Label')
    plt.xlabel('Predicted Label')
    plt.tight_layout()
    plt.savefig('cross_val_confusion_matrix.png')

    print(f"Cross-validation mean accuracy: {np.mean(accuracies):.4f} ± {np.std(accuracies):.4f}")
    return accuracies, avg_cm


# Function to run advanced model (SVM, RF, or KNN)
def run_advanced_model(base_path, categories, model_type="svm"):
    print(f"\n===== Running Advanced {model_type.upper()} Model =====")

    # Load the dataset with balancing for the advanced model
    print("Loading and balancing dataset...")
    images, labels = load_dataset(base_path, categories, balance_classes=True, target_count=2000)
    print(f"Loaded and balanced dataset: {len(images)} images")

    # Extract features
    print("Extracting combined features...")
    X = extract_features_from_dataset(images)
    y = labels

    # Split the dataset into train and test sets
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    print(f"Training set: {X_train.shape}")
    print(f"Test set: {X_test.shape}")

    # Set up cross-validation
    cv = StratifiedKFold(n_splits=3, shuffle=True, random_state=42)

    # Choose model based on type
    if model_type.lower() == "svm":
        # Tune SVM
        model = tune_svm(X_train, y_train, cv=cv)
        model_name = "Optimized SVM"
    elif model_type.lower() == "rf":
        # Tune Random Forest
        model = tune_rf(X_train, y_train, cv=cv)
        model_name = "Optimized RF"
    elif model_type.lower() == "knn":
        # Tune KNN
        model = tune_knn(X_train, y_train, cv=cv)
        model_name = "Optimized KNN"
    else:
        print(f"Unknown model type: {model_type}. Using SVM as default.")
        model = tune_svm(X_train, y_train, cv=cv)
        model_name = "Optimized SVM"

    # Save model
    model_file = f'tuned_{model_type.lower()}_model.joblib'
    joblib.dump(model, model_file)
    print(f"Saved tuned model to {model_file}")

    # Evaluate model on test set
    print(f"\nEvaluating {model_name} on test set:")
    accuracy, report, cm = evaluate_model(model, X_test, y_test, categories, model_name=model_name)

    # Visualize model performance with cross-validation
    cross_val_accuracies, avg_cm = cross_val_visualization(model, X, y, cv, categories)

    # Analyze class difficulties
    print("\nAnalyzing classification difficulties:")
    class_accuracies = []
    for i in range(len(categories)):
        class_acc = avg_cm[i, i] / avg_cm[i, :].sum()
        class_accuracies.append(class_acc)
        print(f"{categories[i]}: {class_acc:.4f}")

    # Plot class accuracies
    plt.figure(figsize=(10, 6))
    plt.bar(categories, class_accuracies)
    plt.title(f'Classification Accuracy by Class - {model_name}')
    plt.ylabel('Accuracy')
    plt.ylim(0, 1)
    for i, v in enumerate(class_accuracies):
        plt.text(i, v + 0.02, f"{v:.2f}", ha='center')
    plt.tight_layout()
    plt.savefig(f'class_accuracies_{model_type.lower()}.png')

    # Print final results
    print(f"\nFinal {model_name} Results:")
    print(f"Test Accuracy: {accuracy:.4f}")
    print(f"Cross-validation Accuracy: {np.mean(cross_val_accuracies):.4f} ± {np.std(cross_val_accuracies):.4f}")

    return model


# Function to test model on complex dataset
def test_on_complex_dataset(model, complex_dataset_path, categories, model_type="svm"):
    print(f"\n===== Testing {model_type.upper()} on Complex Dataset =====")

    # Load complex dataset without balancing
    print("Loading complex dataset...")
    complex_images, complex_labels = load_complex_dataset(complex_dataset_path, categories)
    print(f"Loaded complex dataset: {len(complex_images)} images")

    # Extract features from complex dataset
    print("Extracting features from complex dataset...")
    complex_X = extract_features_from_dataset(complex_images)
    complex_y = complex_labels

    # Evaluate model on complex dataset
    print(f"\nEvaluating model on complex dataset:")
    model_name = f"{model_type.upper()} on Complex Dataset"
    complex_accuracy, complex_report, complex_cm = evaluate_model(
        model, complex_X, complex_y, categories, model_name=model_name
    )

    # Analyze class accuracies on complex dataset
    print("\nAnalyzing class accuracies on complex dataset:")
    complex_class_accuracies = []
    for i in range(len(categories)):
        if np.sum(complex_y == i) > 0:  # Ensure the class has samples
            class_acc = complex_cm[i, i] / np.sum(complex_y == i)
            complex_class_accuracies.append(class_acc)
            print(f"{categories[i]}: {class_acc:.4f}")
        else:
            complex_class_accuracies.append(0)
            print(f"{categories[i]}: No samples")

    # Plot class accuracies on complex dataset
    plt.figure(figsize=(10, 6))
    plt.bar(categories, complex_class_accuracies)
    plt.title(f'Class Accuracies on Complex Dataset - {model_type.upper()}')
    plt.ylabel('Accuracy')
    plt.ylim(0, 1)
    for i, v in enumerate(complex_class_accuracies):
        if v > 0:  # Only show for classes with samples
            plt.text(i, v + 0.02, f"{v:.2f}", ha='center')
    plt.tight_layout()
    plt.savefig(f'complex_class_accuracies_{model_type.lower()}.png')

    # Save analysis to file
    with open(f"complex_dataset_analysis_{model_type.lower()}.txt", "w") as f:
        f.write(f"Complex Dataset Analysis for {model_type.upper()}\n")
        f.write("=" * 50 + "\n\n")
        f.write(f"Overall Accuracy: {complex_accuracy:.4f}\n\n")
        f.write("Classification Report:\n")
        f.write(complex_report + "\n\n")
        f.write("Class Accuracies:\n")
        for i, category in enumerate(categories):
            if np.sum(complex_y == i) > 0:
                acc = complex_class_accuracies[i]
                f.write(f"{category}: {acc:.4f} ({complex_cm[i, i]}/{np.sum(complex_y == i)} samples)\n")
            else:
                f.write(f"{category}: No samples\n")

        f.write("\nComplexity Challenges Analysis:\n")
        f.write("1. Background Interference: Complex backgrounds can contain features similar to the target objects\n")
        f.write("2. Lighting Variations: Different lighting conditions affect feature extraction\n")
        f.write("3. Occlusions: Partial object visibility reduces feature completeness\n")
        f.write("4. Viewpoint Changes: Non-standard angles create dissimilar features from training examples\n")
        f.write("5. Scale Variations: Objects at different distances appear at different sizes\n\n")

        f.write("Recommendations for Improvement:\n")
        f.write("1. Object Detection/Segmentation: Locate or segment objects prior to classification\n")
        f.write("2. Feature Engineering: Use more robust feature descriptors or multi-scale features\n")
        f.write("3. Data Augmentation: Simulate complex backgrounds and lighting variations in training\n")
        f.write("4. Deep Learning Approach: Consider CNN or other deep learning models\n")
        f.write("5. Ensemble Methods: Combine multiple classifiers for greater robustness\n")

    print(f"Complex dataset analysis saved to complex_dataset_analysis_{model_type.lower()}.txt")
    return complex_accuracy, complex_report, complex_cm


# Main interface function
def main():
    print("===== Waste Classification System =====\n")

    # Default categories
    categories = ['battery', 'clothes', 'glass', 'paper', 'shoes']

    print("IMPORTANT REMINDER: To test models on complex scenes, you need to train the")
    print("optimized models first (options 2-4). After training, you can either test")
    print("immediately or use option 5 to test on complex scenes later.\n")

    # Ask user for dataset path
    base_path = input(
        "Enter the path to your dataset (or press Enter for default '/Users/shenzitong/Desktop/dataset'): ")
    if not base_path:
        base_path = '/Users/shenzitong/Desktop/dataset'

    # Check if directory exists
    if not os.path.isdir(base_path):
        print(f"Error: Directory {base_path} does not exist. Exiting.")
        return

    # Check for category folders
    missing_categories = []
    for category in categories:
        if not os.path.isdir(os.path.join(base_path, category)):
            missing_categories.append(category)

    if missing_categories:
        print(f"Warning: The following category folders are missing in {base_path}:")
        for category in missing_categories:
            print(f"  - {category}")
        print("\nPlease ensure your dataset contains folders for each category.")
        continue_anyway = input("Continue anyway? (y/n): ").lower()
        if continue_anyway != 'y':
            return

    # Check which models have been trained
    trained_models = []
    if os.path.exists('tuned_svm_model.joblib'):
        trained_models.append("SVM")
    if os.path.exists('tuned_rf_model.joblib'):
        trained_models.append("RF (Random Forest)")
    if os.path.exists('tuned_knn_model.joblib'):
        trained_models.append("KNN")

    if trained_models:
        print(f"\nPreviously trained models: {', '.join(trained_models)}")
    else:
        print("\nNo trained models found. Please train models before testing on complex scenes.")

    # Ask user to select model
    print("\nSelect a model to run:")
    print("1. Baseline Models (SVM, RF, KNN with simple features)")
    print("2. Optimized SVM")
    print("3. Optimized Random Forest")
    print("4. Optimized KNN")
    print("5. Test on Complex Dataset (requires trained models)")

    choice = input("\nEnter your choice (1-5): ")

    # Process choice
    if choice == '1':
        run_baseline(base_path, categories)

    elif choice == '2':
        model = run_advanced_model(base_path, categories, model_type="svm")

        # Ask if user wants to test on complex dataset
        test_complex = input("\nWould you like to test this model on a complex dataset? (y/n): ").lower()
        if test_complex == 'y':
            complex_path = input(
                "Enter the path to your complex dataset (or press Enter for default '/Users/shenzitong/Desktop/complex_dataset'): ")
            if not complex_path:
                complex_path = '/Users/shenzitong/Desktop/complex_dataset'

            if os.path.isdir(complex_path):
                test_on_complex_dataset(model, complex_path, categories, model_type="svm")
            else:
                print(f"Error: Complex dataset directory {complex_path} does not exist.")

    elif choice == '3':
        model = run_advanced_model(base_path, categories, model_type="rf")

        # Ask if user wants to test on complex dataset
        test_complex = input("\nWould you like to test this model on a complex dataset? (y/n): ").lower()
        if test_complex == 'y':
            complex_path = input(
                "Enter the path to your complex dataset (or press Enter for default '/Users/shenzitong/Desktop/complex_dataset'): ")
            if not complex_path:
                complex_path = '/Users/shenzitong/Desktop/complex_dataset'

            if os.path.isdir(complex_path):
                test_on_complex_dataset(model, complex_path, categories, model_type="rf")
            else:
                print(f"Error: Complex dataset directory {complex_path} does not exist.")

    elif choice == '4':
        model = run_advanced_model(base_path, categories, model_type="knn")

        # Ask if user wants to test on complex dataset
        test_complex = input("\nWould you like to test this model on a complex dataset? (y/n): ").lower()
        if test_complex == 'y':
            complex_path = input(
                "Enter the path to your complex dataset (or press Enter for default '/Users/shenzitong/Desktop/complex_dataset'): ")
            if not complex_path:
                complex_path = '/Users/shenzitong/Desktop/complex_dataset'

            if os.path.isdir(complex_path):
                test_on_complex_dataset(model, complex_path, categories, model_type="knn")
            else:
                print(f"Error: Complex dataset directory {complex_path} does not exist.")

    elif choice == '5':
        # Check if any models are trained
        if not trained_models:
            print("No trained models found. Please train at least one model first (options 2-4).")
            return

        # Ask which model to load for testing
        print("\nWhich model would you like to use for testing?")
        for i, model_name in enumerate(["SVM", "Random Forest", "KNN"]):
            model_status = "✓" if model_name.split()[0] in [m.split()[0] for m in trained_models] else "✗"
            print(f"{i + 1}. {model_name} [{model_status}]")

        model_choice = input("Enter your choice (1-3): ")

        if model_choice == '1':
            model_type = "svm"
            if "SVM" not in trained_models:
                print("SVM model has not been trained yet. Training now...")
                model = run_advanced_model(base_path, categories, model_type="svm")
            else:
                model_file = f'tuned_{model_type}_model.joblib'
                print(f"Loading existing {model_type.upper()} model from {model_file}")
                model = joblib.load(model_file)

        elif model_choice == '2':
            model_type = "rf"
            if "RF" not in trained_models:
                print("Random Forest model has not been trained yet. Training now...")
                model = run_advanced_model(base_path, categories, model_type="rf")
            else:
                model_file = f'tuned_{model_type}_model.joblib'
                print(f"Loading existing {model_type.upper()} model from {model_file}")
                model = joblib.load(model_file)

        elif model_choice == '3':
            model_type = "knn"
            if "KNN" not in trained_models:
                print("KNN model has not been trained yet. Training now...")
                model = run_advanced_model(base_path, categories, model_type="knn")
            else:
                model_file = f'tuned_{model_type}_model.joblib'
                print(f"Loading existing {model_type.upper()} model from {model_file}")
                model = joblib.load(model_file)

        else:
            print("Invalid choice. Using SVM as default.")
            model_type = "svm"
            if "SVM" not in trained_models:
                print("SVM model has not been trained yet. Training now...")
                model = run_advanced_model(base_path, categories, model_type="svm")
            else:
                model_file = f'tuned_{model_type}_model.joblib'
                print(f"Loading existing {model_type.upper()} model from {model_file}")
                model = joblib.load(model_file)

        # Get complex dataset path
        complex_path = input(
            "Enter the path to your complex dataset (or press Enter for default '/Users/shenzitong/Desktop/complex_dataset'): ")
        if not complex_path:
            complex_path = '/Users/shenzitong/Desktop/complex_dataset'

        if os.path.isdir(complex_path):
            # To ensure feature consistency, we'll reload and immediately test
            try:
                print("Testing on complex dataset using the newly trained/loaded model...")
                test_on_complex_dataset(model, complex_path, categories, model_type=model_type)
            except ValueError as e:
                if "features" in str(e):
                    print(f"Error: Feature dimension mismatch. This usually happens when models are trained")
                    print(f"with a different version of the code. To fix this, please retrain the model.")
                    print(f"Original error: {e}")

                    retrain = input("Would you like to retrain the model now? (y/n): ").lower()
                    if retrain == 'y':
                        print(f"Retraining {model_type.upper()} model...")
                        model = run_advanced_model(base_path, categories, model_type=model_type)
                        print("Testing on complex dataset using the newly trained model...")
                        test_on_complex_dataset(model, complex_path, categories, model_type=model_type)
                else:
                    print(f"An error occurred: {e}")
        else:
            print(f"Error: Complex dataset directory {complex_path} does not exist.")

    else:
        print("Invalid choice. Please run the script again and select a valid option.")


if __name__ == "__main__":
    main()