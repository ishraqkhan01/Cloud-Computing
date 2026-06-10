import os
import joblib
import numpy as np

from PIL import Image

from sklearn.model_selection import train_test_split
from sklearn.decomposition import PCA
from sklearn.svm import SVC
from sklearn.pipeline import Pipeline
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix
)

# =====================================================
# CONFIGURATION
# =====================================================

DATASET_PATH = "flower_photos"

IMAGE_SIZE = (64, 64)

MODEL_OUTPUT_PATH = "model/image_classifier.pkl"

PCA_COMPONENTS = 150

RANDOM_STATE = 42

SUPPORTED_EXTENSIONS = (
    ".jpg",
    ".jpeg",
    ".png",
    ".bmp",
    ".webp"
)

# =====================================================
# LOAD DATASET
# =====================================================

def load_dataset(dataset_path):

    X = []
    y = []

    categories = sorted(
        [
            folder
            for folder in os.listdir(dataset_path)
            if os.path.isdir(
                os.path.join(dataset_path, folder)
            )
        ]
    )

    if len(categories) < 2:
        raise Exception(
            "Dataset must contain at least two class folders."
        )

    print("\nClasses Found:")
    print(categories)

    label_mapping = {}

    for label, category in enumerate(categories):

        label_mapping[label] = category

        category_path = os.path.join(
            dataset_path,
            category
        )

        print(f"\nLoading {category}...")

        image_count = 0

        for file_name in os.listdir(category_path):

            if not file_name.lower().endswith(
                SUPPORTED_EXTENSIONS
            ):
                continue

            image_path = os.path.join(
                category_path,
                file_name
            )

            try:

                img = Image.open(image_path)

                img = img.convert("RGB")

                img = img.resize(
                    IMAGE_SIZE
                )

                img_array = np.array(
                    img,
                    dtype=np.float32
                )

                feature_vector = img_array.flatten()

                X.append(feature_vector)

                y.append(label)

                image_count += 1

            except Exception as e:

                print(
                    f"Skipping {image_path}"
                )

                print(e)

        print(
            f"{image_count} images loaded."
        )

    return (
        np.array(X),
        np.array(y),
        label_mapping
    )

# =====================================================
# MAIN TRAINING
# =====================================================

def main():

    print("=" * 60)
    print("FLOWER IMAGE CLASSIFICATION TRAINING")
    print("=" * 60)

    # ---------------------------------------------
    # LOAD DATASET
    # ---------------------------------------------

    X, y, label_mapping = load_dataset(
        DATASET_PATH
    )

    print("\nDataset Loaded Successfully")

    print(
        f"Total Samples: {len(X)}"
    )

    print(
        f"Feature Dimension: {X.shape[1]}"
    )

    # ---------------------------------------------
    # TRAIN TEST SPLIT
    # ---------------------------------------------

    X_train, X_test, y_train, y_test = (
        train_test_split(
            X,
            y,
            test_size=0.20,
            random_state=RANDOM_STATE,
            stratify=y
        )
    )

    print(
        f"\nTraining Samples: {len(X_train)}"
    )

    print(
        f"Testing Samples: {len(X_test)}"
    )

    # ---------------------------------------------
    # PCA + SVM PIPELINE
    # ---------------------------------------------

    model = Pipeline(
        [
            (
                "pca",
                PCA(
                    n_components=PCA_COMPONENTS,
                    random_state=RANDOM_STATE
                )
            ),
            (
                "svm",
                SVC(
                    kernel="rbf",
                    C=10,
                    gamma="scale",
                    probability=True
                )
            )
        ]
    )

    print("\nTraining Model...")

    model.fit(
        X_train,
        y_train
    )

    print(
        "Training Completed Successfully."
    )

    # ---------------------------------------------
    # EVALUATION
    # ---------------------------------------------

    predictions = model.predict(
        X_test
    )

    accuracy = accuracy_score(
        y_test,
        predictions
    )

    print("\n" + "=" * 60)
    print("MODEL EVALUATION")
    print("=" * 60)

    print(
        f"\nAccuracy: {accuracy:.4f}"
    )

    print(
        "\nClassification Report:\n"
    )

    print(
        classification_report(
            y_test,
            predictions,
            target_names=[
                label_mapping[i]
                for i in sorted(
                    label_mapping.keys()
                )
            ]
        )
    )

    print(
        "\nConfusion Matrix:\n"
    )

    print(
        confusion_matrix(
            y_test,
            predictions
        )
    )

    # ---------------------------------------------
    # SAVE MODEL
    # ---------------------------------------------

    os.makedirs(
        "model",
        exist_ok=True
    )

    model_data = {

        "model": model,

        "label_mapping": label_mapping,

        "image_size": IMAGE_SIZE

    }

    joblib.dump(
        model_data,
        MODEL_OUTPUT_PATH
    )

    print("\nModel Saved Successfully")

    print(
        f"Location: {MODEL_OUTPUT_PATH}"
    )

    print(
        "\nLabel Mapping:"
    )

    print(
        label_mapping
    )

    print(
        "\nTraining Finished Successfully."
    )

# =====================================================
# ENTRY POINT
# =====================================================

if __name__ == "__main__":

    main()