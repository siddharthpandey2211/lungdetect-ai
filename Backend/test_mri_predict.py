from pathlib import Path

from mri_predict import predict_mri


PROJECT_ROOT = Path(__file__).resolve().parent.parent
image_path = (
    PROJECT_ROOT
    / "Data"
    / "mri"
    / "extracted"
    / "lung_cancer_MRI_dataset"
    / "validate"
    / "no_cancer"
    / "LUNG MRI NC (1).jpg"
)


if __name__ == "__main__":
    print("=" * 60)
    print("MRI PREDICTION")
    print("=" * 60)

    result = predict_mri(image_path)

    for key, value in result.items():
        print(f"{key}: {value}")