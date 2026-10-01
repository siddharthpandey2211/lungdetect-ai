# ==========================================================
# LungDetect AI
# File: preprocess.py
# ==========================================================

import os
from pathlib import Path

os.environ.setdefault("MPLBACKEND", "Agg")
os.environ.setdefault("MPLCONFIGDIR", os.path.join(os.getcwd(), ".mplconfig"))
os.makedirs(os.environ["MPLCONFIGDIR"], exist_ok=True)

import joblib
import warnings

import numpy as np
import pandas as pd

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler

warnings.filterwarnings("ignore")

# ==========================================================
# Create folders if they don't exist
# ==========================================================

BASE_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = BASE_DIR / "Data"
MODELS_DIR = BASE_DIR / "models"
PLOTS_DIR = BASE_DIR / "plots"

DATA_DIR.mkdir(exist_ok=True)
MODELS_DIR.mkdir(exist_ok=True)
PLOTS_DIR.mkdir(exist_ok=True)

DATASET_CANDIDATES = [
    DATA_DIR / "lung_cancer.csv",
    BASE_DIR / "lung_cancer.csv",
]

DATASET_PATH = None

for path in DATASET_CANDIDATES:
    if not path.exists():
        continue

    columns = pd.read_csv(path, nrows=0).columns
    if "LUNG_CANCER" in columns:
        DATASET_PATH = path
        break

if DATASET_PATH is None:
    DATASET_PATH = next((path for path in DATASET_CANDIDATES if path.exists()), None)

if DATASET_PATH is None:
    raise FileNotFoundError("Dataset not found. Expected lung_cancer.csv in the project root or Data folder.")

# ==========================================================
# Load Dataset
# ==========================================================

print("="*50)
print("Loading Dataset...")
print("="*50)

df = pd.read_csv(DATASET_PATH)

print("\nDataset Loaded Successfully!\n")

# ==========================================================
# Basic Information
# ==========================================================

print("="*50)
print("First 5 Rows")
print("="*50)

print(df.head())

print("\n")

print("="*50)
print("Dataset Information")
print("="*50)

print(df.info())

print("\n")

print("="*50)
print("Statistical Summary")
print("="*50)

print(df.describe())

print("\n")

print("="*50)
print("Missing Values")
print("="*50)

print(df.isnull().sum())

print("\n")

print("="*50)
print("Duplicate Rows")
print("="*50)

print(df.duplicated().sum())

# ==========================================================
# Remove Duplicate Rows
# ==========================================================

df.drop_duplicates(inplace=True)
df.reset_index(drop=True, inplace=True)

print("\nDuplicates Removed.\n")

# ==========================================================
# Encode Categorical Columns
# ==========================================================

print("="*50)
print("Encoding Categorical Features")
print("="*50)

encoders = {}

for column in df.columns:

    if df[column].dtype == object or pd.api.types.is_string_dtype(df[column]):

        encoder = LabelEncoder()

        df[column] = encoder.fit_transform(df[column])

        encoders[column] = encoder

print("Encoding Completed.\n")

# Save Encoders

joblib.dump(encoders, MODELS_DIR / "encoders.pkl")

# ==========================================================
# Save Clean Dataset
# ==========================================================

df.to_csv(DATA_DIR / "processed.csv", index=False)

# ==========================================================
# Exploratory Data Analysis
# ==========================================================

print("="*50)
print("Generating Graphs...")
print("="*50)

target_col = "LUNG_CANCER" if "LUNG_CANCER" in df.columns else "survived"

# ----------------------------------------------------------

plot_df = df.sample(n=min(20000, len(df)), random_state=42)

plt.figure(figsize=(8,6))

sns.countplot(x=plot_df[target_col])

plt.title(f"Target Class Distribution ({target_col})")

plt.savefig(PLOTS_DIR / "class_distribution.png")

plt.close()

# ----------------------------------------------------------

plt.figure(figsize=(14,10))

numeric_df = plot_df.select_dtypes(include=[np.number])

sns.heatmap(numeric_df.corr(),
            annot=True,
            cmap="coolwarm")

plt.title("Correlation Matrix")

plt.savefig(PLOTS_DIR / "correlation.png")

plt.close()

# ----------------------------------------------------------

plot_df.hist(figsize=(15,12))

plt.tight_layout()

plt.savefig(PLOTS_DIR / "histograms.png")

plt.close()

# ----------------------------------------------------------

plt.figure(figsize=(14,8))

sns.boxplot(data=plot_df[numeric_df.columns])

plt.xticks(rotation=90)

plt.tight_layout()

plt.savefig(PLOTS_DIR / "boxplot.png")

plt.close()

# ==========================================================
# Feature and Target
# ==========================================================

target_col = "LUNG_CANCER" if "LUNG_CANCER" in df.columns else "survived"

print("="*50)
print("Preparing Dataset")
print("="*50)

X = df.drop(target_col, axis=1)

y = df[target_col]

print("Features Shape :", X.shape)

print("Target Shape   :", y.shape)

# ==========================================================
# Train Test Split
# ==========================================================

X_train, X_test, y_train, y_test = train_test_split(

    X,
    y,

    test_size=0.20,

    random_state=42,

    stratify=y

)

print("\nTraining Samples :", len(X_train))

print("Testing Samples  :", len(X_test))

# ==========================================================
# Feature Scaling
# ==========================================================

scaler = StandardScaler()

X_train = scaler.fit_transform(X_train)

X_test = scaler.transform(X_test)

joblib.dump(scaler, MODELS_DIR / "scaler.pkl")

# ==========================================================
# Save Processed Data
# ==========================================================

joblib.dump(X_train, DATA_DIR / "X_train.pkl")
joblib.dump(X_test, DATA_DIR / "X_test.pkl")
joblib.dump(y_train, DATA_DIR / "y_train.pkl")
joblib.dump(y_test, DATA_DIR / "y_test.pkl")

print("\nProcessed datasets saved.")

# ==========================================================
# Dataset Summary
# ==========================================================

print("\n")
print("="*50)

print("Dataset Summary")

print("="*50)

print("Total Samples :", len(df))

print("Features      :", X.shape[1])

print("Train Samples :", len(X_train))

print("Test Samples  :", len(X_test))

print("="*50)

print("\nPreprocessing Completed Successfully!")
