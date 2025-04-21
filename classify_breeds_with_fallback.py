import os
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay, accuracy_score, precision_score, recall_score, f1_score
from api_utils import classify_breed, get_dog_breed_from_api

# Load files
ground_truth_path = "F:/suryesh project/vision_pipeline/ground_truth_with_class.csv"
metadata_path = "F:/suryesh project/vision_pipeline/metadata_cleaned.csv"
DOG_API_KEY = "live_yz3xpAklihbmaImi9DNKUXghiMS6SQrHprzlJV0L8umvbrfK0S4dfbxaf7N3sAoP"

# Load CSVs
ground_df = pd.read_csv(ground_truth_path)
meta_df = pd.read_csv(metadata_path)

# Normalize image paths
ground_df['image_path'] = ground_df['image_path'].apply(lambda x: os.path.normpath(x).lower().strip())
meta_df['image_path'] = meta_df['image_path'].apply(lambda x: os.path.normpath(x).lower().strip())

# Get breed classification via API
ground_df = ground_df.sample(n=100, random_state=42)
ground_df['breed_name'] = ground_df['true_class'].str.lower().str.strip()

# Classify breeds using APIs
unique_breeds = ground_df['breed_name'].unique()
breed_to_class = {}
for breed in unique_breeds:
    result = classify_breed(breed)
    if result == "not found":
        print(f"🌐 Wikipedia failed for {breed}, trying TheDogAPI...")
        result = get_dog_breed_from_api(breed, DOG_API_KEY)
    breed_to_class[breed] = result
    print(f"✓ {breed} → {result}")

ground_df['api_class'] = ground_df['breed_name'].map(breed_to_class)

# Merge with model predictions
merged_df = pd.merge(
    ground_df[['image_path', 'api_class', 'class_name']],  # true class label + original label
    meta_df[['image_path', 'class_name']],  # model predictions
    on='image_path',
    suffixes=('_label', '_pred'),
    how='inner'
)

# Normalize string values
for col in ['api_class', 'class_name_label', 'class_name_pred']:
    merged_df[col] = merged_df[col].str.lower().str.strip()

# --- 1. Bar Chart of Class Distribution from Model ---
plt.figure(figsize=(10, 6))
model_class_counts = merged_df['class_name_pred'].value_counts()
model_class_counts.head(20).plot(kind='barh', color='skyblue')
plt.title('Top Predicted Classes from Model')
plt.xlabel('Count')
plt.ylabel('Class Name')
plt.gca().invert_yaxis()
plt.show()

# --- 2. Confusion Matrix: Model vs API-classified class ---
print("\n📊 Evaluation 1: Model Prediction vs API Class")
api_eval = merged_df[
    merged_df['api_class'].isin(['dog', 'cat']) & merged_df['class_name_pred'].isin(['dog', 'cat'])
]
y_true_api = api_eval['api_class']
y_pred_api = api_eval['class_name_pred']
cm_api = confusion_matrix(y_true_api, y_pred_api, labels=['cat', 'dog'])

plt.figure(figsize=(6, 5))
sns.heatmap(cm_api, annot=True, fmt='d', cmap='Blues', xticklabels=['cat', 'dog'], yticklabels=['cat', 'dog'])
plt.title("Confusion Matrix: Model vs API")
plt.xlabel('Predicted')
plt.ylabel('API Ground Truth')
plt.show()

print(f"Accuracy: {accuracy_score(y_true_api, y_pred_api):.2f}")
print(f"Precision: {precision_score(y_true_api, y_pred_api, average='weighted'):.2f}")
print(f"Recall: {recall_score(y_true_api, y_pred_api, average='weighted'):.2f}")
print(f"F1 Score: {f1_score(y_true_api, y_pred_api, average='weighted'):.2f}")

# --- 3. Confusion Matrix: Model vs original ground truth label ---
print("\n📊 Evaluation 2: Model Prediction vs Ground Truth Label")
gt_eval = merged_df[
    merged_df['class_name_label'].isin(['dog', 'cat']) & merged_df['class_name_pred'].isin(['dog', 'cat'])
]
y_true_gt = gt_eval['class_name_label']
y_pred_gt = gt_eval['class_name_pred']
cm_gt = confusion_matrix(y_true_gt, y_pred_gt, labels=['cat', 'dog'])

plt.figure(figsize=(6, 5))
sns.heatmap(cm_gt, annot=True, fmt='d', cmap='Greens', xticklabels=['cat', 'dog'], yticklabels=['cat', 'dog'])
plt.title("Confusion Matrix: Model vs Ground Truth")
plt.xlabel('Predicted')
plt.ylabel('Ground Truth Label')
plt.show()

print(f"Accuracy: {accuracy_score(y_true_gt, y_pred_gt):.2f}")
print(f"Precision: {precision_score(y_true_gt, y_pred_gt, average='weighted'):.2f}")
print(f"Recall: {recall_score(y_true_gt, y_pred_gt, average='weighted'):.2f}")
print(f"F1 Score: {f1_score(y_true_gt, y_pred_gt, average='weighted'):.2f}")
