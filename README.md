# 🐾 Dog vs Cat Classifier with Public API Validation

An automated pipeline that detects and classifies animals in images using a combination of computer vision and public data sources. This project integrates object detection, data filtering, and API-based breed classification to validate and benchmark predictions across multiple perspectives.

---

## ✨ Key Features

- 📸 **Object Detection with YOLOv5**  
  Detects dogs and cats in raw image datasets using pretrained models.

- 🌐 **Breed Classification via APIs**  
  Uses Wikipedia and TheDogAPI to classify animal breeds into high-level categories (`dog` or `cat`).

- 🧹 **Data Cleaning & Filtering**  
  Cleans and filters raw predictions and labels for reliable evaluation.

- 🔁 **End-to-End Automation**  
  From detection to classification to evaluation — all handled through modular, reusable Python scripts.

- 📊 **Multi-perspective Evaluation**  
  Generates classification reports and confusion matrices:
  - Model predictions vs. API-based class inference
  - Model predictions vs. original dataset labels

---

## 🧰 Technologies Used

| Technology           | Purpose                                                                   |
|----------------------|---------------------------------------------------------------------------|
| Python               | Core scripting language used across all modules                           |
| pandas               | Data loading, cleaning, filtering, merging (CSV and tabular processing)   |
| requests             | Making API calls to Wikipedia and TheDogAPI                               |
| os                   | File path normalization and directory handling                            |
| OpenCV               | Image loading, enhancement, and pre-processing                            |
| YOLOv5 (via PyTorch) | Object detection model used to classify dogs and cats in images           |
| torch.hub            | Easy access to pretrained YOLOv5 model                                    |
| Pillow               | Image enhancement (brightness, contrast, etc.)                            |
| matplotlib & seaborn | Data visualization — bar charts, heatmaps, and confusion matrices         |
| scikit-learn         | Evaluation metrics and confusion matrix generation                        |


---

## 📊 Sample Evaluation Results

| Sample Size | Evaluation Type                         | Accuracy | Precision | Recall | F1 Score |
|-------------|------------------------------------------|----------|-----------|--------|----------|
| 20          | Model vs API Class                       | 0.95     | 0.96      | 0.95   | 0.95     |
| 20          | Model vs Ground Truth Label              | 0.95     | 0.96      | 0.95   | 0.95     |
| 40          | Model vs API Class                       | 0.98     | 0.98      | 0.98   | 0.98     |
| 40          | Model vs Ground Truth Label              | 0.98     | 0.98      | 0.98   | 0.98     |
| 100         | Model vs API Class                       | 0.97     | 0.97      | 0.97   | 0.97     |
| 100         | Model vs Ground Truth Label              | 0.97     | 0.97      | 0.97   | 0.97     |
| ~7000       | Model vs API Class (full sample size)    | 0.94     | 0.94      | 0.94   | 0.94     |
| ~7000       | Model vs Ground Truth Label (full sample)| 0.94     | 0.94      | 0.94   | 0.94     |


---


## 🔧 Setup & Installation

```bash
# 1. Install Dependencies
pip install pandas numpy requests scikit-learn matplotlib seaborn opencv-python torch Pillow

# 2. Run the full pipeline
python classify_breeds_with_fallback.py


## 📂 Project Structure

```plaintext
vision_pipeline/
├── images/                      # Raw & processed image storage
├── api_utils.py                 # API-based breed classifier
├── classify_breeds_with_fallback.py  # Main script for full pipeline + evaluation
├── processor.py                 # YOLOv5 inference + preprocessing
├── filtering_meta_data.py       # Filters YOLO predictions to cat/dog only
├── filtering_ground_truth.py    # Maps breed names to categories in ground truth
├── metadata_cleaned.csv         # Processed YOLO output
├── ground_truth_with_class.csv  # Breed-based labels with category mapping
├── config.py                    # Path and directory setup

