# YOLO-Object-Detection-Model-on-BCCD-Dataset


# YOLO Object Detection on BCCD Dataset

## 📌 Project Overview
This project fine-tunes the **YOLOv10** object detection model on the **BCCD (Blood Cell Count and Detection) Dataset** to detect and classify blood cells. The model is trained and evaluated using **Google Colab**, with an interactive web app built using **Gradio** or **Streamlit** for real-time predictions. The final deployment is hosted on **Hugging Face Spaces**.

## 🚀 Features
- **Fine-tuned YOLOv10** for detecting RBCs, WBCs, and Platelets.
- **Image Preprocessing & Data Augmentation** (rotation, cropping, etc.).
- **Model Inference** with bounding boxes, class labels, and confidence scores.
- **Interactive Web App** for easy image uploads and predictions.
- **Performance Evaluation** using Precision-Recall metrics.
- **Deployed on Hugging Face Spaces**.

## 📂 Dataset
The **BCCD Dataset** is a widely used dataset for blood cell detection and classification, containing:
- **Red Blood Cells (RBCs)**
- **White Blood Cells (WBCs)**
- **Platelets**

📥 **Download the dataset**: [BCCD Dataset](https://github.com/Shenggan/BCCD_Dataset)

## 📦 Installation & Setup
### 1️⃣ Clone the Repository
```bash
git clone https://github.com/Sim14code/YOLO-Object-Detection-Model-on-BCCD-Dataset.git
cd YOLO-Object-Detection-Model-on-BCCD-Dataset
```

### 2️⃣ Install Dependencies
```bash
pip install -r requirements.txt
```

### 3️⃣ Prepare the Dataset
- Download the **BCCD dataset** from the given link.
- Extract the dataset and ensure the images and annotations are properly structured.
- Convert annotations if needed (e.g., VOC to YOLO format).

### 4️⃣ Train the Model in Google Colab
- Open the `train_yolo.ipynb` notebook in Google Colab.
- Mount Google Drive and upload the dataset.
- Configure the YOLO model settings (e.g., batch size, epochs, learning rate).
- Run the training script to fine-tune YOLOv10.
- Save the trained model weights for inference.

### 5️⃣ Perform Model Inference
- Use the `inference.ipynb` notebook to test the trained model.
- Load an image and run the detection pipeline.
- Display bounding boxes with class labels and confidence scores.

### 6️⃣ Run the Web App (Gradio)
-  **Gradio**:
```bash
python gradio_app.py
```
- Upload an image and visualize the detection results in the UI.

## 🎯 Model Performance
- **Evaluation Metrics**: Precision, Recall, mAP (Mean Average Precision)
- **Precision-Recall Table** included in the web app.
- **Visualized bounding boxes** on test images.


  ## The gradio link of the app works well.





🚀 **Happy Coding!**

