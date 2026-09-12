# **Uvular Disease Detection using Deep Learning 🩺**

## **About the Project**

This project uses **Deep Learning and Computer Vision** to classify uvular disease images into **2 classes**.

A **MobileNetV2** model with **ImageNet pretrained weights** is used for image classification.

The project also uses **Grad-CAM (Gradient-weighted Class Activation Mapping)** to visualize the regions of the image that influenced the model's prediction.

## **Technologies Used**

- **Python**
- **TensorFlow / Keras**
- **OpenCV**
- **NumPy**
- **Pandas**
- **Matplotlib**
- **Seaborn**
- **Scikit-learn**
- **Google Colab**

## **Dataset**

The dataset contains images related to **Uvular Disease**.

The dataset is divided into:

- **Training Data**
- **Testing Data**

The training data contains **2 classes**.

## **Data Preprocessing**

The following preprocessing steps were performed:

- Extracted the dataset from a ZIP file
- Checked images for corruption
- Removed corrupted images if found
- Resized images to **160 × 160 pixels**
- Rescaled pixel values using **1/255**
- Applied **data augmentation**
- Used **20% of the training data for validation**

## **Data Augmentation**

The following techniques were used:

- **Rotation**
- **Zoom**
- **Horizontal Flip**

## **Model Used**

The project uses **MobileNetV2**, a pretrained Convolutional Neural Network.

The pretrained layers are frozen and additional layers are added for classification:

- **MobileNetV2**
- **Global Average Pooling**
- **Dense Layer (128 neurons)**
- **Dropout (0.4)**
- **Sigmoid Output Layer**

The model has approximately **2.42 million parameters**, with about **164K trainable parameters**.

## **Training**

The model was trained for **10 epochs** using:

- **Optimizer:** Adam
- **Learning Rate:** 0.0001
- **Loss Function:** Binary Crossentropy
- **Metric:** Accuracy
- **Batch Size:** 16

**Early Stopping** was also used to prevent unnecessary training.

## **Model Performance**

The model achieved:

- **Training Accuracy:** 94.45%
- **Validation Accuracy:** 95.73%
- **Validation Loss:** 0.1334

## **Model Evaluation**

The model was evaluated using:

- **Accuracy**
- **Precision**
- **Recall**
- **F1-Score**
- **Confusion Matrix**
- **Classification Report**

## **Grad-CAM Visualization 🔍**

**Grad-CAM (Gradient-weighted Class Activation Mapping)** was used to understand which regions of the image contributed to the model's prediction.

The Grad-CAM process:

1. Takes an input image
2. Generates the model prediction
3. Calculates the important feature regions
4. Creates a **heatmap**
5. Overlays the heatmap on the original image

This helps provide **visual interpretability** for the deep learning model.


## ** Architecture Diagram

<img width="1000" height="500" alt="uvula_mobilenet_architecture" src="https://github.com/user-attachments/assets/244debdb-b3b1-4fae-80b4-a77db97b6310" />



## **Project Workflow**

```text
Dataset
   ↓
Extract Dataset
   ↓
Check & Clean Images
   ↓
Image Preprocessing
   ↓
Data Augmentation
   ↓
MobileNetV2
   ↓
Model Training
   ↓
Validation
   ↓
Model Evaluation
   ↓
Disease Classification
   ↓
Grad-CAM Visualization
