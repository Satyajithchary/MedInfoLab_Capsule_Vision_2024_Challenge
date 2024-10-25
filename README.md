# MedInfoLab_Capsule_Vision_2024_Challenge

## 🌟 Overview

Welcome to our project on **Endoscopic Image Classification Using BiomedCLIP-PubMedBERT**! This repository contains our code and insights from the Capsule Vision 2024 Challenge, where we developed a multimodal approach to accurately classify abnormalities in Video Capsule Endoscopy (VCE) images to aid gastrointestinal diagnostics.

The project's primary objective is to streamline the diagnostic process by automating abnormality detection in VCE images, potentially reducing clinician workload and enhancing diagnostic precision.

## 🏆 Challenge Details

### Motivation

Video Capsule Endoscopy (VCE) captures thousands of gastrointestinal tract images, presenting a substantial challenge for manual interpretation. Traditional diagnostic workflows can be slow and are prone to errors due to fatigue. Our solution aims to automate this task, allowing for faster and more accurate classification of common abnormalities to improve clinical efficiency.

### Dataset

This project utilizes a **diverse dataset** of VCE frames provided by the Capsule Vision Challenge. The dataset encompasses **ten classes** representing different gastrointestinal abnormalities, including:

- **Angioectasia**
- **Bleeding**
- **Erosion**
- **Erythema**
- **Foreign Body**
- **Lymphangiectasia**
- **Polyp**
- **Ulcer**
- **Worms**
- **Normal**

The dataset, sourced from public (SEE-AI, KID, and Kvasir-Capsule) and private (AIIMS) collections, includes 37,607 training images and 16,132 validation images. Each class has its own dedicated folder, and metadata files list image paths and labels for organized retrieval.

## ⚙️ Methodology

Our solution is based on **BiomedCLIP-PubMedBERT**, a fine-tuned multimodal model that integrates:

- **Vision Transformer (ViT)**: Extracts detailed features from the endoscopic images.
- **PubMedBERT**: Processes text embeddings of abnormality classes for precise classification.

### Model Architecture

1. **Data Loading and Preprocessing**: VCE images are resized to 224x224 pixels and undergo augmentation (rotation, flipping) for robustness.
2. **Vision Transformer (ViT)**: The ViT processes images, extracting spatial features to produce high-dimensional image embeddings.
3. **Text Embedding with PubMedBERT**: Class labels are transformed into text embeddings, which are aligned with image embeddings.
4. **Image-Text Matching and Classification**: Using cross-modal similarity scoring, the model generates a probability for each class, selecting the one with the highest probability as its prediction.

### Training and Evaluation

The model was fine-tuned with batches of 32 images over 30 epochs. Performance was evaluated using multiple metrics, including **accuracy, precision, recall,** and **F1-score**. The fine-tuning enabled efficient learning and precise classification across the dataset's diverse classes.

## 📊 Results

The model demonstrated high accuracy across most abnormality classes, achieving:

- **Precision and Recall**: Balanced scores across categories, indicating minimal false positives/negatives.
- **F1 Score**: High scores, particularly in distinct classes like **Foreign Body** and **Normal**.
- **Challenges**: Slightly lower precision for visually similar classes, such as **Erosion** and **Ulcer**.

## 🚀 Future Directions

Our project opens doors for further enhancement:

- **Temporal Features**: Integrating sequential information in VCE frames for improved context.
- **Active Learning**: Leveraging human feedback to improve performance in uncertain cases.
- **Interpretability**: Incorporating heatmaps to help clinicians understand model predictions.

## 📂 Repository Structure

```bash
📂 Capsule_Vision_Challenge_2024
├── 📂 data                # Dataset and metadata files
├── 📂 notebooks           # Jupyter notebooks for experiments
├── 📂 src                 # Main scripts for data processing and model training
├── 📄 README.md           # Project documentation
└── 📄 LICENSE             # License details
```

## 🔗 Usage

1. **Clone the Repository**:
   ```bash
   git clone https://github.com/Satyajith/MedInfoLab_Capsule_Vision_2024_Challenge.git
   ```
2. **Setup Environment**:
   ```bash
   pip install -r requirements.txt
   ```
3. **Run Training**:
   ```bash
   python src/train_model.py
   ```

## 💼 Citation

If you find this work useful, please consider citing us:

```
@article{Ganapathy2024capsulevision,
  title={A Multimodal Approach for Endoscopic VCE Image Classification Using BiomedCLIP-PubMedBERT},
  author={Nagarajan Ganapathy, Podakanti Satyajith Chary, et al.},
  year={2024}
}
```

## 🎉 Acknowledgments

Thanks to Capsule Vision 2024 Challenge organizers, BiomedCLIP-PubMedBERT developers, and our collaborators for their invaluable support.

--- 

This README introduces the project and provides interactive and easy-to-follow instructions for users interested in running, understanding, or contributing to this research.
