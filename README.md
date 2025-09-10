# Sino-Nôm Image Classification

## Overview
This project focuses on classifying Sino-Nôm characters from images using a Vision Transformer (ViT) model. Sino-Nôm is a historical script used in Vietnam, combining Chinese (Sino) characters and Vietnam-specific (Nôm) characters. The goal is to develop an accurate and efficient model to recognize and classify these characters from scanned or photographed documents.

## Features
- **Vision Transformer Model**: Leverages the power of Vision Transformers for high-accuracy image classification.
- **Sino-Nôm Dataset**: Uses a curated dataset of Sino-Nôm character images for training and evaluation.
- **Preprocessing Pipeline**: Includes image preprocessing techniques to handle noise, distortions, and variations in historical documents.
- **Evaluation Metrics**: Provides detailed metrics such as accuracy, precision, recall, and F1-score for model performance.

## Installation
1. Clone the repository:
   ```bash
   git clone https://github.com/imthanhluan203/Sino-N-m-Image-Classification.git
   cd Sino-N-m-Image-Classification
   ```
2. Install the required dependencies:
   ```bash
   pip install -r requirements.txt
   ```
3. Ensure you have Python 3.8+ and the necessary libraries (e.g., PyTorch, Transformers, OpenCV) installed.

## Dataset
- The dataset consists of images of Sino-Nôm characters extracted from historical Vietnamese texts.
- Images are preprocessed to standardize size, remove noise, and enhance quality.
- The dataset is split into training, validation, and test sets.
- Due to the large size of the dataset (7.9GB), the dataset link is provided in the Dataset.txt file.

## Usage
1. **Prepare the Dataset**:
   - Place your dataset in the `data/` directory or update the data path in the configuration file.
2. **Train the Model**:
   ```bash
   python train.py --config config.yaml
   ```
3. **Evaluate the Model**:
   ```bash
   python evaluate.py --model_path path/to/saved/model
   ```
4. **Inference**:
   ```bash
   python inference.py --image_path path/to/image
   ```

## Model Architecture
- **Vision Transformer (ViT)**: The model divides images into patches, processes them through transformer layers, and outputs class probabilities for Sino-Nôm characters.
- Pretrained weights from Hugging Face's Transformers library are fine-tuned on the Sino-Nôm dataset.

## Requirements
- Python 3.8+
- PyTorch
- Transformers (Hugging Face)
- OpenCV
- NumPy
- Matplotlib (for visualization)

Install dependencies using:
```bash
pip install torch transformers opencv-python numpy matplotlib
```

## Results
- The model achieves high accuracy on the test set (detailed results in `results/` directory).
- Visualizations of model predictions and attention maps are available for analysis.

