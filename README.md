# Deep Learning Assignment 04

This repository contains two deep learning experiments demonstrating neural network architectures for classification tasks.

## Experiments Overview

### Experiment 1: Binary Classification with Dense Neural Network

**Objective**: Build and train a dense neural network for binary income classification on the Adult dataset.

**Dataset**: Adult Census Income Dataset
- Features: 14 attributes including age, workclass, education, marital status, occupation, etc.
- Target: Binary classification (income ≤ $50K or > $50K)
- Size: ~30,000 samples after preprocessing

**Model Architecture**:
```
Input Layer → Dense(64, relu) → Dropout(0.3) → Dense(32, relu) 
→ Dropout(0.3) → Dense(1, sigmoid)
```

**Key Preprocessing Steps**:
- Removed rows with missing values (marked as '?')
- Binary encoded target variable (0 = ≤50K, 1 = >50K)
- One-hot encoded categorical features
- Standardized numerical features using StandardScaler
- 80-20 train-test split

**Results**:
- Test Accuracy: 50%
- Training Configuration:
  - Optimizer: Adam
  - Loss Function: Binary Crossentropy
  - Epochs: 15
  - Batch Size: 32
  - Validation Split: 20%

**Files**: 
- `Capstone_Assignment.ipynb` - Complete implementation and analysis

---

### Experiment 2: Multi-class Image Classification with CNN

**Objective**: Build and train a Convolutional Neural Network for clothing item classification on the Fashion MNIST dataset.

**Dataset**: Fashion MNIST
- 60,000 training samples
- 10,000 test samples
- 10 clothing categories (T-shirt, Trouser, Pullover, Dress, Coat, Sandal, Shirt, Sneaker, Bag, Ankle boot)
- Image size: 28×28 grayscale

**Model Architecture**:
```
Conv2D(32, 3×3, relu) → MaxPooling(2×2)
→ Conv2D(64, 3×3, relu) → MaxPooling(2×2)
→ Flatten → Dense(64, relu) → Dropout(0.3)
→ Dense(10, softmax)
```

**Key Preprocessing Steps**:
- Normalized pixel values to [0, 1] range (divided by 255)
- Reshaped images from (28, 28) to (28, 28, 1) for channel dimension
- 80-20 train-validation split

**Results**:
- Test Accuracy: 90.41%
- Detailed Classification Metrics:
  - Classes with high precision: Shirt (1.0), Sandal (0.99), Ankle boot (0.99)
  - Classes with lower performance: Coat (0.71), T-shirt (0.86)
  - Macro-average F1-score: 0.90

**Training Configuration**:
- Optimizer: Adam
- Loss Function: Sparse Categorical Crossentropy
- Epochs: 10
- Batch Size: 32

**Visualizations Generated**:
- Confusion Matrix (10×10 heatmap showing per-class performance)
- Training/Validation Accuracy curves
- Training/Validation Loss curves

---

## Requirements

```
tensorflow>=2.8.0
keras>=2.8.0
pandas>=1.3.0
numpy>=1.20.0
scikit-learn>=0.24.0
matplotlib>=3.3.0
seaborn>=0.11.0
```

## Installation

```bash
pip install -r requirements.txt
```

## Usage

### Running Experiment 1 (Adult Dataset):
```python
# Open Capstone_Assignment.ipynb in Jupyter or Colab
# Follow cells sequentially from data loading through evaluation
```

### Running Experiment 2 (Fashion MNIST):
```python
# Cells in the same notebook starting from "EXPERIMENT 2: CNN for Image Classification"
# The Fashion MNIST dataset is automatically downloaded
```

## Key Findings

### Experiment 1 Observations:
- Limited test accuracy suggests potential data quality issues or model underfitting
- Validation accuracy remains near 0 throughout training, indicating possible batch size incompatibility
- Model requires hyperparameter tuning and more comprehensive feature engineering

### Experiment 2 Observations:
- Strong overall performance (90.41% accuracy) demonstrates CNN effectiveness for image classification
- Perfect precision on some classes but lower recall on others suggests class imbalance
- Confusion primarily occurs between similar clothing items
- Model converges well within 10 epochs without significant overfitting

## Model Comparison

| Aspect | Experiment 1 | Experiment 2 |
|--------|-------------|-------------|
| Architecture | Dense NN | CNN |
| Dataset | Tabular (Adult) | Images (Fashion MNIST) |
| Classes | 2 (Binary) | 10 (Multi-class) |
| Test Accuracy | 50% | 90.41% |
| Primary Issue | Low accuracy | Strong performance |

## Future Improvements

### Experiment 1:
- Implement feature engineering and domain-specific preprocessing
- Hyperparameter tuning (learning rate, batch size, layer sizes)
- Class imbalance handling (weighted loss, oversampling)
- Ensemble methods or different architectures
- Data augmentation and validation strategy revision

### Experiment 2:
- Implement data augmentation (rotation, zoom, shift)
- Use transfer learning (pre-trained models like MobileNet)
- Regularization techniques (L1/L2, batch normalization)
- Learning rate scheduling
- Ensemble of multiple models

## Author Notes

This assignment demonstrates the application of two distinct neural network architectures:
- **Dense networks** for structured/tabular data processing
- **Convolutional networks** for image data processing

The stark difference in performance highlights the importance of choosing appropriate architectures for specific data types.

## References

- TensorFlow/Keras Documentation: https://www.tensorflow.org/
- Adult Dataset: https://archive.ics.uci.edu/ml/datasets/Adult
- Fashion MNIST: https://github.com/zalandoresearch/fashion-mnist

## License

This project is for educational purposes as part of a deep learning course assignment.
