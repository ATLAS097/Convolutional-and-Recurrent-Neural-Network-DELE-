# Convolutional and Recurrent Neural Network Project (DELE)

**Author**: Kaung Myat San
**Institution**: Singapore Polytechnic  
**Date**: May 2025

## 📖 Project Overview

This project demonstrates the implementation and application of deep learning techniques using both **Convolutional Neural Networks (CNNs)** and **Recurrent Neural Networks (RNNs)** for two distinct machine learning tasks:

- **Part A**: Image classification of vegetables using CNNs
- **Part B**: Sentiment analysis of movie reviews using RNNs

## 🗂️ Project Structure

```
├── DELE_CA1_Brief.pdf          # Project assignment brief
├── README.md                   # This file
├── Slides.pdf                  # Project presentation slides
├── Part-A/                     # CNN Image Classification
│   ├── Part-A.ipynb           # Main notebook for vegetable classification
│   ├── Part-A-ATLAS.ipynb     # ATLAS version of the notebook
│   ├── Part-A.html            # HTML export of the notebook
│   ├── Dataset_A/              # Vegetable image dataset
│   │   ├── train/              # Training images
│   │   ├── test/               # Testing images
│   │   └── validation/         # Validation images
│   ├── 101_models/             # Saved models and histories (11 classes)
│   └── 23_models/              # Saved models and histories (13 classes)
└── Part-B/                     # RNN Sentiment Analysis
    ├── Part-B.ipynb           # Main notebook for sentiment analysis
    ├── Movie reviews.csv       # Original movie reviews dataset
    ├── augmented_data.xlsx     # Augmented dataset
    ├── best_classification_models/  # Classification models
    ├── best_reg_stem_models/   # Regression models (stemmed)
    └── best_regression_models/ # Regression models
```

## 🥬 Part A: Vegetable Classification using CNNs

### Dataset Information
- **Image Size**: 224 × 224 pixels (RGB)
- **Classes**: 11 grouped classes / 13 individual classes
- **Vegetables**: Bean, Bitter Gourd, Bottle Gourd, Cucumber, Brinjal, Broccoli, Cauliflower, Cabbage, Capsicum, Carrot, Radish, Potato, Pumpkin, Tomato

### Key Features
- **Data Preprocessing**: Image normalization, augmentation, and resizing
- **Model Architecture**: Custom CNN architectures optimized for vegetable classification
- **Class Grouping**: Strategic grouping of visually similar vegetables
- **Model Evaluation**: Comprehensive metrics including accuracy, loss, and confusion matrices

### Models Saved
- **11-class models**: `101_models/` directory contains 16 different model variations
- **13-class models**: `23_models/` directory contains 14 different model variations
- **History files**: Training history saved as `.pkl` files for analysis

### Data Quality Issues Identified
- Mislabeled images in training data (carrots labeled as beans)
- Cross-contamination in test folders (pumpkin/tomato confusion)

## 🎬 Part B: Movie Review Sentiment Analysis using RNNs

### Dataset Information
- **Source**: Movie reviews dataset
- **Task**: Predict sentiment scores of movie reviews
- **Approach**: Both classification and regression models

### Key Features
- **Text Preprocessing**: Comprehensive NLP pipeline including:
  - Contraction expansion
  - Stopword removal
  - Stemming and lemmatization
  - Text normalization
- **Data Augmentation**: Using contextual word embeddings for text augmentation
- **Model Types**: 
  - Classification models for sentiment categories
  - Regression models for sentiment scores
- **Text Variants**: Models trained on both stemmed and non-stemmed text

### Models Saved
- **Classification Models**: Stored in `best_classification_models/`
- **Regression Models**: Stored in `best_regression_models/`
- **Stemmed Text Models**: Stored in `best_reg_stem_models/`

## 🛠️ Technical Stack

### Libraries and Frameworks
- **Deep Learning**: TensorFlow, Keras
- **Data Processing**: NumPy, Pandas
- **Computer Vision**: OpenCV, PIL
- **NLP**: NLTK, TextBlob, Contractions
- **Text Augmentation**: nlpaug
- **Visualization**: Matplotlib, Seaborn
- **Utilities**: Pickle, OS, Warnings

### Model Architecture Highlights
- **CNN Models**: Custom architectures with multiple convolutional layers, pooling, and dropout
- **RNN Models**: LSTM/GRU-based architectures for sequence processing
- **Transfer Learning**: Potential use of pre-trained models for enhanced performance

## 📊 Results and Performance

### Part A - CNN Results
- Multiple model variations tested with different class groupings
- Best models saved with corresponding training histories
- Performance metrics tracked across 16 different model configurations

### Part B - RNN Results
- Separate models for classification and regression tasks
- Comparison between stemmed and non-stemmed text processing
- Multiple model architectures explored for optimal performance

## 🚀 Getting Started

### Prerequisites
```bash
pip install tensorflow
pip install numpy pandas matplotlib seaborn
pip install nltk opencv-python
pip install nlpaug contractions
pip install jupyter notebook
```

### Running the Notebooks
1. **Part A - Vegetable Classification**:
   ```bash
   cd Part-A
   jupyter notebook Part-A.ipynb
   ```

2. **Part B - Sentiment Analysis**:
   ```bash
   cd Part-B
   jupyter notebook Part-B.ipynb
   ```

## 📁 Model Files

### Part A Models
- **Format**: `.h5` (Keras model format)
- **Naming Convention**: `best_[classes]_model-[version].h5`
- **History Files**: `.pkl` files containing training history

### Part B Models
- **Format**: `.h5` (Keras model format)
- **Categories**: Classification vs Regression models
- **Text Processing**: Separate models for different preprocessing approaches

## 🔍 Key Insights

### Part A Insights
- Class grouping strategy improved model performance for visually similar vegetables
- Data quality issues significantly impact model performance
- Image preprocessing and augmentation are crucial for generalization

### Part B Insights
- Text preprocessing pipeline significantly affects model performance
- Data augmentation techniques can improve model robustness
- Both classification and regression approaches have their merits for sentiment analysis

## 📈 Future Improvements

1. **Data Quality**: Address mislabeling issues in Part A dataset
2. **Model Architecture**: Experiment with more advanced architectures (ResNet, BERT)
3. **Ensemble Methods**: Combine multiple models for better performance
4. **Cross-Validation**: Implement k-fold cross-validation for more robust evaluation
5. **Hyperparameter Tuning**: Systematic optimization of model parameters

## 📚 References

- TensorFlow Documentation
- Keras API Reference
- NLTK Documentation
- Computer Vision and NLP best practices

---

**Note**: This project was completed as part of the DELE coursework at Singapore Polytechnic, demonstrating practical applications of deep learning in both computer vision and natural language processing domains.
