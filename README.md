# **Sentiment Classification Project**

## **Overview**

This project focuses on building and evaluating various neural network models for sentiment classification using the Rotten Tomatoes movie review dataset. The models implemented include:

- **Recurrent Neural Network (RNN)**
- **Bidirectional LSTM (BiLSTM)**
- **Bidirectional GRU (BiGRU)**
- **Convolutional Neural Network (CNN)**
- **BiLSTM with Attention Mechanism**

We utilize pre-trained GloVe embeddings to represent words and explore different strategies to handle out-of-vocabulary (OOV) words. The project demonstrates the impact of various enhancements on model performance and provides insights into effective techniques for text classification tasks.

---

## **Table of Contents**

- [Overview](#overview)
- [Project Structure](#project-structure)
- [Dataset](#dataset)
- [Requirements](#requirements)
- [Installation and Setup](#installation-and-setup)
- [Usage](#usage)
  - [1. Data Analysis and Preprocessing](#1-data-analysis-and-preprocessing)
  - [2. Model Training](#2-model-training)
  - [3. Model Enhancement and Evaluation](#3-model-enhancement-and-evaluation)
- [Model Architectures](#model-architectures)
- [Results](#results)


---

## **Project Structure**

```plaintext
sentiment-classification-project/
├── data/
│   └── glove.6B.300d.txt        # Pre-trained GloVe embeddings (not included)
├── notebooks/
│   ├── 1_data_analysis.ipynb    # Data analysis and preprocessing
│   ├── 2_model_training.ipynb   # Model training
│   └── 3_model_evaluation.ipynb # Model enhancement and evaluation
├── src/
│   ├── data_loader.py           # Data loading and preprocessing functions
│   ├── embeddings.py            # Functions for loading embeddings
│   ├── models/
│   │   ├── rnn_model.py         # RNN model implementation
│   │   ├── bilstm_model.py      # BiLSTM model implementation
│   │   ├── bigru_model.py       # BiGRU model implementation
│   │   ├── cnn_model.py         # CNN model implementation
│   │   └── bilstm_attention.py  # BiLSTM with attention mechanism
│   └── utils.py                 # Utility functions (training, evaluation)
├── vocab.pkl                    # Saved vocabulary
├── requirements.txt             # Required Python packages
├── .gitignore                   # Git ignore file
└── README.md                    # Project documentation
```

---

## **Dataset**

The project uses the **Rotten Tomatoes movie review dataset** from the [Hugging Face Datasets](https://huggingface.co/datasets). The dataset consists of movie reviews labeled as positive or negative.

---

## **Requirements**

- Python 3.7 or higher
- Packages specified in `requirements.txt`

---

## **Installation and Setup**

### **1. Clone the Repository**

```bash
git clone https://github.com/yourusername/sentiment-classification-project.git
cd sentiment-classification-project
```

### **2. Create a Virtual Environment**

```bash
python3 -m venv venv
source venv/bin/activate  # On Windows, use: venv\Scripts\activate
```

### **3. Install Dependencies**

```bash
pip install --upgrade pip
pip install -r requirements.txt
```

### **4. Download GloVe Embeddings**

- Download the `glove.6B.zip` file from [GloVe Website](https://nlp.stanford.edu/projects/glove/).
- Extract `glove.6B.300d.txt` into the `data/` directory.

### **5. Download NLTK Data**

```python
import nltk
nltk.download('punkt')
```

### **6. Set Up the Project Structure**


## **Usage**

The project is organized into three main Jupyter notebooks located in the `notebooks/` directory. Each notebook corresponds to a specific part of the assignment.

### **1. Data Analysis and Preprocessing**

- **Notebook**: `1_data_analysis.ipynb`
- **Purpose**: Perform exploratory data analysis, build the vocabulary, and handle out-of-vocabulary words.
- **Steps**:
  - Load the Rotten Tomatoes dataset.
  - Preprocess the text data (tokenization, lowercasing, removing non-alphabetic characters).
  - Build the vocabulary from the training data.
  - Analyze the vocabulary size and OOV words.
  - Implement strategies to mitigate OOV issues.

### **2. Model Training**

- **Notebook**: `2_model_training.ipynb`
- **Purpose**: Train a basic RNN model for sentiment classification and evaluate its performance.
- **Steps**:
  - Load data loaders for training, validation, and testing.
  - Load pre-trained GloVe embeddings and create an embedding matrix.
  - Initialize the RNN model with static embeddings.
  - Define the loss function and optimizer.
  - Train the model and record training losses and validation accuracies.
  - Plot validation accuracy over epochs.
  - Evaluate the model on the test set.

### **3. Model Enhancement and Evaluation**

- **Notebook**: `3_model_evaluation.ipynb`
- **Purpose**: Implement various enhancements to improve model performance and compare results.
- **Enhancements**:
  - Updating word embeddings during training.
  - Handling OOV words by initializing embeddings and updating them.
  - Implementing BiLSTM and BiGRU models.
  - Implementing a CNN model.
  - Incorporating an attention mechanism into the BiLSTM model.
- **Steps**:
  - Modify models to include enhancements.
  - Train each model variant and record performance metrics.
  - Evaluate each model on the test set.
  - Compare results and provide observations.

---

## **Model Architectures**

### **1. Recurrent Neural Network (RNN)**

- **Description**: A simple RNN model that processes input sequences and outputs a fixed-size representation.
- **Enhancements**:
  - Updating word embeddings during training.
  - Handling OOV words.

### **2. Bidirectional LSTM (BiLSTM)**

- **Description**: An LSTM model that processes input sequences in both forward and backward directions.
- **Enhancements**:
  - Incorporating an attention mechanism to focus on important words.

### **3. Bidirectional GRU (BiGRU)**

- **Description**: A GRU model similar to BiLSTM but with a simpler architecture.

### **4. Convolutional Neural Network (CNN)**

- **Description**: A CNN model that applies convolutional filters to extract local n-gram features from text.

---

## **Results**

### **Test Accuracies of Different Models**

| **Model**                   | **Test Accuracy** |
|-----------------------------|-------------------|
| RNN (static embeddings)     | 50.00%            |
| RNN (updated embeddings)    | 51.41%            |
| BiGRU                       | 68.11%            |
| BiLSTM                      | 67.17%            |
| CNN                         | 67.45%            |
| BiLSTM + Attention          | 77.49%            |

### **Observations**

- Updating word embeddings slightly improves performance in simple RNNs.
- Advanced architectures like BiGRU, BiLSTM, and CNN significantly outperform the basic RNN model.
- Incorporating an attention mechanism into the BiLSTM model yields the highest accuracy, demonstrating the effectiveness of attention in capturing important features.


---


## **References**

- [PyTorch Documentation](https://pytorch.org/docs/stable/index.html)
- [NLTK Documentation](https://www.nltk.org/)
- [Hugging Face Datasets](https://huggingface.co/docs/datasets/index)

