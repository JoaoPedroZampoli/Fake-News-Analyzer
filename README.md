# Fake News Analyzer
Final project for the Artificial Intelligence course [2025.1] at ICT - Unifesp

Special thanks to my partner, [Luiza de Souza Ferreira](https://github.com/souza-luiza) for her contributions to this project

<img width="5760" height="3240" alt="Trabalho Final de IA (1)" src="https://github.com/user-attachments/assets/737c1c58-e73a-4510-aca1-55e5680b0375" />

## 📚 About

**Fake News Analyzer** is a machine learning project designed to classify news articles as real or fake. It leverages traditional ML algorithms and modern NLP techniques, including SVM, Naive Bayes, KNN, and BERT-based transformers.

## 🚀 Features

- Preprocessing and cleaning of news datasets
- Training and evaluation of multiple ML models
- Support for both classical ML and transformer-based models

<img width="1189" height="590" alt="image" src="https://github.com/user-attachments/assets/6615b73c-7fbf-4ddb-883e-d757798923d6" />


## 🛠️ Technologies

- Python 3.12+
- scikit-learn
- pandas, numpy
- TensorFlow & Hugging Face Transformers (for BERT)

## 📦 Project Structure
    Fake-News-Analyzer/
    │   .gitattributes
    │   Apresentação Final - Fake News Analyzer.pdf
    │   Fake News Analyzer.ipynb
    │   pre-processed.csv
    │   README.md
    │   Relatório - Detecção de Fake News usando NLP.pdf
    │
    └───modelos/
        │   knn_model.pkl
        │   knn_vectorizer.pkl
        │   naive_bayes_model.pkl
        │   naive_bayes_vectorizer.pkl
        │   svm_model.pkl
        │   svm_vectorizer.pkl
        │
        └───bert/
            ├───bert_model/
            │       config.json
            │       tf_model.h5
            │
            └───bert_tokenizer/
                    special_tokens_map.json
                    tokenizer_config.json
                    vocab.txt

## ⚡ Getting Started

1. **Clone the repository:**
    ```bash
    git clone https://github.com/JoaoPedroZampoli/Fake-News-Analyzer.git
    cd Fake-News-Analyzer
    ```

2. **Install dependencies**
    ```
    pip install numpy pandas matplotlib seaborn tensorflow transformers scikit-learn

3. **Create the models**
   - run Fake News Analyzer.ipynb cells with Python to create new models if wanted

## 📄 License
This project is for educational purposes.
