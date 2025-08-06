# 📰 Fake News Analyzer - Streamlit Application

Uma aplicação web abrangente para detecção de fake news usando múltiplos modelos de machine learning incluindo KNN, Naive Bayes, SVM e BERT.

## 🚀 Funcionalidades

- **Múltiplos Modelos ML**: Análise individual de cada modelo (KNN, Naive Bayes, SVM, BERT)
- **Interface Web Interativa**: Interface amigável usando Streamlit
- **Análise em Tempo Real**: Detecção instantânea de fake news
- **Estatísticas Detalhadas**: Visualização de acurácia, tempo de execução e matrizes de confusão para cada modelo
- **Visualizações Interativas**: Gráficos interativos mostrando performance dos modelos
- **Comparação de Modelos**: Comparação lado a lado de todos os modelos
- **Teste Individual**: Teste cada modelo separadamente com textos personalizados

## 🛠️ Modelos Utilizados

1. **KNN (K-Nearest Neighbors)**
   - Bag of Words: Acurácia ~85%
   - TF-IDF: Acurácia ~88%

2. **Naive Bayes (Multinomial)**
   - Bag of Words: Acurácia ~92%
   - TF-IDF: Acurácia ~94%

3. **SVM (Support Vector Machine)**
   - Bag of Words: Acurácia ~91%
   - TF-IDF: Acurácia ~95%

4. **BERT (Transformer)**
   - Embeddings contextuais: Acurácia ~96%

## 📋 Pré-requisitos

- Python 3.8 ou superior
- Modelos treinados (na pasta `../modelos/`)
- Dataset pré-processado (`../pre-processed.csv`)

## 🔧 Instalação

1. Clone o repositório:
```bash
git clone https://github.com/JoaoPedroZampoli/Fake-News-Analyzer.git
cd Fake-News-Analyzer/streamlit
```

2. Instale as dependências:
```bash
pip install -r requirements.txt
```

3. Certifique-se de que os modelos estão na pasta correta:
```
modelos/
├── knn_model.pkl
├── knn_vectorizer.pkl
├── naive_bayes_model.pkl
├── naive_bayes_vectorizer.pkl
├── svm_model.pkl
├── svm_vectorizer.pkl
└── bert/
    ├── bert_pytorch_model.pth
    ├── label_encoder.pkl
    └── bert_tokenizer/
```

## 🚀 Como Executar

Execute a aplicação Streamlit:
```bash
streamlit run ProjectPage.py
```

A aplicação será aberta automaticamente no seu navegador em `http://localhost:8501`

## 📊 Páginas da Aplicação

### 🏠 Visão Geral
- Introdução ao projeto
- Estatísticas do dataset
- Distribuição das classes (Real vs Fake)

### 🤖 Testar Modelos
- Teste individual de cada modelo
- Input de texto personalizado
- Visualização de confiança e probabilidades

### � Comparação de Modelos
- Gráficos comparativos de acurácia
- Análise de tempo de execução
- Tabela detalhada de performance

### 📈 Estatísticas Detalhadas
- Análise aprofundada de cada modelo
- Comparação entre Bag of Words e TF-IDF
- Recomendações de melhor configuração

## 🎯 Como Usar

1. **Navegação**: Use a sidebar para navegar entre as páginas
2. **Teste de Modelos**: 
   - Vá para "Testar Modelos"
   - Escolha um modelo
   - Digite ou cole o texto da notícia
   - Clique em "Analisar Notícia"
3. **Comparação**: Visite "Comparação de Modelos" para ver performance geral
4. **Detalhes**: Use "Estatísticas Detalhadas" para análise aprofundada

## 📝 Estrutura do Código

- `ProjectPage.py`: Aplicação principal Streamlit
- `requirements.txt`: Dependências necessárias
- Modelos carregados dinamicamente da pasta `../modelos/`

## 🔍 Funcionalidades Técnicas

- **Cache de Dados**: Uso de `@st.cache_data` e `@st.cache_resource` para performance
- **Predições em Tempo Real**: Suporte a todos os 4 modelos
- **Visualizações Plotly**: Gráficos interativos e responsivos
- **CSS Customizado**: Interface moderna e profissional
- **Tratamento de Erros**: Mensagens informativas em caso de problemas

## 🎨 Customização

O arquivo inclui CSS customizado para:
- Cards de modelos estilizados
- Cores baseadas na acurácia (verde/amarelo/vermelho)
- Layout responsivo
- Tipografia melhorada

## 🚨 Problemas Conhecidos

- BERT pode ser lento em máquinas sem GPU
- Alguns modelos podem requerer versões específicas do scikit-learn
- TensorFlow warnings são esperados para o BERT (use PyTorch quando possível)

## 📄 Licença

Este projeto está sob licença MIT. Veja o arquivo LICENSE para mais detalhes.

## 📋 Requirements

- Python 3.8+
- Streamlit
- scikit-learn
- PyTorch
- Transformers (Hugging Face)
- Pandas, NumPy
- Matplotlib, Seaborn, Plotly

## 🔧 Installation

1. **Clone the repository** (if not already done):
   ```bash
   git clone <repository-url>
   cd Trabalho-de-IA/streamlit
   ```

2. **Install dependencies**:
   ```bash
   pip install -r requirements.txt
   ```

3. **Ensure model files are present** in the `../modelos/` directory:
   - `knn_model.pkl` & `knn_vectorizer.pkl`
   - `naive_bayes_model.pkl` & `naive_bayes_vectorizer.pkl`
   - `svm_model.pkl` & `svm_vectorizer.pkl`
   - `bert/bert_pytorch_model.pth`
   - `bert/label_encoder.pkl`
   - `bert_tokenizer/` (directory with tokenizer files)

## 🚀 Running the Application

### Option 1: Using Python Script
```bash
python run_app.py
```

### Option 2: Using Batch File (Windows)
```bash
run_app.bat
```

### Option 3: Direct Streamlit Command
```bash
streamlit run ProjectPage.py
```

The application will start and automatically open in your default browser at `http://localhost:8501`

## 📱 How to Use

1. **Enter Text**: Type or paste a news article in the text area, or upload a text file
2. **Analyze**: Click the "🔍 Analyze Article" button
3. **View Results**: 
   - See the overall consensus (Real/Fake News)
   - Check individual model predictions
   - View confidence scores and visualizations
4. **Interpret**: Use the confidence charts to understand model agreement

## 📊 Understanding the Results

### Overall Consensus
- Based on majority vote from all 4 models
- **REAL NEWS**: Likely legitimate content
- **FAKE NEWS**: Potentially misleading or false information

### Individual Model Predictions
- **KNN**: Shows prediction and confidence score
- **Naive Bayes**: Shows prediction and confidence score  
- **SVM**: Shows prediction and decision function value
- **BERT**: Shows prediction and probability score

### Visualizations
- **Confidence Chart**: Bar chart showing each model's confidence
- **Prediction Summary**: Shows how many models predicted Real vs Fake

## 🔍 Tips for Best Results

- Use complete articles rather than just headlines
- Ensure text is in Portuguese or English (models were trained on Portuguese data)
- Remove excessive formatting or special characters
- Longer articles generally provide more reliable predictions

## 🛠️ Troubleshooting

### Common Issues

1. **Models not loading**: 
   - Check that all model files are in the correct `../modelos/` directory
   - Ensure file permissions allow reading

2. **BERT model errors**:
   - Make sure PyTorch is installed correctly
   - Verify the BERT tokenizer directory exists and contains all files

3. **Import errors**:
   - Install all requirements: `pip install -r requirements.txt`
   - Check Python version compatibility

### File Structure
```
streamlit/
├── ProjectPage.py          # Main application
├── run_app.py             # Python runner script
├── run_app.bat            # Windows batch runner
├── requirements.txt       # Dependencies
└── README.md             # This file

../modelos/               # Model files directory
├── knn_model.pkl
├── knn_vectorizer.pkl
├── naive_bayes_model.pkl
├── naive_bayes_vectorizer.pkl
├── svm_model.pkl
├── svm_vectorizer.pkl
└── bert/
    ├── bert_pytorch_model.pth
    ├── label_encoder.pkl
    └── bert_tokenizer/
        ├── special_tokens_map.json
        ├── tokenizer_config.json
        └── vocab.txt
```

## 📝 Notes

- The application uses caching to improve performance when loading models
- Models are loaded once when the app starts and reused for all predictions
- Analysis time is typically under 1 second for most articles
- The BERT model provides the most sophisticated analysis but takes slightly longer

## 🤝 Contributing

Feel free to submit issues, feature requests, or improvements to enhance the application!
