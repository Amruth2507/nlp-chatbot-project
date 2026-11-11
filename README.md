# Chatbot Arena: Large Language Model Evaluation and Prediction
![Python](https://img.shields.io/badge/Python-3.9+-blue.svg)
![Scikit-Learn](https://img.shields.io/badge/Machine%20Learning-Scikit--Learn-orange)
![Pandas](https://img.shields.io/badge/Data%20Processing-Pandas-green)

A comprehensive Machine Learning pipeline designed to evaluate Large Language Models (LLMs) based on human preference data. This project processes thousands of real-world interactions from the Chatbot Arena, extracts meaningful textual embeddings, and trains robust models to predict both the winner of an LLM battle and the objective difficulty of user prompts.

## 🎯 Project Objectives
The objective of this repository is deeply rooted in computational linguistics and performance evaluation:
1. **Task A (Winner Prediction):** Train a classification model to accurately predict which LLM (`model_a` or `model_b`) won a conversation based on embedding cosine similarity matrices, prompt lengths, and response vector variations.
2. **Task B (Hardness Prediction):** Train a Ridge regression model to objectively evaluate the difficulty/complexity (scaled 1-10) of a user's prompt by analyzing text lengths and embedding spaces.

## 📂 Project Structure
```text
nlp-chatbot-project/
├── data/                  # Original datasets, text embeddings (.npy), and labels
├── notebooks/             # Core computational pipeline scripts
│   ├── 01_eda.py          # Exploratory Data Analysis & visual metrics
│   ├── 02_task_a.py       # Logistic Regression classification pipeline
│   ├── 03_task_b.py       # Ridge Regression hardness predictor
│   └── 04_submission.py   # Final test-set inference and CSV generator
├── outputs/               # Visualized metrics, heatmaps, and final predictions
├── download_data.py       # Parquet ingestion and text extraction pipeline
└── README.md              # Project documentation
```

## 🛠️ Installation & Setup
To run this pipeline locally, ensure you have Python 3.9+ installed.

1. **Clone the repository:**
   ```bash
   git clone https://github.com/Amruth2507/nlp-chatbot-project.git
   cd nlp-chatbot-project
   ```

2. **Set up a Virtual Environment (Recommended):**
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```

3. **Install Dependencies:**
   Ensure you have the required science stack libraries:
   ```bash
   pip install pandas numpy scikit-learn matplotlib seaborn pyarrow
   ```

## 🚀 Running the Pipeline
The project is strictly modular. The pipeline must be executed in order:

### 1. Data Ingestion & Embeddings
Extracts structural Python objects from native `.parquet` formatting and generates lightweight Truncated SVD embeddings.
```bash
python download_data.py
```

### 2. Exploratory Data Analysis
Generates Win-Rates arrays, Elo Ratings models, and difficulty metrics to the `outputs/` directory.
```bash
python notebooks/01_eda.py
```

### 3. Training & Validation
Cross-validates, conducts hyperparameter GridSearches, and isolates performance metrics for Task A and Task B.
```bash
python notebooks/02_task_a.py
python notebooks/03_task_b.py
```

### 4. Inferencing & Final Submission
Applies heavily-tuned parameters against an unseen dataset structure and builds the final Kaggle-ready output context.
```bash
python notebooks/04_submission.py
```

## 📈 Key Findings
- **Elo Performance Alignment:** Derived Win-Rates seamlessly mirror actual Elo Ratings distributions across LLMs.
- **Multimodal Embedding Influence:** Comparing cosine distances between `prompt-response_a` and `prompt-response_b` provides a highly accurate heuristic for LLM victory prediction.
- **Hardness Curves:** Prompt logic heavily correlates with the `prompt_len` feature and text matrices, establishing a firm objective complexity standard generalized strictly off the user's initial token sequence.

---
*Developed automatically, driven by deep natural language understanding pipelines.*
