"""
02_task_a.py - Task A: Winner Prediction (fixed version of 02_task_a.ipynb)
Run with: python notebooks/02_task_a.py (from project root)
"""
import pandas as pd
import numpy as np
import ast
import os
from sklearn.model_selection import train_test_split, cross_val_score, GridSearchCV
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.metrics import accuracy_score, classification_report, ConfusionMatrixDisplay
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns

# Resolve data paths
DATA_DIR = '../data' if os.path.basename(os.getcwd()) == 'notebooks' else 'data'
OUT_DIR  = '../outputs' if os.path.basename(os.getcwd()) == 'notebooks' else 'outputs'
os.makedirs(OUT_DIR, exist_ok=True)

# ── Setup: Load Data ──
print("=" * 60)
print("Loading data ...")
df = pd.read_json(f'{DATA_DIR}/chatbot-arena-conversations.jsonl.gz', lines=True, compression='gzip')

# Extract text and lengths
def get_text(conversation, role):
    """Safely extract text from a conversation turn."""
    if isinstance(conversation, str):
        try:
            conversation = ast.literal_eval(conversation)
        except:
            return ''
    if not isinstance(conversation, list):
        return ''
    for turn in conversation:
        if isinstance(turn, dict) and turn.get('role') == role:
            return turn.get('content', '')
    return ''

df['prompt'] = df['conversation_a'].apply(lambda x: get_text(x, 'user'))
df['response_a'] = df['conversation_a'].apply(lambda x: get_text(x, 'assistant'))
df['response_b'] = df['conversation_b'].apply(lambda x: get_text(x, 'assistant'))

df['prompt_len'] = df['prompt'].str.len()
df['response_a_len'] = df['response_a'].str.len()
df['response_b_len'] = df['response_b'].str.len()

# Load scores (deduplicate to prevent row expansion on merge)
scores_df = pd.read_json(f'{DATA_DIR}/chatbot-arena-gpt3-scores.jsonl.gz', lines=True, compression='gzip')
scores_df = scores_df.drop_duplicates(subset='question_id', keep='first')
df = pd.merge(df, scores_df[['question_id', 'hardness', 'topic']], on='question_id', how='left')

# Drop invalid winner rows if any
df = df[df['winner'].isin(['model_a', 'model_b', 'tie', 'tie (bothbad)'])]

# Load embeddings - align indices with filtered df
emb_prompt_full = np.load(f'{DATA_DIR}/chatbot-arena-prompts-embeddings.npy')
emb_a_full = np.load(f'{DATA_DIR}/chatbot-arena-model_a_response-embeddings.npy')
emb_b_full = np.load(f'{DATA_DIR}/chatbot-arena-model_b_response-embeddings.npy')

# Use original indices to slice embeddings before resetting index
valid_indices = df.index.values
emb_prompt = emb_prompt_full[valid_indices]
emb_a = emb_a_full[valid_indices]
emb_b = emb_b_full[valid_indices]

df = df.reset_index(drop=True)

print(f"Data loaded: {len(df)} rows, embeddings shape: {emb_prompt.shape}")

# ── 5.1 - Create Features ──
print("\n" + "=" * 60)
print("5.1 - Creating features ...")

# 1) Text length features
lengths = df[['prompt_len', 'response_a_len', 'response_b_len', 'hardness']].fillna(0).values

# 2) Cosine similarity
def cosine_sim(a, b):
    dot = np.sum(a * b, axis=1)
    norm_a = np.linalg.norm(a, axis=1)
    norm_b = np.linalg.norm(b, axis=1)
    return dot / (norm_a * norm_b + 1e-10)

sim_a = cosine_sim(emb_prompt, emb_a).reshape(-1, 1)
sim_b = cosine_sim(emb_prompt, emb_b).reshape(-1, 1)

# 3) Embedding difference
emb_diff = emb_a - emb_b

# 4) Combine
X = np.hstack([lengths, sim_a, sim_b, emb_diff])
print("Feature matrix X shape:", X.shape)

# ── 5.2 - Encode Targets ──
print("\n" + "=" * 60)
print("5.2 - Encoding targets ...")
le = LabelEncoder()
y = le.fit_transform(df['winner'])

print("Classes mapping:")
for i, label in enumerate(le.classes_):
    print(f"{i}: {label}")

# ── 5.3 - Split Data ──
print("\n" + "=" * 60)
print("5.3 - Splitting data ...")
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

print("Training set size:", X_train.shape[0])
print("Test set size:", X_test.shape[0])

# ── 5.4 - Baseline Accuracy ──
print("\n" + "=" * 60)
print("5.4 - Baseline accuracy ...")
baseline_acc = df['winner'].value_counts().max() / len(df)
print(f"Baseline Accuracy (predicting most common class): {baseline_acc:.4f}")

# ── 5.5 - Logistic Regression Pipeline ──
print("\n" + "=" * 60)
print("5.5 - Training Logistic Regression ...")
pipeline = Pipeline([
    ('scaler', StandardScaler()),
    ('clf', LogisticRegression(max_iter=2000))
])

pipeline.fit(X_train, y_train)
y_pred = pipeline.predict(X_test)

print("Test Accuracy:", accuracy_score(y_test, y_pred))
print("\nClassification Report:\n", classification_report(y_test, y_pred, target_names=le.classes_))

# ── 5.6 - Cross-Validation ──
print("\n" + "=" * 60)
print("5.6 - Cross-validation ...")
cv_scores = cross_val_score(pipeline, X_train, y_train, cv=5, scoring='accuracy')

print("CV Accuracies:", cv_scores)
print(f"Mean CV Accuracy: {cv_scores.mean():.4f} +/- {cv_scores.std():.4f}")

train_pred = pipeline.predict(X_train)
print(f"Training Accuracy: {accuracy_score(y_train, train_pred):.4f}")

# ── 5.7 - Confusion Matrix ──
print("\n" + "=" * 60)
print("5.7 - Confusion matrix ...")
fig, ax = plt.subplots(figsize=(8, 6))
ConfusionMatrixDisplay.from_predictions(
    y_test, y_pred,
    display_labels=le.classes_,
    cmap='Blues',
    ax=ax
)
plt.title('Confusion Matrix - Task A')
plt.savefig(f'{OUT_DIR}/confusion_matrix_a.png')
plt.close()
print(f"  Saved: {OUT_DIR}/confusion_matrix_a.png")

# ── 5.8 - Hyperparameter Tuning ──
print("\n" + "=" * 60)
print("5.8 - Hyperparameter tuning ...")
param_grid = {'clf__C': [0.01, 0.1, 0.5, 1.0, 5.0, 10.0]}

grid = GridSearchCV(
    Pipeline([('scaler', StandardScaler()), ('clf', LogisticRegression(max_iter=2000))]),
    param_grid,
    cv=5,
    scoring='accuracy',
    n_jobs=None
)

grid.fit(X_train, y_train)

print(f"Best C value: {grid.best_params_['clf__C']}")
print(f"Best CV Accuracy: {grid.best_score_:.4f}")

# Retrain with best and predict
best_model = grid.best_estimator_
y_pred_best = best_model.predict(X_test)
print(f"Test Accuracy (Tuned): {accuracy_score(y_test, y_pred_best):.4f}")

print("\n" + "=" * 60)
print("✅ Task A Complete!")
