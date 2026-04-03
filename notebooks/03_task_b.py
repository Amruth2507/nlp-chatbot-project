"""
03_task_b.py - Task B: Hardness Prediction (fixed version of 03_task_b.ipynb)
Run with: python notebooks/03_task_b.py (from project root)
"""
import pandas as pd
import numpy as np
import ast
import os
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import Ridge
from sklearn.pipeline import Pipeline
from sklearn.metrics import mean_squared_error
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
df['prompt_len'] = df['prompt'].str.len()

scores_df = pd.read_json(f'{DATA_DIR}/chatbot-arena-gpt3-scores.jsonl.gz', lines=True, compression='gzip')
scores_df = scores_df.drop_duplicates(subset='question_id', keep='first')
df = pd.merge(df, scores_df[['question_id', 'hardness']], on='question_id', how='left')

emb_prompt = np.load(f'{DATA_DIR}/chatbot-arena-prompts-embeddings.npy')

# Ensure df and embeddings stay aligned (merge shouldn't expand rows now)
assert len(df) == emb_prompt.shape[0], f"Shape mismatch: df={len(df)}, emb={emb_prompt.shape[0]}"

print(f"Data loaded: {len(df)} rows, embeddings shape: {emb_prompt.shape}")

# ── 5.9 - Prepare Data for Task B ──
print("\n" + "=" * 60)
print("5.9 - Preparing data for Task B ...")

# 1) Boolean mask
mask = df['hardness'].notna()

# 2) Apply mask
df_filtered = df[mask].copy()
emb_prompt_filtered = emb_prompt[mask]

# 3) Feature matrix X_hard (embeddings + prompt length)
prompt_len_filtered = df_filtered['prompt_len'].values.reshape(-1, 1)
X_hard = np.hstack([emb_prompt_filtered, prompt_len_filtered])

# 4) Target array y_hard
y_hard = df_filtered['hardness'].values

print("X_hard shape:", X_hard.shape)
print("y_hard shape:", y_hard.shape)

# ── 5.10 - Build Ridge Regression Model ──
print("\n" + "=" * 60)
print("5.10 - Building Ridge Regression model ...")

# 1) Split
X_train, X_test, y_train, y_test = train_test_split(
    X_hard, y_hard, test_size=0.2, random_state=42
)

# 2) Pipeline
pipeline = Pipeline([
    ('scaler', StandardScaler()),
    ('ridge', Ridge())
])

# 3) Fit
pipeline.fit(X_train, y_train)

# 4) Predict
y_pred_raw = pipeline.predict(X_test)

# 5) Round and Clip
y_pred_clipped = np.clip(np.round(y_pred_raw), 1, 10)

# 6) MSE
mse = mean_squared_error(y_test, y_pred_clipped)
print(f"Initial Mean Squared Error: {mse:.4f}")

# ── 5.11 - Tune Alpha Hyperparameter ──
print("\n" + "=" * 60)
print("5.11 - Tuning alpha hyperparameter ...")
param_grid = {'ridge__alpha': [0.001, 0.01, 0.1, 1, 10, 100, 1000]}

grid = GridSearchCV(
    Pipeline([('scaler', StandardScaler()), ('ridge', Ridge())]),
    param_grid,
    cv=5,
    scoring='neg_mean_squared_error',
    n_jobs=None
)

grid.fit(X_train, y_train)

best_alpha = grid.best_params_['ridge__alpha']
best_neg_mse = grid.best_score_
print(f"Best alpha: {best_alpha}")
print(f"Best Negative MSE (CV): {best_neg_mse:.4f}")

# Retrain, Round, Clip, Evaluate
best_model = grid.best_estimator_
y_pred_raw_best = best_model.predict(X_test)
y_pred_clipped_best = np.clip(np.round(y_pred_raw_best), 1, 10)

final_mse = mean_squared_error(y_test, y_pred_clipped_best)
print(f"Final Mean Squared Error (Tuned): {final_mse:.4f}")

# ── 5.12 - Visualise Predictions ──
print("\n" + "=" * 60)
print("5.12 - Visualising predictions ...")
fig, axs = plt.subplots(1, 2, figsize=(14, 5))

# Scatter plot: Actual vs Predicted
sns.scatterplot(x=y_test, y=y_pred_clipped_best, alpha=0.1, ax=axs[0])
axs[0].plot([1, 10], [1, 10], 'r--', lw=2)
axs[0].set_title('Actual vs Predicted Hardness')
axs[0].set_xlabel('Actual Hardness')
axs[0].set_ylabel('Predicted Hardness (Clipped & Rounded)')

# Histogram of Residuals
residuals = y_pred_clipped_best - y_test
sns.histplot(residuals, bins=19, discrete=True, ax=axs[1])
axs[1].set_title('Histogram of Residuals')
axs[1].set_xlabel('Prediction Error (Predicted - Actual)')
axs[1].set_ylabel('Count')

plt.tight_layout()
plt.savefig(f'{OUT_DIR}/hardness_predictions.png')
plt.close()
print(f"  Saved: {OUT_DIR}/hardness_predictions.png")

print("\n" + "=" * 60)
print("✅ Task B Complete!")
