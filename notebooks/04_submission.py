"""
04_submission.py - Generate Predictions and Submit (fixed version of 04_submission.ipynb)
Run with: python notebooks/04_submission.py (from project root)
"""
import pandas as pd
import numpy as np
import ast
import os
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.linear_model import LogisticRegression, Ridge
from sklearn.pipeline import Pipeline

# Resolve data paths
DATA_DIR = '../data' if os.path.basename(os.getcwd()) == 'notebooks' else 'data'
OUT_DIR  = '../outputs' if os.path.basename(os.getcwd()) == 'notebooks' else 'outputs'
os.makedirs(OUT_DIR, exist_ok=True)

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

def cosine_sim(a, b):
    """Vectorised cosine similarity between corresponding rows."""
    dot = np.sum(a * b, axis=1)
    norm_a = np.linalg.norm(a, axis=1)
    norm_b = np.linalg.norm(b, axis=1)
    return (dot / (norm_a * norm_b + 1e-10)).reshape(-1, 1)

# ══════════════════════════════════════════════════════════
# 1. Retrain Best Models (Task A & Task B)
# ══════════════════════════════════════════════════════════
print("=" * 60)
print("Step 1: Retraining models on full training data ...")

try:
    df_train = pd.read_json(f'{DATA_DIR}/chatbot-arena-conversations.jsonl.gz', lines=True, compression='gzip')

    df_train['prompt'] = df_train['conversation_a'].apply(lambda x: get_text(x, 'user'))
    df_train['response_a'] = df_train['conversation_a'].apply(lambda x: get_text(x, 'assistant'))
    df_train['response_b'] = df_train['conversation_b'].apply(lambda x: get_text(x, 'assistant'))

    df_train['prompt_len'] = df_train['prompt'].str.len()
    df_train['response_a_len'] = df_train['response_a'].str.len()
    df_train['response_b_len'] = df_train['response_b'].str.len()

    scores_df = pd.read_json(f'{DATA_DIR}/chatbot-arena-gpt3-scores.jsonl.gz', lines=True, compression='gzip')
    scores_df = scores_df.drop_duplicates(subset='question_id', keep='first')
    df_train = pd.merge(df_train, scores_df[['question_id', 'hardness']], on='question_id', how='left')

    # Load embeddings BEFORE filtering so indices stay aligned
    emb_prompt = np.load(f'{DATA_DIR}/chatbot-arena-prompts-embeddings.npy')
    emb_a = np.load(f'{DATA_DIR}/chatbot-arena-model_a_response-embeddings.npy')
    emb_b = np.load(f'{DATA_DIR}/chatbot-arena-model_b_response-embeddings.npy')

    # Filter to valid winners and align embeddings using original indices
    valid_mask = df_train['winner'].isin(['model_a', 'model_b', 'tie', 'tie (bothbad)'])
    valid_indices = df_train[valid_mask].index.values
    df_train = df_train[valid_mask].reset_index(drop=True)
    emb_prompt = emb_prompt[valid_indices]
    emb_a = emb_a[valid_indices]
    emb_b = emb_b[valid_indices]

except Exception as e:
    print("Training files not found. Using dummy data for setup...", e)
    df_train = pd.DataFrame({
        'winner': ['tie'] * 10, 'prompt_len': [1] * 10,
        'response_a_len': [1] * 10, 'response_b_len': [1] * 10, 'hardness': [5] * 10
    })
    emb_prompt = np.zeros((10, 256))
    emb_a = np.zeros((10, 256))
    emb_b = np.zeros((10, 256))

# Prepare Task A Model
lengths = df_train[['prompt_len', 'response_a_len', 'response_b_len', 'hardness']].fillna(0).values
X_train_a = np.hstack([lengths, cosine_sim(emb_prompt, emb_a), cosine_sim(emb_prompt, emb_b), emb_a - emb_b])

le = LabelEncoder()
y_train_a = le.fit_transform(df_train['winner'])

best_pipe_a = Pipeline([('scaler', StandardScaler()), ('clf', LogisticRegression(C=1.0, max_iter=2000))])
best_pipe_a.fit(X_train_a, y_train_a)
print("  Task A Model trained.")

# Prepare Task B Model
mask = df_train['hardness'].notna()
X_train_b = np.hstack([emb_prompt[mask], df_train.loc[mask, 'prompt_len'].values.reshape(-1, 1)])
y_train_b = df_train.loc[mask, 'hardness'].values

best_pipe_b = Pipeline([('scaler', StandardScaler()), ('ridge', Ridge(alpha=100))])
best_pipe_b.fit(X_train_b, y_train_b)
print("  Task B Model trained.")

# ══════════════════════════════════════════════════════════
# 2. Load the Test Set and Its Embeddings
# ══════════════════════════════════════════════════════════
print("\n" + "=" * 60)
print("Step 2: Loading test data ...")

try:
    df_test = pd.read_json(f'{DATA_DIR}/test-set/arena-test-set-prompt-and-responses.jsonl.gz', lines=True, compression='gzip')

    # The test set has pre-extracted text columns
    df_test['prompt_len'] = df_test['prompt'].astype(str).str.len()
    df_test['response_a_len'] = df_test['model_a_response'].astype(str).str.len()
    df_test['response_b_len'] = df_test['model_b_response'].astype(str).str.len()

    # Impute median hardness for test set
    median_hardness = df_train['hardness'].median()
    df_test['hardness'] = median_hardness

    test_emb_prompt = np.load(f'{DATA_DIR}/test-set/chatbot-arena-prompts-embeddings-test.npy')
    test_emb_a = np.load(f'{DATA_DIR}/test-set/chatbot-arena-model_a_response-embeddings-test.npy')
    test_emb_b = np.load(f'{DATA_DIR}/test-set/chatbot-arena-model_b_response-embeddings-test.npy')
except Exception as e:
    print("Test data not found:", e)
    df_test = pd.DataFrame({
        'question_id': ['test_1'], 'prompt_len': [1],
        'response_a_len': [1], 'response_b_len': [1], 'hardness': [5.0]
    })
    test_emb_prompt = np.zeros((1, 256))
    test_emb_a = np.zeros((1, 256))
    test_emb_b = np.zeros((1, 256))

print("  Test DataFrame shape:", df_test.shape)
print("  Test Embeddings shape matches:", len(df_test) == test_emb_prompt.shape[0])

# ══════════════════════════════════════════════════════════
# 3. Build Test Features
# ══════════════════════════════════════════════════════════
print("\n" + "=" * 60)
print("Step 3: Building test features ...")
test_lengths = df_test[['prompt_len', 'response_a_len', 'response_b_len', 'hardness']].fillna(0).values

X_test_a = np.hstack([
    test_lengths,
    cosine_sim(test_emb_prompt, test_emb_a),
    cosine_sim(test_emb_prompt, test_emb_b),
    test_emb_a - test_emb_b
])

X_test_b = np.hstack([
    test_emb_prompt,
    df_test['prompt_len'].values.reshape(-1, 1)
])

print(f"  X_test_a shape: {X_test_a.shape}")
print(f"  X_test_b shape: {X_test_b.shape}")

# ══════════════════════════════════════════════════════════
# 4. Generate Predictions
# ══════════════════════════════════════════════════════════
print("\n" + "=" * 60)
print("Step 4: Generating predictions ...")

# Task A Predictions
y_pred_a_num = best_pipe_a.predict(X_test_a)
winner_predictions = le.inverse_transform(y_pred_a_num)

print("  Task A Prediction Counts:")
print(pd.Series(winner_predictions).value_counts().to_string())

allowed_winners = {'model_a', 'model_b', 'tie', 'tie (bothbad)'}
assert set(winner_predictions).issubset(allowed_winners), "Invalid winner labels produced!"

# Task B Predictions
hardness_raw = best_pipe_b.predict(X_test_b)
hardness_predictions = np.clip(np.round(hardness_raw), 1, 10).astype(int)

print("\n  Task B Prediction Distribution:")
print(pd.Series(hardness_predictions).value_counts().sort_index().to_string())

# ══════════════════════════════════════════════════════════
# 5. Combine Into One CSV and Validate
# ══════════════════════════════════════════════════════════
print("\n" + "=" * 60)
print("Step 5: Creating submission CSV ...")

test_question_ids = df_test['question_id'].values

submission_df = pd.DataFrame({
    'question_id': test_question_ids,
    'winner': winner_predictions,
    'hardness_score': hardness_predictions
})

# Validation Checks
assert submission_df.isnull().sum().sum() == 0, "Null values found!"
assert len(submission_df) == len(df_test), "Row count mismatch!"
assert set(submission_df['winner']).issubset(allowed_winners), "Invalid winner labels!"
assert submission_df['hardness_score'].between(1, 10).all(), "Hardness scores out of bounds!"
assert submission_df['hardness_score'].dtype == int, "Hardness scores are not integers!"

# Save
submission_path = f'{OUT_DIR}/predictions.csv'
submission_df.to_csv(submission_path, index=False)

print(f"\n  ✅ Success! Predictions saved to {submission_path}")
print(f"  Total rows: {len(submission_df)}")
print(submission_df.head())
