"""
01_eda.py - Exploratory Data Analysis (fixed version of 01_eda.ipynb)
Run with: python notebooks/01_eda.py (from project root)
"""
import pandas as pd
import numpy as np
import json
import ast
import os
import matplotlib
matplotlib.use('Agg')  # Non-interactive backend
import matplotlib.pyplot as plt
import seaborn as sns

# Create outputs directory if it doesn't exist
os.makedirs('../outputs', exist_ok=True) if os.path.basename(os.getcwd()) == 'notebooks' else os.makedirs('outputs', exist_ok=True)

# Resolve data paths (works from project root or notebooks dir)
DATA_DIR = '../data' if os.path.basename(os.getcwd()) == 'notebooks' else 'data'
OUT_DIR  = '../outputs' if os.path.basename(os.getcwd()) == 'notebooks' else 'outputs'

# ── 3.1 & 3.2 - Load main data ──
print("=" * 60)
print("3.1 & 3.2 - Loading data ...")
df = pd.read_json(f'{DATA_DIR}/chatbot-arena-conversations.jsonl.gz', lines=True, compression='gzip')
print("Shape of DataFrame:", df.shape)
print("Columns:", df.columns.tolist())

# ── 3.3 - Extract the Prompt and Responses ──
print("\n" + "=" * 60)
print("3.3 - Extracting text ...")

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

print("Extracted prompt, response_a, and response_b.")
print(df[['prompt', 'response_a', 'response_b']].head(2))

# ── 3.4 - Load the GPT-3.5 Scores File ──
print("\n" + "=" * 60)
print("3.4 - Loading scores ...")
scores_df = pd.read_json(f'{DATA_DIR}/chatbot-arena-gpt3-scores.jsonl.gz', lines=True, compression='gzip')
scores_df = scores_df.drop_duplicates(subset='question_id', keep='first')

# Merge keeping all rows from df (no row expansion thanks to dedup)
df = pd.merge(df, scores_df[['question_id', 'hardness', 'topic']], on='question_id', how='left')

non_null_hardness = df['hardness'].notna().sum()
print(f"Rows with non-null hardness value: {non_null_hardness}")

# ── 3.5 - Load the Embedding Files ──
print("\n" + "=" * 60)
print("3.5 - Loading embeddings ...")
df = df.reset_index(drop=True)

prompts_emb = np.load(f'{DATA_DIR}/chatbot-arena-prompts-embeddings.npy')
model_a_emb = np.load(f'{DATA_DIR}/chatbot-arena-model_a_response-embeddings.npy')
model_b_emb = np.load(f'{DATA_DIR}/chatbot-arena-model_b_response-embeddings.npy')

print("Prompts embeddings shape:", prompts_emb.shape)
print("Model A response embeddings shape:", model_a_emb.shape)
print("Model B response embeddings shape:", model_b_emb.shape)
print("DataFrame rows match embeddings:", len(df) == prompts_emb.shape[0])

# ── 4.1 - Analyse the Winner Distribution ──
print("\n" + "=" * 60)
print("4.1 - Winner distribution ...")
winner_counts = df['winner'].value_counts()
plt.figure(figsize=(8, 5))
sns.barplot(x=winner_counts.index, y=winner_counts.values, palette='viridis')
plt.title('Distribution of Votes (Winners)')
plt.xlabel('Winner')
plt.ylabel('Count')
plt.tight_layout()
plt.savefig(f'{OUT_DIR}/vote_distribution.png')
plt.close()
print(winner_counts)
print(f"  Saved: {OUT_DIR}/vote_distribution.png")

# ── 4.2 - Calculate Win Rates for Each Model ──
print("\n" + "=" * 60)
print("4.2 - Win rates ...")
models_a = df['model_a'].dropna().unique()
models_b = df['model_b'].dropna().unique()
all_models = set(models_a).union(set(models_b))

win_rates = {}
for model in all_models:
    games_as_a = df[df['model_a'] == model]
    games_as_b = df[df['model_b'] == model]
    total_games = len(games_as_a) + len(games_as_b)
    if total_games == 0:
        continue
    wins_as_a = len(games_as_a[games_as_a['winner'] == 'model_a'])
    wins_as_b = len(games_as_b[games_as_b['winner'] == 'model_b'])
    win_rates[model] = (wins_as_a + wins_as_b) / total_games

win_rates_df = pd.DataFrame(list(win_rates.items()), columns=['model', 'win_rate']).sort_values('win_rate', ascending=False)

plt.figure(figsize=(10, 8))
sns.barplot(x='win_rate', y='model', data=win_rates_df, palette='coolwarm')
plt.title('Win Rate by Model')
plt.xlabel('Win Rate')
plt.ylabel('Model')
plt.tight_layout()
plt.savefig(f'{OUT_DIR}/win_rates.png', bbox_inches='tight')
plt.close()
print(f"  Saved: {OUT_DIR}/win_rates.png")

# ── 4.3 - Build an Elo Rating System ──
print("\n" + "=" * 60)
print("4.3 - Elo ratings ...")

def expected_score(rating_a, rating_b):
    return 1 / (1 + 10 ** ((rating_b - rating_a) / 400))

elo_ratings = {model: 1000 for model in all_models}
K = 32

for _, row in df.iterrows():
    ma, mb, winner = row['model_a'], row['model_b'], row['winner']
    if pd.isna(ma) or pd.isna(mb):
        continue
    ra, rb = elo_ratings[ma], elo_ratings[mb]
    ea, eb = expected_score(ra, rb), expected_score(rb, ra)
    if winner == 'model_a':
        sa, sb = 1, 0
    elif winner == 'model_b':
        sa, sb = 0, 1
    elif winner in ['tie', 'tie (bothbad)']:
        sa, sb = 0.5, 0.5
    else:
        continue
    elo_ratings[ma] = ra + K * (sa - ea)
    elo_ratings[mb] = rb + K * (sb - eb)

elo_df = pd.DataFrame(list(elo_ratings.items()), columns=['model', 'elo']).sort_values('elo', ascending=False)

plt.figure(figsize=(10, 8))
sns.barplot(x='elo', y='model', data=elo_df, palette='magma')
plt.title('Elo Ratings by Model')
plt.xlabel('Elo Rating')
plt.ylabel('Model')
plt.tight_layout()
plt.savefig(f'{OUT_DIR}/elo_ratings.png', bbox_inches='tight')
plt.close()
print(f"  Saved: {OUT_DIR}/elo_ratings.png")

# ── 4.4 - Explore Text Lengths ──
print("\n" + "=" * 60)
print("4.4 - Text lengths ...")
df['prompt_len'] = df['prompt'].str.len()
df['response_a_len'] = df['response_a'].str.len()
df['response_b_len'] = df['response_b'].str.len()

fig, axs = plt.subplots(1, 3, figsize=(15, 4))
sns.histplot(df['prompt_len'].clip(upper=5000), bins=30, ax=axs[0], color='steelblue')
axs[0].set_title('Prompt Length')
sns.histplot(df['response_a_len'].clip(upper=5000), bins=30, ax=axs[1], color='coral')
axs[1].set_title('Response A Length')
sns.histplot(df['response_b_len'].clip(upper=5000), bins=30, ax=axs[2], color='seagreen')
axs[2].set_title('Response B Length')
plt.tight_layout()
plt.savefig(f'{OUT_DIR}/length_distributions.png')
plt.close()

print("Average Lengths grouped by Winner:")
print(df.groupby('winner')[['response_a_len', 'response_b_len']].mean())
print(f"  Saved: {OUT_DIR}/length_distributions.png")

# ── 4.5 - Explore Hardness Scores ──
print("\n" + "=" * 60)
print("4.5 - Hardness scores ...")
plt.figure(figsize=(8, 5))
sns.countplot(x='hardness', data=df, palette='rocket')
plt.title('Distribution of Hardness Scores')
plt.xlabel('Hardness (1-10)')
plt.ylabel('Count')
plt.tight_layout()
plt.savefig(f'{OUT_DIR}/hardness_distribution.png')
plt.close()

print("Average Hardness by Winner:")
print(df.groupby('winner')['hardness'].mean().sort_values(ascending=False))

print("\nAverage Hardness by Topic:")
print(df.groupby('topic')['hardness'].mean().sort_values(ascending=False))
print(f"  Saved: {OUT_DIR}/hardness_distribution.png")

# ── 4.6 - Explore Topics ──
print("\n" + "=" * 60)
print("4.6 - Topics ...")
topic_counts = df['topic'].value_counts()
plt.figure(figsize=(10, 5))
sns.barplot(x=topic_counts.values, y=topic_counts.index, palette='Set2')
plt.title('Most Common Question Topics')
plt.xlabel('Count')
plt.ylabel('Topic')
plt.tight_layout()
plt.savefig(f'{OUT_DIR}/topic_distribution.png')
plt.close()
print(f"  Saved: {OUT_DIR}/topic_distribution.png")

# ── 4.7 - Verify Embeddings with Cosine Similarity ──
print("\n" + "=" * 60)
print("4.7 - Cosine similarity verification ...")
from sklearn.metrics.pairwise import cosine_similarity

# Compare prompt[0] with its own response vs a random response
same_pair = cosine_similarity(prompts_emb[0:1], model_a_emb[0:1])[0][0]
diff_pair = cosine_similarity(prompts_emb[0:1], model_a_emb[100:101])[0][0]

print(f"Cosine similarity (same pair): {same_pair:.4f}")
print(f"Cosine similarity (different pair): {diff_pair:.4f}")
print(f"Same-pair similarity is higher: {same_pair > diff_pair}")

print("\n" + "=" * 60)
print("✅ EDA Complete! All plots saved to outputs/")
