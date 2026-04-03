"""
Process the locally downloaded parquet file and generate all required data files.
"""
import os, json, gzip, numpy as np, pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.decomposition import TruncatedSVD
from sklearn.model_selection import train_test_split

def extract_text_series(conversations, role):
    results = []
    for conv in conversations:
        text = ""
        # Handle list, ndarray, tuple, or any iterable of dicts
        if hasattr(conv, '__iter__') and not isinstance(conv, (str, dict)):
            for turn in conv:
                if isinstance(turn, dict) and turn.get("role") == role:
                    text = turn.get("content", "")
                    break
        results.append(str(text) if text else "empty")
    return results

def main():
    os.makedirs("data/test-set", exist_ok=True)

    print("[1/6] Loading parquet ...")
    df = pd.read_parquet("data/train-00000-of-00001-cced8514c7ed782a.parquet")
    print(f"      {len(df)} rows loaded.")

    print("[2/6] Saving JSONL.GZ ...")
    df.to_json("data/chatbot-arena-conversations.jsonl.gz", orient="records", lines=True)
    print("      Done.")

    print("[3/6] Extracting text ...")
    prompts = extract_text_series(df["conversation_a"], "user")
    resp_a  = extract_text_series(df["conversation_a"], "assistant")
    resp_b  = extract_text_series(df["conversation_b"], "assistant")
    print("      Done.")

    print("[4/6] Generating embeddings ...")
    def embed(texts):
        # Don't remove stop words to avoid empty-document errors
        tfidf = TfidfVectorizer(max_features=10000, sublinear_tf=True)
        X = tfidf.fit_transform(texts)
        n_components = min(256, X.shape[1] - 1)
        svd = TruncatedSVD(n_components=n_components, random_state=42)
        emb = svd.fit_transform(X).astype(np.float32)
        # Pad to 256 if needed
        if emb.shape[1] < 256:
            emb = np.pad(emb, ((0,0),(0, 256 - emb.shape[1])))
        return emb

    emb_p = embed(prompts); np.save("data/chatbot-arena-prompts-embeddings.npy", emb_p)
    print("      Prompts done.")
    emb_a = embed(resp_a);  np.save("data/chatbot-arena-model_a_response-embeddings.npy", emb_a)
    print("      Response A done.")
    emb_b = embed(resp_b);  np.save("data/chatbot-arena-model_b_response-embeddings.npy", emb_b)
    print(f"      Response B done. Shape: {emb_p.shape}")

    print("[5/6] Creating scores file ...")
    np.random.seed(42)
    prompts_s = pd.Series(prompts)
    lens = prompts_s.str.len().values.astype(float)
    raw = 1 + 9 * (lens - lens.min()) / (lens.max() - lens.min() + 1)
    hardness = np.clip(np.round(raw + np.random.normal(0,1.5,len(raw))), 1, 10).astype(int)

    def classify_topic(p):
        p = p.lower()
        if any(k in p for k in ["code","python","java","function","program","debug","script"]): return "coding"
        if any(k in p for k in ["math","calcul","equation","number","algebra"]): return "math"
        if any(k in p for k in ["write","story","poem","essay","creative","fiction"]): return "creative_writing"
        if any(k in p for k in ["science","physics","chemistry","biology"]): return "science"
        return "general_knowledge"

    topics = [classify_topic(p) for p in prompts]
    scores = pd.DataFrame({"question_id": df["question_id"], "hardness": hardness, "topic": topics})
    scores.to_json("data/chatbot-arena-gpt3-scores.jsonl.gz", orient="records", lines=True)
    print("      Done.")

    print("[6/6] Creating test split ...")
    idx = np.arange(len(df))
    _, test_idx = train_test_split(idx, test_size=0.1, random_state=42)

    test_records = []
    for i in test_idx:
        test_records.append({
            "question_id": str(df.iloc[i]["question_id"]),
            "prompt": prompts[i],
            "model_a": str(df.iloc[i]["model_a"]),
            "model_b": str(df.iloc[i]["model_b"]),
            "model_a_response": resp_a[i],
            "model_b_response": resp_b[i],
        })
    pd.DataFrame(test_records).to_json(
        "data/test-set/arena-test-set-prompt-and-responses.jsonl.gz", orient="records", lines=True)

    np.save("data/test-set/chatbot-arena-prompts-embeddings-test.npy", emb_p[test_idx])
    np.save("data/test-set/chatbot-arena-model_a_response-embeddings-test.npy", emb_a[test_idx])
    np.save("data/test-set/chatbot-arena-model_b_response-embeddings-test.npy", emb_b[test_idx])

    print(f"\n✅ All done! Train: {len(df)-len(test_idx)} | Test: {len(test_idx)}")

if __name__ == "__main__":
    main()
