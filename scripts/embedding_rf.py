import pandas as pd
import numpy as np
from sentence_transformers import SentenceTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report
from typing import List

class MultiModelEmbedder:
    def __init__(self, model_names: List[str]):
        self.models = {name: SentenceTransformer(name) for name in model_names}

    def embed_text(self, text_list: List[str]) -> np.ndarray:
        all_embeddings = []
        for model_name, model in self.models.items():
            embeddings = model.encode(text_list, convert_to_numpy=True)
            embeddings_norm = embeddings / np.linalg.norm(embeddings, axis=1, keepdims=True)
            all_embeddings.append(embeddings_norm)
        if len(all_embeddings) == 1:
            return all_embeddings[0]
        return np.concatenate(all_embeddings, axis=1)

def train_and_eval_embedding_rf(train_path="resources/train.csv", test_path="resources/test.csv"):

    train_df = pd.read_csv(train_path)
    test_df = pd.read_csv(test_path)

    train_df.dropna(subset=["COMPANY", "CATEGORY"], inplace=True)
    test_df.dropna(subset=["COMPANY", "CATEGORY"], inplace=True)

    train_df["COMPANY"] = train_df["COMPANY"].str.strip()
    train_df["CATEGORY"] = train_df["CATEGORY"].str.strip().str.lower()
    test_df["COMPANY"] = test_df["COMPANY"].str.strip()
    test_df["CATEGORY"] = test_df["CATEGORY"].str.strip().str.lower()

    model_names = ["all-MiniLM-L6-v2", "BAAI/bge-m3"]
    embedder = MultiModelEmbedder(model_names)

    X_train = embedder.embed_text(train_df["COMPANY"].tolist())
    y_train = train_df["CATEGORY"].tolist()

    X_test = embedder.embed_text(test_df["COMPANY"].tolist())
    y_test = test_df["CATEGORY"].tolist()

    clf = RandomForestClassifier(n_estimators=400, random_state=42)
    clf.fit(X_train, y_train)

    y_pred = clf.predict(X_test)

    cls_report = classification_report(y_test, y_pred)

    print(cls_report)

    return (y_test, y_pred)