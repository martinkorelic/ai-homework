import json
import pandas as pd
import networkx as nx
import numpy as np
from sentence_transformers import SentenceTransformer, util
from typing import List

def load_train_csv(path):
    df = pd.read_csv(path)
    return {row["COMPANY"].strip().lower(): row["CATEGORY"].strip() for _, row in df.iterrows()}

def load_company_json(path):
    with open(path) as f:
        return json.load(f)

def load_tag_descriptions(path):
    with open(path) as f:
        return json.load(f)

def load_forbes_categories(path):
    df = pd.read_csv(path)
    return sorted(df["CATEGORY"].dropna().unique().tolist())

class MultiModelEmbedder:
    def __init__(self, model_names: List[str]):
        self.models = {name: SentenceTransformer(name) for name in model_names}

    def embed_text(self, text_list: List[str]) -> np.ndarray:
        all_embeddings = []
        for model in self.models.values():
            emb = model.encode(text_list, convert_to_numpy=True)
            emb = emb / np.linalg.norm(emb, axis=1, keepdims=True)
            all_embeddings.append(emb)
        if len(all_embeddings) == 1:
            return all_embeddings[0]
        return np.concatenate(all_embeddings, axis=1)

def build_bipartite_graph_with_similarity(company_data, train_map, tag_descriptions, forbes_categories, model_names=None):
    if model_names is None:
        model_names = ["all-MiniLM-L6-v2"]

    embedder = MultiModelEmbedder(model_names)

    # Initialize graph
    G = nx.Graph()

    # === Add nodes and frequency-based edges ===
    for entry in company_data:
        
        company_name = entry.get("name", "")
        if company_name is None:
            continue
        company_name = company_name.strip().lower()
        forbes_category = train_map.get(company_name)
        if not forbes_category:
            continue

        tags = [t for t in entry.get("tags", []) if t not in {"B2B", "B2C", "B2G"}]

        for tag in tags:

            if tag in forbes_categories:
                continue

            if not G.has_node(tag):
                G.add_node(tag, type="tag", description=tag_descriptions.get(tag, ""))
            if not G.has_node(forbes_category):
                G.add_node(forbes_category, type="forbes")

            if G.has_edge(tag, forbes_category):
                G[tag][forbes_category]["weight"] += 1
            else:
                G.add_edge(tag, forbes_category, weight=1)
    
    # Add the rest of nodes and connections for other Forbes categories
    for forbes_category in forbes_categories:
        # Check if this Forbes category is already in the graph
        if not G.has_node(forbes_category):
            # Add the missing Forbes category as a node
            G.add_node(forbes_category, type="forbes")
        
        # Connect this Forbes category to all tag nodes with weight 0
        for tag in [n for n, d in G.nodes(data=True) if d["type"] == "tag"]:
            if not G.has_edge(tag, forbes_category):
                G.add_edge(tag, forbes_category, weight=0)

    # Ensuring bipartite graph
    for forbes_category in forbes_categories:
        for tag in [n for n, d in G.nodes(data=True) if d["type"] == "tag"]:
            if not G.has_edge(tag, forbes_category):  # Check if the edge already exists
                G.add_edge(tag, forbes_category, weight=0)  # Add edge with weight 0 if it doesn't exist

    for tag in [n for n, d in G.nodes(data=True) if d["type"] == "tag"]:
        neighbors = list(G.neighbors(tag))
        total = sum(G[tag][nbr]["weight"] for nbr in neighbors)
        if total > 0:
            for nbr in neighbors:
                G[tag][nbr]["weight"] /= total

    tag_nodes = [n for n, d in G.nodes(data=True) if d["type"] == "tag"]

    tag_texts = [G.nodes[tag]["description"] for tag in tag_nodes]

    tag_embeddings = embedder.embed_text(tag_texts)
    forbes_embeddings = embedder.embed_text(forbes_categories)

    for i, tag in enumerate(tag_nodes):
        tag_embedding = tag_embeddings[i].reshape(1, -1)
        scores = util.cos_sim(tag_embedding, forbes_embeddings)[0].cpu().numpy()

        for j, fc in enumerate(forbes_categories):
            if G.has_edge(tag, fc):
                G[tag][fc]["sim_score"] = round(float(scores[j]), 4)
            else:
                # add edge only if above threshold
                G.add_edge(tag, fc, weight=0.0, sim_score=round(float(scores[j]), 4))

    return G

def save_graph(graph, path="tag_to_forbes_similarity.graphml"):
    nx.write_graphml(graph, path)
    print(f"Graph saved to {path}")

#if __name__ == "__main__":
#    csv_path = "resources/train.csv"
#    json_path = "resources/train_companies.json"
#    tag_desc_path = "resources/forbes_category_descriptions.json"
#    forbes_cat_path = "resources/categories.csv"

#    train_map = load_train_csv(csv_path)
#    company_data = load_company_json(json_path)
#    tag_descriptions = load_tag_descriptions(tag_desc_path)
#    forbes_categories = load_forbes_categories(forbes_cat_path)

#    embedding_models = [
#        "all-MiniLM-L6-v2",
#        "BAAI/bge-m3"
#    ]

#    graph = build_bipartite_graph_with_similarity(
#        company_data,
#        train_map,
#        tag_descriptions,
#        forbes_categories,
#        model_names=embedding_models
#    )

#    save_graph(graph)