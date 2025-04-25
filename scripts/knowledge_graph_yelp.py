import json
import numpy as np
from sentence_transformers import SentenceTransformer, util
import networkx as nx
from scripts.utils import load_json

def load_yelp_category_descriptions(path="resources/yelp_category_descriptions.json"):
    with open(path) as f:
        return json.load(f)

def load_forbes_categories(path=None):
    if path:
        with open(path) as f:
            return [line.strip() for line in f if line.strip() and not line.lower().startswith("category")]
    return [
        "Technology hardware & equipment", "Food drink & tobacco", "Construction", 
        "Health care equipment & services", "Conglomerates", "Capital goods", "Retailing", 
        "Utilities", "Food markets", "Aerospace & defense", "Materials", "Banking", 
        "Oil & gas operations", "Business services & supplies", "Insurance", "Semiconductors", 
        "Trading companies", "Diversified financials", "Drugs & biotechnology", 
        "Telecommunications services", "Software & services", "Household & personal products", 
        "Hotels restaurants & leisure", "Transportation", "Consumer durables", "Media", "Chemicals"
    ]

class MultiModelEmbedder:
    def __init__(self, model_names):
        self.models = {name: SentenceTransformer(name) for name in model_names}

    def embed_text(self, text_list):
        all_embeddings = []
        for model in self.models.values():
            emb = model.encode(text_list, convert_to_numpy=True)
            emb = emb / np.linalg.norm(emb, axis=1, keepdims=True)  # Normalize embeddings
            all_embeddings.append(emb)
        if len(all_embeddings) == 1:
            return all_embeddings[0]
        return np.concatenate(all_embeddings, axis=1)  # Concatenate embeddings from all models

def build_similarity_graph(yelp_descs, forbes_cats, model_names=None):
    if model_names is None:
        model_names = ["all-MiniLM-L6-v2"]

    # Initialize the MultiModelEmbedder with the selected models
    embedder = MultiModelEmbedder(model_names)

    yelp_labels = list(yelp_descs.keys())
    yelp_texts = [yelp_descs[label] for label in yelp_labels]

    # Get embeddings for Yelp descriptions and Forbes categories
    yelp_embeddings = embedder.embed_text(yelp_texts)
    forbes_embeddings = embedder.embed_text(forbes_cats)

    G = nx.Graph()

    # Add nodes to the graph for Yelp labels and Forbes categories
    for yc in yelp_labels:
        G.add_node(yc, type="yelp", description=yelp_descs[yc])
    for fc in forbes_cats:
        G.add_node(fc, type="forbes")

    # Compute and add edge similarities between each Yelp label and Forbes category
    for i, fc in enumerate(forbes_cats):
        for j, yc in enumerate(yelp_labels):
            edge_attrs = {}
            sim_score = util.cos_sim(forbes_embeddings[i], yelp_embeddings[j]).item()

            # Round similarity score for clarity
            edge_attrs["score"] = round(sim_score, 4)
            edge_attrs["weight"] = round(sim_score, 4)  # Use the similarity score as the edge weight

            # Add the edge to the graph with the computed similarity score
            G.add_edge(fc, yc, **edge_attrs)

    return G

def save_graph_graphml(G, path="category_similarity_graph.graphml"):
    nx.write_graphml(G, path)
    print(f"GraphML saved to {path} with {len(G.nodes)} nodes and {len(G.edges)} edges.")

def save_graph_json(G, path="category_similarity_graph.json"):
    data = {
        "nodes": [{"id": n, **G.nodes[n]} for n in G.nodes],
        "edges": [{"source": u, "target": v, **G[u][v]} for u, v in G.edges]
    }
    with open(path, "w") as f:
        json.dump(data, f, indent=2)

#if __name__ == "__main__":
#    yelp_category_descs = load_yelp_category_descriptions("resources/yelp_category_descriptions.json")
#    forbes_categories = load_forbes_categories("resources/categories.csv")
#
#    embedding_models = [
#        "all-MiniLM-L6-v2",
#        "BAAI/bge-m3"
#    ]
#
#    graph = build_similarity_graph(
#        yelp_category_descs,
#        forbes_categories,
#        model_names=embedding_models
#    )
#
#   save_graph_json(graph)
#    save_graph_graphml(graph)
#
#    print("Graph constructed with", len(graph.nodes), "nodes and", len(graph.edges), "edges.")