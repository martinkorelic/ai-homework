import os
import csv
import json
import numpy as np
import chromadb
from chromadb.config import Settings
from sentence_transformers import SentenceTransformer

def load_companies_from_csv(csv_path: str):
    companies = []
    with open(csv_path, newline='', encoding='utf-8') as f:
        reader = csv.DictReader(f)
        for row in reader:
            name = row.get("COMPANY")
            category = row.get("CATEGORY")
            if name and category:
                companies.append((name.strip(), category.strip()))
    return companies

def load_categories_from_json(json_path: str):
    with open(json_path, 'r', encoding='utf-8') as f:
        categories_dict = json.load(f)
    return categories_dict

def embed_text_with_models(text_list, model_names):
    """
    Embed a list of texts with multiple models and concatenate the normalized embeddings.
    
    Args:
        text_list (list): List of texts to embed
        model_names (list): List of model names to use for embedding
    
    Returns:
        numpy.ndarray: Concatenated normalized embeddings
    """
    if not model_names:
        raise ValueError("At least one embedding model must be provided")
    
    all_embeddings = []
    model_info = []
    
    for model_name in model_names:
        model = SentenceTransformer(model_name)
        embeddings = model.encode(text_list, convert_to_numpy=True)
        
        # Normalize embeddings
        embeddings_norm = embeddings / np.linalg.norm(embeddings, axis=1, keepdims=True)
        all_embeddings.append(embeddings_norm)
        
        model_info.append({
            "name": model_name,
            "dim": embeddings.shape[1]
        })
    
    # If only one model, return its embeddings directly
    if len(all_embeddings) == 1:
        return all_embeddings[0], model_info
    
    # Otherwise concatenate the embeddings
    combined_embeddings = np.concatenate(all_embeddings, axis=1)
    return combined_embeddings, model_info

def embed_and_store_to_chroma(
    csv_path: str, 
    persist_dir: str = "./chroma_db",
    model_names: list = ["all-MiniLM-L6-v2"],
    collection_name: str = "businesses"
):
    # Load companies
    companies = load_companies_from_csv(csv_path)
    
    names = [c[0] for c in companies]
    categories = [c[1] for c in companies]
    
    # Create embeddings from all models
    combined_embeddings, model_info = embed_text_with_models(names, model_names)
    
    # Setup Chroma persistent client
    client = chromadb.PersistentClient(path=persist_dir)
    collection = client.get_or_create_collection(name=collection_name)
    
    documents = names
    metadatas = [{"category": cat} for cat in categories]
    ids = [f"biz-{i}" for i in range(len(names))]
    
    collection.add(
        documents=documents,
        embeddings=combined_embeddings.tolist(),
        metadatas=metadatas,
        ids=ids
    )
    
    print(f"✅ Stored {len(names)} businesses into ChromaDB at '{persist_dir}'.")
    for model in model_info:
        print(f"   Model: {model['name']} - Vector size: {model['dim']}")
    if len(model_info) > 1:
        print(f"   Combined vector size: {combined_embeddings.shape[1]}")

def embed_and_store_categories(
    json_path: str,
    persist_dir: str = "./chroma_db_categories",
    model_names: list = ["all-MiniLM-L6-v2"],
    collection_name: str = "category_descriptions"
):
    # Load categories
    categories_dict = load_categories_from_json(json_path)
    
    category_names = list(categories_dict.keys())
    descriptions = list(categories_dict.values())
    
    # Create embeddings from all models for descriptions
    combined_embeddings, model_info = embed_text_with_models(descriptions, model_names)
    
    # Setup Chroma persistent client for categories
    client = chromadb.PersistentClient(path=persist_dir)
    collection = client.get_or_create_collection(name=collection_name)
    
    documents = descriptions
    metadatas = [{"category": cat} for cat in category_names]
    ids = [f"cat-{i}" for i in range(len(category_names))]
    
    collection.add(
        documents=documents,
        embeddings=combined_embeddings.tolist(),
        metadatas=metadatas,
        ids=ids
    )
    
    print(f"✅ Stored {len(category_names)} category descriptions into ChromaDB at '{persist_dir}'.")
    for model in model_info:
        print(f"   Model: {model['name']} - Vector size: {model['dim']}")
    if len(model_info) > 1:
        print(f"   Combined vector size: {combined_embeddings.shape[1]}")

def embed_and_store_categories_graph(
    json_path: str,
    graph_path: str,
    persist_dir: str = "./chroma_db_categories",
    model_names: list = ["all-MiniLM-L6-v2"],
    collection_name: str = "category_descriptions",
    yelp = True
):
    # Load categories from JSON
    categories_dict = load_categories_from_json(json_path)
    
    # Load the refined graph
    import networkx as nx
    G = nx.read_graphml(graph_path)

    subtype = 'yelp' if yelp else 'tag'
    
    # Extract filtered category names from the graph
    # Get all nodes that remain in the graph
    filtered_yelp_nodes = [node for node, attrs in G.nodes(data=True) 
                          if attrs.get('type') == subtype]
    
    forbes_nodes = [node for node, attrs in G.nodes(data=True) 
                   if attrs.get('type') == 'forbes']
    
    print(f"Found {len(filtered_yelp_nodes)} {subtype} categories and {len(forbes_nodes)} Forbes categories in the filtered graph.")
    
    # Create lists for filtered categories
    filtered_category_names = filtered_yelp_nodes + forbes_nodes
    filtered_descriptions = [categories_dict.get(cat, f"Missing description for {cat}") for cat in filtered_category_names]
    
    # Create embeddings from all models for descriptions
    combined_embeddings, model_info = embed_text_with_models(filtered_descriptions, model_names)
    
    # Setup Chroma persistent client for categories
    client = chromadb.PersistentClient(path=persist_dir)
    collection = client.get_or_create_collection(name=collection_name)
    
    metadatas = [{"category": cat, "type": "yelp" if cat in filtered_yelp_nodes else "forbes"} 
                for cat in filtered_category_names]
    ids = [f"cat-{i}" for i in range(len(filtered_category_names))]
    
    collection.add(
        documents=filtered_descriptions,
        embeddings=combined_embeddings.tolist(),
        metadatas=metadatas,
        ids=ids
    )
    
    print(f"✅ Stored {len(filtered_category_names)} category descriptions into ChromaDB at '{persist_dir}'.")
    for model in model_info:
        print(f"   Model: {model['name']} - Vector size: {model['dim']}")
    if len(model_info) > 1:
        print(f"   Combined vector size: {combined_embeddings.shape[1]}")

def query_similarity(
    query_text: str,
    persist_dir: str,
    collection_name: str,
    model_names: list = ["all-MiniLM-L6-v2"],
    top_k: int = 5
):
    # Embed query with the same models
    query_embedding, _ = embed_text_with_models([query_text], model_names)
    
    # Connect to persistent Chroma
    client = chromadb.PersistentClient(path=persist_dir)
    collection = client.get_collection(name=collection_name)
    
    # Query Chroma
    results = collection.query(
        query_embeddings=query_embedding.tolist(),
        n_results=top_k,
        include=["distances", "metadatas", "documents"]
    )
    
    print(f"🔍 Top {top_k} similar items for '{query_text}':\n")
    for i in range(min(top_k, len(results["documents"][0]))):
        document = results["documents"][0][i]
        category = results["metadatas"][0][i]["category"]
        distance = results["distances"][0][i]
        print(f"{i+1}. {document}")
        print(f"   Category: {category}")
        print(f"   Distance: {distance:.4f}")
        print()
    
    return results

#if __name__ == "__main__":
    # Example embedding businesses with multiple models
    #embed_and_store_to_chroma(
    #    "resources/train.csv",
    #    persist_dir="./chroma_db_biz",
    #    model_names=["all-MiniLM-L6-v2", "BAAI/bge-m3"],
    #    collection_name="businesses_multi"
    #)
    
    # Example embedding categories with multiple models
    #embed_and_store_categories_graph(
    #    "resources/yelp_category_descriptions.json",
    #    "resources/yelp_categories.graphml",
    #    persist_dir="./chroma_db_yelp_cat",
    #    model_names=["all-MiniLM-L6-v2", "BAAI/bge-m3"],
    #    collection_name="categories_multi"
    #)

    #embed_and_store_categories_graph(
    #    "resources/forbes_category_descriptions.json",
    #    "resources/forbes_categories.graphml",
    #    persist_dir="./chroma_db_forbes_cat",
    #    model_names=["all-MiniLM-L6-v2", "BAAI/bge-m3"],
    #    collection_name="categories_multi",
    #    yelp=False
    #)
    
    # Example querying businesses
    #query_similarity(
    #    "H & M Hennes & Mauritz",
    #    persist_dir="./chroma_db_cat",
    #    collection_name="categories_multi",
    #    model_names=["all-MiniLM-L6-v2", "BAAI/bge-m3"],
    #    top_k=5
    #)

    #query_similarity(
    #    "Deloitte",
    #    persist_dir="./chroma_db_forbes_cat",
    #    collection_name="categories_multi",
    #    model_names=["all-MiniLM-L6-v2", "BAAI/bge-m3"],
    #    top_k=5
    #)