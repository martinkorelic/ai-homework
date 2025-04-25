import json
import os
from dotenv import load_dotenv
import csv
import random
from collections import defaultdict
from typing import Tuple
import networkx as nx
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import os
from sklearn.metrics import classification_report, confusion_matrix

def load_config(env_path=".env") -> dict:
    """Load configuration from a .env file."""
    load_dotenv(dotenv_path=env_path)

    return {
        "embedding_model": os.getenv("EMBEDDING_MODEL")
    }


def load_json(file_path):
    """Load JSON data from a file."""
    with open(file_path, 'r', encoding='utf-8') as f:
        return json.load(f)

def save_json(file_path, data):
    """Save JSON data to a file."""
    with open(file_path, 'w', encoding='utf-8') as f:
        json.dump(data, f, indent=2, ensure_ascii=False)

def extract_name_and_categories(data):
    """
    Extract only the 'name' and split 'categories' into a list for each entry.
    
    Args:
        data (list): List of business data dicts.

    Returns:
        list: Processed list of dicts with 'name' and 'categories' (as a list).
    """
    extracted = []
    for entry in data:
        name = entry.get("name", "").strip()
        raw_categories = entry.get("categories", "")

        if raw_categories is None:
            continue

        categories = [cat.strip() for cat in raw_categories.split(",") if cat.strip()]
        extracted.append({
            "name": name,
            "categories": categories
        })
    return extracted

def clean_and_simplify_data(input_file: str, output_file: str) -> None:
    """
    Clean and simplify the dataset by:
    1. Converting similarity scores dict to ordered list of values
    2. Removing redundant information
    
    Args:
        input_file: Path to original JSON file
        output_file: Path to save simplified JSON file
    """
    # Load original data
    with open(input_file, 'r') as f:
        data = json.load(f)
    
    # Get the category order (assuming it's consistent across all entries)
    if data and 'similarity_scores' in data[0]:
        categories_order = list(data[0]['similarity_scores'].keys())
    else:
        print("Warning: No similarity scores found in data")
        categories_order = []
    
    # Simplify each entry
    simplified_data = []
    for entry in data:
        # Extract just the values in the correct order
        if 'similarity_scores' in entry and categories_order:
            similarity_vector = [entry['similarity_scores'][cat] for cat in categories_order]
        else:
            similarity_vector = []
        
        # Create simplified entry
        simplified_entry = {
            "name": entry.get("name", ""),
            "categories": entry.get("categories", []),
            "forbes_category": entry.get("most_likely_label", ""),
            "similarity_vector": similarity_vector
        }
        simplified_data.append(simplified_entry)
    
    # Save simplified data
    with open(output_file, 'w') as f:
        json.dump(simplified_data, f, indent=2, ensure_ascii=False)

def extract_unique_yelp_categories(businesses):
    unique_categories = set()
    for biz in businesses:
        for cat in biz.get("categories", []):
            unique_categories.add(cat.strip())
    return unique_categories

def split_dataset_stratified(
    input_csv: str,
    train_csv: str,
    test_csv: str,
    train_ratio: float = 0.7
) -> Tuple[int, int]:
    """
    Splits a dataset CSV into stratified train/test CSVs.

    Args:
        input_csv: Path to input CSV with COMPANY,CATEGORY.
        train_csv: Path to output training CSV.
        test_csv: Path to output test CSV.
        train_ratio: Ratio of data to use for training per category.

    Returns:
        Tuple of (number of training rows, number of test rows)
    """
    # Load input CSV
    with open(input_csv, newline='', encoding='utf-8') as f:
        reader = csv.DictReader(f)
        data = [(row["COMPANY"], row["CATEGORY"]) for row in reader if row["CATEGORY"]]

    # Group by category
    grouped = defaultdict(list)
    for company, category in data:
        grouped[category].append((company, category))

    train_set = []
    test_set = []

    for category, entries in grouped.items():
        random.shuffle(entries)
        cutoff = int(train_ratio * len(entries))
        train_set.extend(entries[:cutoff])
        test_set.extend(entries[cutoff:])

    # Save to output CSVs
    def save_csv(filename, rows):
        with open(filename, "w", newline="", encoding="utf-8") as f:
            writer = csv.writer(f)
            writer.writerow(["COMPANY", "CATEGORY"])
            writer.writerows(rows)

    save_csv(train_csv, train_set)
    save_csv(test_csv, test_set)

    return len(train_set), len(test_set)

def enhance_and_save_graph(input_graph_path: str, output_graph_path: str) -> None:
    """
    Load the GraphML file, enhance it with additional node and edge attributes, and save it.
    
    Args:
        input_graph_path: Path to the input GraphML file
        output_graph_path: Path to save the enhanced GraphML file
    """
    # Load the graph
    print(f"Loading graph from {input_graph_path}")
    graph = nx.read_graphml(input_graph_path)
    
    # Identify Forbes vs Yelp nodes
    forbes_categories = set([
        "Technology hardware & equipment", "Food drink & tobacco", "Construction",
        "Health care equipment & services", "Conglomerates", "Capital goods",
        "Retailing", "Utilities", "Food markets", "Aerospace & defense",
        "Materials", "Banking", "Oil & gas operations", "Business services & supplies",
        "Insurance", "Semiconductors", "Trading companies", "Diversified financials",
        "Drugs & biotechnology", "Telecommunications services", "Software & services",
        "Household & personal products", "Hotels restaurants & leisure", "Transportation",
        "Consumer durables", "Media", "Chemicals"
    ])
    
    # Add node types and other attributes
    for node, attrs in graph.nodes(data=True):
        if node in forbes_categories:
            graph.nodes[node]['type'] = 'forbes'
            graph.nodes[node]['label'] = f"Forbes: {node}"
        else:
            graph.nodes[node]['type'] = 'yelp'
            # Extract description if available
            description = attrs.get('d1', '')
            graph.nodes[node]['label'] = f"Yelp: {node}"
            if description:
                graph.nodes[node]['description'] = description
    
    # Add edge types - simplify to just mark all edges as yelp2forbes regardless of direction
    for u, v in graph.edges():
        graph.edges[u, v]['edge_type'] = 'yelp2forbes'
    
    # Save the enhanced graph
    print(f"Saving enhanced graph to {output_graph_path}")
    nx.write_graphml(graph, output_graph_path)
    print(f"Enhanced graph saved successfully with {len(graph.nodes())} nodes and {len(graph.edges())} edges")

def refine_graph(input_file, output_file, top_n=50):
    # Load the graph
    G = nx.read_graphml(input_file)
    
    # Identify Forbes nodes
    forbes_nodes = [node for node, attrs in G.nodes(data=True) if attrs.get('type') == 'forbes']
    
    # Keep track of Yelp nodes to keep
    yelp_nodes_to_keep = set()
    
    # For each Forbes node
    for forbes_node in forbes_nodes:
        # Get all edges connected to this Forbes node
        connected_edges = []
        for edge in G.edges(forbes_node, data=True):
            # We're only interested in edges from Forbes to Yelp nodes
            if edge[2].get('edge_type') == 'yelp2forbes':
                yelp_node = edge[1]
                # Use the overall weight attribute
                weight = float(edge[2].get('weight', 0))
                connected_edges.append((yelp_node, weight))
        
        # Sort by weight (descending) and keep top N
        connected_edges.sort(key=lambda x: x[1], reverse=True)
        top_yelp_nodes = [edge[0] for edge in connected_edges[:top_n]]
        
        # Add these to our set of nodes to keep
        yelp_nodes_to_keep.update(top_yelp_nodes)
    
    # Identify Yelp nodes to remove
    yelp_nodes_to_remove = {node for node, attrs in G.nodes(data=True) 
                          if attrs.get('type') == 'yelp' and node not in yelp_nodes_to_keep}
    
    # Remove Yelp nodes not in our keep set
    G.remove_nodes_from(yelp_nodes_to_remove)
    
    # Save the refined graph
    nx.write_graphml(G, output_file)
    
    print(f"Graph refined: kept {len(yelp_nodes_to_keep)} Yelp nodes connected to {len(forbes_nodes)} Forbes nodes")
    print(f"Removed {len(yelp_nodes_to_remove)} Yelp nodes")
    
    return G

def get_unique_sectors_from_json(json_path: str):
    with open(json_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    sectors = set()
    for entry in data:
        category = entry.get("category", {})
        sector = category.get("sector")
        if sector:
            sectors.add(sector)

    print("Unique sectors found:")
    for s in sorted(sectors):
        print(f"- {s}")

    return sectors

def get_unique_tags_from_json(json_path: str):
    with open(json_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    tags = set()
    for entry in data:
        for tag in entry.get("tags", []):
            tags.add(tag)

    print("🏷️ Unique tags found:")
    for t in sorted(tags):
        print(f"- {t}")

    print(len(tags))
    return tags

def visualize_classification_report(y_true, y_pred, labels, output_dir="visualizations"):
    """
    Visualize classification report with confusion matrix and per-class accuracy.
    
    Args:
        y_true: List of ground truth labels
        y_pred: List of predicted labels
        labels: List of all class labels
        output_dir: Directory to save PNGs
    """
    os.makedirs(output_dir, exist_ok=True)

    # === Confusion Matrix ===
    cm = confusion_matrix(y_true, y_pred, labels=labels)
    plt.figure(figsize=(10, 8))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", xticklabels=labels, yticklabels=labels)
    plt.xlabel("Predicted")
    plt.ylabel("True")
    plt.title("Confusion Matrix")
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "confusion_matrix.png"))
    plt.close()

    # === Per-category Accuracy Bar Plot ===
    report_dict = classification_report(y_true, y_pred, labels=labels, output_dict=True)
    category_acc = {label: report_dict[label]["recall"] for label in labels if label in report_dict}

    plt.figure(figsize=(12, 6))
    sns.barplot(x=list(category_acc.keys()), y=list(category_acc.values()), palette="viridis")
    plt.ylabel("Recall (Accuracy per Category)")
    plt.ylim(0, 1)
    plt.title("Per-Category Accuracy")
    plt.xticks(rotation=45, ha="right")
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "category_accuracy.png"))
    plt.close()

    # === Save Classification Report as JSON ===
    with open(os.path.join(output_dir, "classification_report.json"), "w") as f:
        json.dump(report_dict, f, indent=4, ensure_ascii=False)

def plot_model_comparisons(evaluation_result_paths, results_paths, labels, output_path="model_comparison.png"):
    """
    Create a bar plot comparing multiple models based on accuracy, precision, recall, and F1-score.

    Args:
        evaluation_result_paths (List[str]): Paths to JSONs containing `weighted avg` metrics.
        results_paths (List[str]): Paths to JSONs containing `metrics` block.
        labels (List[str]): List of labels (names) for each model.
        output_path (str): Where to save the resulting plot.
    """
    assert len(labels) == len(evaluation_result_paths) + len(results_paths), "Mismatch in number of labels and JSONs."

    metric_names = ["Accuracy", "Precision", "Recall", "F1-Score"]
    model_scores = []

    # Read evaluation_results (classification_report-style)
    for path in evaluation_result_paths:
        with open(path) as f:
            data = json.load(f)
            model_scores.append([
                data["accuracy"],
                data["weighted avg"]["precision"],
                data["weighted avg"]["recall"],
                data["weighted avg"]["f1-score"]
            ])

    # Read results (metrics-style)
    for path in results_paths:
        with open(path) as f:
            data = json.load(f)["metrics"]
            model_scores.append([
                data["accuracy"],
                data["precision"],
                data["recall"],
                data["f1_score"]
            ])

    # Plot setup
    num_models = len(model_scores)
    num_metrics = len(metric_names)
    x = np.arange(num_metrics)
    width = 0.8 / num_models  # total width divided among models

    plt.figure(figsize=(10, 6))

    for i, scores in enumerate(model_scores):
        positions = x + i * width - (width * (num_models - 1) / 2)
        bars = plt.bar(positions, scores, width=width, label=labels[i])
        for bar in bars:
            height = bar.get_height()
            plt.text(bar.get_x() + bar.get_width()/2, height + 0.01, f"{height:.2f}",
                     ha='center', va='bottom', fontsize=8)

    plt.xticks(ticks=x, labels=metric_names)
    plt.ylim(0, 1.05)
    plt.ylabel("Score")
    plt.title("Model Comparison by Metric")
    plt.legend()
    plt.tight_layout()
    plt.savefig(output_path)
    plt.close()