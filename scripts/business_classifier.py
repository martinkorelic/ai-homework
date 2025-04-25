import os
import csv
import json, time
import numpy as np
import networkx as nx
from typing import List, Dict
import chromadb
from sentence_transformers import SentenceTransformer
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix, classification_report
import seaborn as sns
from collections import defaultdict
import matplotlib.pyplot as plt
from scripts.utils import load_json

class BusinessClassifier:
    """
    A comprehensive business classifier that uses multiple models and data sources:
    - Multi-model embeddings
    - Chroma vector database
    - Category similarity graph
    """
    
    def __init__(
        self,
        model_names: List[str] = ["all-MiniLM-L6-v2", "BAAI/bge-m3"],
        business_db_path: str = "./chroma_db_businesses",
        yelp_category_db_path: str = "./chroma_db_yelp_categories",
        forbes_category_db_path: str = "./chroma_db_forbes_categories",
        yelp_graph_path: str = "./yelp_category_similarity_graph.graphml",
        forbes_graph_path: str = "./forbes_category_similarity_graph.graphml",
        forbes_path: str = "./train_companies.json",
        yelp2forbes_path: str = "./yelp2forbes_business_data.json",
        business_collection_name: str = "businesses_multimodel",
        category_collection_name: str = "yelp_category_descriptions",
        enrichment_db_path: str = ""
    ):
        """
        Initialize the business classifier with all necessary components.
        
        Args:
            model_names: List of embedding model names to use
            business_db_path: Path to the Chroma DB with business embeddings
            yelp_category_db_path: Path to the Chroma DB with Yelp category embeddings
            forbes_category_db_path: Path to the Chroma DB with Forbes category embeddings
            yelp_graph_path: Path to the GraphML file with Yelp category relationships
            forbes_graph_path: Path to the GraphML file with Forbes category relationships
            train_companies_path: Path to JSON file with company details and tags
            yelp2forbes_path: Path to JSON file with Yelp to Forbes category mappings
            business_collection_name: Name of the business collection in Chroma
            yelp_category_collection_name: Name of the Yelp category collection in Chroma
            forbes_category_collection_name: Name of the Forbes category collection in Chroma
        """
        self.model_names = model_names
        self.models = {}
        
        print(f"Loading embedding models: {', '.join(model_names)}")
        for model_name in model_names:
            self.models[model_name] = SentenceTransformer(model_name)
        
        print(f"Loading business database from {business_db_path}")
        self.business_db_client = chromadb.PersistentClient(path=business_db_path)
        try:
            self.business_collection = self.business_db_client.get_collection(business_collection_name)
            print(f"Successfully loaded business collection '{business_collection_name}'")
        except Exception as e:
            print(f"Warning: Could not load business collection: {e}")
            self.business_collection = None
        
        print(f"Loading Yelp category database from {yelp_category_db_path}")
        self.yelp_category_db_client = chromadb.PersistentClient(path=yelp_category_db_path)
        try:
            self.yelp_category_collection = self.yelp_category_db_client.get_collection(category_collection_name)
            print(f"Successfully loaded Yelp category collection '{category_collection_name}'")
        except Exception as e:
            print(f"Warning: Could not load Yelp category collection: {e}")
            self.yelp_category_collection = None

        print(f"Loading Forbes category database from {forbes_category_db_path}")
        self.forbes_category_db_client = chromadb.PersistentClient(path=forbes_category_db_path)
        try:
            self.forbes_category_collection = self.forbes_category_db_client.get_collection(category_collection_name)
            print(f"Successfully loaded Forbes category collection '{category_collection_name}'")
        except Exception as e:
            print(f"Warning: Could not load Forbes category collection: {e}")
            self.forbes_category_collection = None
        
        print(f"Loading Yelp category graph from {yelp_graph_path}")
        self.yelp_graph = self._load_graph(yelp_graph_path)

        print(f"Loading Forbes category graph from {forbes_graph_path}")
        self.forbes_graph = self._load_graph(forbes_graph_path)
        
        # Extract classification categories (main classification targets)
        self.classification_categories = [node for node, attrs in self.yelp_graph.nodes(data=True) 
                                 if attrs.get('type') == 'forbes']
        print(f"Loaded {len(self.classification_categories)} classification categories")
        
        # Extract Yelp subcategories
        self.yelp_categories = [node for node, attrs in self.yelp_graph.nodes(data=True) 
                               if attrs.get('type') == 'yelp']
        print(f"Loaded {len(self.yelp_categories)} Yelp subcategories")

        # Extract Forbes subcategories
        self.forbes_subcategories = [node for node, attrs in self.forbes_graph.nodes(data=True) 
                               if attrs.get('type') == 'tag']
        print(f"Loaded {len(self.forbes_subcategories)} Forbes subcategories")
        
        # Load additional JSON data
        print(f"Loading company data from {forbes_path}")
        try:
            with open(forbes_path, 'r') as f:
                self.train_companies = json.load(f)
            print(f"Loaded {len(self.train_companies)} company records")
        except Exception as e:
            print(f"Warning: Could not load company data: {e}")
            self.train_companies = []
            
        print(f"Loading Yelp to Forbes mapping from {yelp2forbes_path}")
        try:
            with open(yelp2forbes_path, 'r') as f:
                self.yelp2forbes_data = json.load(f)
            print(f"Loaded {len(self.yelp2forbes_data)} Yelp-Forbes mappings")
        except Exception as e:
            print(f"Warning: Could not load Yelp to Forbes mapping: {e}")
            self.yelp2forbes_data = []

        # Load enrichment db
        self.enrichment_db = load_json(enrichment_db_path)
    
    def _load_graph(self, graph_path: str) -> nx.Graph:
        """
        Load the enhanced GraphML file.
        
        Args:
            graph_path: Path to the enhanced GraphML file
            
        Returns:
            NetworkX graph
        """
        # Simply load the graph since enhancement is done separately
        try:
            graph = nx.read_graphml(graph_path)
            print(f"Successfully loaded graph with {len(graph.nodes())} nodes and {len(graph.edges())} edges")
            return graph
        except Exception as e:
            print(f"Error loading graph: {e}")
            # Return an empty graph as fallback
            return nx.Graph()
    
    def embed_text(self, text_list: List[str]) -> np.ndarray:
        """
        Embed a list of texts using all models and concatenate the embeddings.
        
        Args:
            text_list: List of texts to embed
            
        Returns:
            Concatenated normalized embeddings
        """
        all_embeddings = []
        
        for model_name, model in self.models.items():
            embeddings = model.encode(text_list, convert_to_numpy=True)
            # Normalize embeddings
            embeddings_norm = embeddings / np.linalg.norm(embeddings, axis=1, keepdims=True)
            all_embeddings.append(embeddings_norm)
        
        # If only one model, return its embeddings directly
        if len(all_embeddings) == 1:
            return all_embeddings[0]
        
        # Otherwise concatenate the embeddings
        combined_embeddings = np.concatenate(all_embeddings, axis=1)
        return combined_embeddings
    
    def enrich_business_name(self, business_name: str) -> str:
        """
        Enrich a business name with additional context if an enrichment API is available.
        
        Args:
            business_name: The name of the business to enrich
            
        Returns:
            Enriched business description or the original name if no enrichment is available
        """
        if self.enrichment_api:
            try:
                # This is a placeholder for the actual enrichment logic
                # Replace with your specific API implementation
                enriched_info = self.enrichment_api.get_company_description(business_name)
                if enriched_info and 'description' in enriched_info and enriched_info['description']:
                    return enriched_info['description']
            except Exception as e:
                print(f"Error enriching business name '{business_name}': {e}")
        
        # Return the original name if no enrichment is available
        return business_name
    
    def get_business_description(self, business_name):

        if self.enrichment_db:
            for d in self.enrichment_db:

                name = d.get("name", None)

                if not name:
                    continue
                if name == business_name:
                    clean = self.clean_description(d.get("description", business_name))

                    if clean is not None and len(clean) > 10:
                        return clean
                    else:
                        return business_name
        
        return business_name
    
    def clean_description(self, desc):
        # TODO if needed
        return desc

    def find_similar_businesses(
        self, 
        query_name: str, 
        top_k: int = 5
    ) -> Dict:
        """
        Find businesses similar to the query name in the business collection.
        
        Args:
            query_name: The name of the business to find similar businesses for
            top_k: Number of similar businesses to return
            
        Returns:
            Dictionary with similar businesses and their categories
        """
        if not self.business_collection:
            return {"error": "Business collection not available"}
        
        # Embed the query
        query_embedding = self.embed_text([query_name])
        
        # Query the business collection
        results = self.business_collection.query(
            query_embeddings=query_embedding.tolist(),
            n_results=top_k,
            include=["distances", "metadatas", "documents"]
        )
        
        # Format the results
        similar_businesses = []
        
        # Apply softmax normalization to similarity scores
        if len(results["documents"][0]) > 0:
            distances = np.array(results["distances"][0])
            similarities = 1 - distances  # Convert distances to similarities
            
            # Apply softmax to get normalized probabilities
            normalized_similarities = np.exp(similarities) / np.sum(np.exp(similarities))
            
            # Create the result objects with normalized scores
            for i in range(min(top_k, len(results["documents"][0]))):
                business_name = results["documents"][0][i]
                category = results["metadatas"][0][i].get("category", "")
                normalized_similarity = float(normalized_similarities[i])
                
                similar_businesses.append({
                    "name": business_name,
                    "category": category,
                    "similarity": normalized_similarity
                })
        
        return {
            "query": query_name,
            "similar_businesses": similar_businesses
        }
    
    def find_similar_yelp_categories(
        self, 
        query_name: str, 
        top_k: int = 5
    ) -> Dict:
        """
        Find Yelp categories similar to the query name in the category collection.
        
        Args:
            query_name: The name or description to find similar categories for
            top_k: Number of similar categories to return
            
        Returns:
            Dictionary with similar categories
        """
        if not self.yelp_category_collection:
            return {"error": "Yelp category collection not available"}
        
        # Embed the query
        query_embedding = self.embed_text([query_name])
        
        # Query the category collection
        results = self.yelp_category_collection.query(
            query_embeddings=query_embedding.tolist(),
            n_results=top_k,
            include=["distances", "metadatas", "documents"]
        )
        # Format the results
        similar_categories = []
        
        # Convert distances to raw similarity scores
        if len(results["documents"][0]) > 0:
            distances = np.array(results["distances"][0])
            similarities = 1 - distances  # Convert distances to similarities
            
            # Apply softmax to get normalized probabilities
            normalized_similarities = np.exp(similarities) / np.sum(np.exp(similarities))
            
            # Create the result objects with normalized scores
            for i in range(min(top_k, len(results["documents"][0]))):
                description = results["documents"][0][i]
                category = results["metadatas"][0][i].get("category", "")
                normalized_similarity = float(normalized_similarities[i])
                
                similar_categories.append({
                    "category": category,
                    "description": description,
                    "similarity": normalized_similarity
                })
        
        return {
            "query": query_name,
            "similar_categories": similar_categories
        }
    
    def find_similar_forbes_categories(
        self, 
        query_name: str, 
        top_k: int = 5
    ) -> Dict:
        """
        Find Forbes categories similar to the query name in the category collection.
        
        Args:
            query_name: The name or description to find similar categories for
            top_k: Number of similar categories to return
            
        Returns:
            Dictionary with similar categories
        """
        if not self.forbes_category_collection:
            return {"error": "Forbes category collection not available"}
        
        # Embed the query
        query_embedding = self.embed_text([query_name])
        
        # Query the category collection
        results = self.forbes_category_collection.query(
            query_embeddings=query_embedding.tolist(),
            n_results=top_k,
            include=["distances", "metadatas", "documents"]
        )
        
        # Format the results
        similar_categories = []
        
        # Convert distances to raw similarity scores
        if len(results["documents"][0]) > 0:
            distances = np.array(results["distances"][0])
            similarities = 1 - distances  # Convert distances to similarities
            
            # Apply softmax to get normalized probabilities
            normalized_similarities = np.exp(similarities) / np.sum(np.exp(similarities))
            
            # Create the result objects with normalized scores
            for i in range(min(top_k, len(results["documents"][0]))):
                description = results["documents"][0][i]
                category = results["metadatas"][0][i].get("category", "")
                normalized_similarity = float(normalized_similarities[i])
                
                similar_categories.append({
                    "category": category,
                    "description": description,
                    "similarity": normalized_similarity
                })
        
        return {
            "query": query_name,
            "similar_categories": similar_categories
        }
    
    def apply_norm(self, category_scores: dict) -> dict:
        """
        Applies min-max normalization to a dictionary of category scores.
        
        Args:
            category_scores: Dictionary mapping categories to their scores
            
        Returns:
            Dictionary with the same categories but with min-max normalized scores
            that range from 0 to 1, with their sum also normalized to 1.0
        """
        # Extract categories and scores
        categories = list(category_scores.keys())
        scores = np.array(list(category_scores.values()))
        
        # Handle the case where min and max are the same
        if np.max(scores) == np.min(scores):
            # If all scores are equal, return equal probability for all
            equal_value = 1.0 / len(scores)
            return {category: equal_value for category in categories}
        
        # Apply min-max normalization
        min_val = np.min(scores)
        max_val = np.max(scores)
        normalized_scores = (scores - min_val) / (max_val - min_val)
        
        # Ensure values sum to 1
        sum_normalized = np.sum(normalized_scores)
        if sum_normalized > 0:
            normalized_scores = normalized_scores / sum_normalized
        
        # Create new dictionary with normalized scores
        result = {categories[i]: float(normalized_scores[i]) for i in range(len(categories))}
        
        return result

    def _compute_categories_from_yelp(
        self, 
        yelp_categories: List[Dict]
    ) -> Dict[str, float]:
        """
        Compute scores for Forbes categories based on Yelp category connections.
        
        Args:
            yelp_categories: List of similar Yelp categories with similarity scores
            
        Returns:
            Dictionary mapping Forbes categories to scores
        """
        forbes_scores = {category: 0.0 for category in self.classification_categories}
        
        # Add scores from Yelp categories based on graph connections
        for yelp_cat in yelp_categories:
            yelp_name = yelp_cat["category"]
            yelp_similarity = yelp_cat.get('similarity', 0.0)
            
            # Find all connected Forbes categories and their weights
            for neighbor in self.yelp_graph.neighbors(yelp_name):
                if neighbor in forbes_scores:
                    edge_data = self.yelp_graph.get_edge_data(yelp_name, neighbor)
                    edge_weight = edge_data.get('weight', 0.0)
                    
                    # Combine Yelp category similarity with edge weight
                    forbes_scores[neighbor] += yelp_similarity + edge_weight
        
        forbes_scores = self.apply_norm(forbes_scores)

        return forbes_scores
    
    def _compute_categories_from_tags(
        self, 
        tags: List[str]
    ) -> Dict[str, float]:
        """
        Compute scores for Forbes categories based on company tags.
        
        Args:
            tags: List of company tags
            
        Returns:
            Dictionary mapping Forbes categories to scores
        """
        forbes_scores = {category: 0.0 for category in self.classification_categories}
        
        # For each tag, find connections in the Forbes graph
        for tag in tags:

            tag_name = tag.get('category', None)

            # Skip if this tag is not in our graph
            if tag_name not in self.forbes_graph:
                continue

            tag_similarity = tag.get('similarity', 0.0)
            
            # Find all connected Forbes categories and their weights
            for neighbor in self.forbes_graph.neighbors(tag_name):
                if neighbor in forbes_scores:  # Ensure it's a Forbes category
                    edge_data = self.forbes_graph.get_edge_data(tag_name, neighbor)
                    edge_weight = edge_data.get('sim_score', 0.0)
                    edge_weight2 = edge_data.get('weight', 0.0)
                    
                    # Add the edge weight to the category score
                    forbes_scores[neighbor] += tag_similarity + edge_weight + edge_weight2

        forbes_scores = self.apply_norm(forbes_scores)

        return forbes_scores
    
    def _find_company_in_training_data(
        self, 
        business_name: str
    ) -> Dict:
        """
        Find a company in the training data by name.
        
        Args:
            business_name: Name of the business to find
            
        Returns:
            Company data if found, empty dict if not
        """
        for company in self.train_companies:
            if company["name"].lower() == business_name.lower():
                return company
        return {}
    
    def _find_yelp_categories_in_mapping(
        self, 
        business_name: str
    ) -> List[str]:
        """
        Find Yelp categories for a business in the Yelp2Forbes mapping.
        
        Args:
            business_name: Name of the business to find
            
        Returns:
            List of Yelp categories
        """
        for business in self.yelp2forbes_data:
            if business["name"].lower() == business_name.lower():
                return business.get("categories", [])
        return []
    
    def predict(
        self, 
        business_name: str, 
        use_enrichment: bool = False,
        top_k_businesses: int = 5,
        top_k_yelp: int = 10,
        top_k_forbes: int = 10,
        alpha: float = 0.9,
        beta: float = 0.5
    ) -> Dict:
        """
        Predict the Forbes industry category for a business name.
        
        Args:
            business_name: Name of the business to classify
            use_enrichment: Whether to use enrichment API if available
            top_k_businesses: Number of similar businesses to consider
            top_k_categories: Number of similar categories to consider
            top_k_results: Number of top Forbes categories to return
            alpha: Threshold for considering similar business categories directly
            beta: How much to weight between Yelp embedding categories & Forbes embedding categories
            
        Returns:
            Dictionary with predicted categories and confidence scores
        """
        
        similar_businesses_result = self.find_similar_businesses(
            business_name, 
            top_k=top_k_businesses
        )
        
        if "error" in similar_businesses_result:
            print(f"Warning: {similar_businesses_result['error']}")
            similar_businesses = []
        else:
            similar_businesses = similar_businesses_result.get("similar_businesses", [])
        
        # Initialize scores dictionaries
        yelp_derived_scores = {category: 0.0 for category in self.classification_categories}
        forbes_derived_scores = {category: 0.0 for category in self.classification_categories}
        
        
        # Step 1: Optionally enrich the business name with additional context
        if use_enrichment:
            enriched_business = self.get_business_description(business_name)
        else:
            enriched_business = business_name

        # Get similar Yelp categories
        similar_yelp_result = self.find_similar_yelp_categories(
            enriched_business,
            top_k=top_k_yelp
        )
        
        if "error" not in similar_yelp_result:
            similar_yelp_categories = similar_yelp_result.get("similar_categories", [])
            yelp_derived_scores = self._compute_categories_from_yelp(similar_yelp_categories)

        # Get similar Forbes categories
        similar_forbes_result = self.find_similar_forbes_categories(
            enriched_business,
            top_k=top_k_forbes
        )

        if "error" not in similar_forbes_result:
            similar_forbes_categories = similar_forbes_result.get("similar_categories", [])

            forbes_derived_scores = self._compute_categories_from_tags(similar_forbes_categories)

        # Handle direct category matching for high similarity businesses
        for business in similar_businesses:
            if business["similarity"] > alpha and business["category"]:
                direct_category = business["category"]
                if direct_category in self.classification_categories:
                    yelp_derived_scores[direct_category] += business["similarity"]
                    forbes_derived_scores[direct_category] += business["similarity"]

        # Step 4: Combine scores from both sources
        combined_scores = {}
        for category in self.classification_categories:
            combined_scores[category] = beta * yelp_derived_scores.get(category, 0.0) + (1-beta) * forbes_derived_scores.get(category, 0.0)
        
        # Step 5: Sort categories by score and get top K
        sorted_categories = sorted(
            [(category, score) for category, score in combined_scores.items()],
            key=lambda x: x[1],
            reverse=True
        )
        
        # Step 6: Normalize scores to sum to 1 (convert to probabilities)
        top_categories = sorted_categories[:top_k_yelp]
        total_score = sum(score for _, score in top_categories)
        
        if total_score > 0:
            normalized_scores = [(cat, score / total_score) for cat, score in top_categories]
        else:
            normalized_scores = [(cat, 1.0 / len(top_categories)) for cat, score in top_categories]
        
        # Step 7: Prepare the final result
        predictions = []
        for category, probability in normalized_scores:
            predictions.append({
                "category": category,
                "probability": probability
            })
        
        result = {
            "business_name": business_name,
            "enriched_business": enriched_business if use_enrichment else None,
            "predictions": predictions,
            "similar_businesses": similar_businesses
        }
        
        return result
    
    def predict_batch(
        self,
        business_names: List[str],
        **kwargs
    ) -> List[Dict]:
        """
        Predict categories for a batch of business names.
        
        Args:
            business_names: List of business names to classify
            **kwargs: Additional arguments to pass to predict()
            
        Returns:
            List of prediction results
        """
        results = []
        for name in business_names:
            result = self.predict(name, **kwargs)
            results.append(result)
        return results
    
    def evaluate(
        self, 
        test_file_path: str = "test.csv", 
        results_dir: str = "results",
        use_enrichment: bool = False,
        **kwargs
    ) -> Dict:
        """
        Evaluate the classifier on a test set and generate performance metrics and plots.
        
        Args:
            test_file_path: Path to the test CSV file
            results_dir: Directory to save result plots
            use_enrichment: Whether to use enrichment API
            **kwargs: Additional arguments to pass to predict()
            
        Returns:
            Dictionary with evaluation metrics
        """
        # Create results directory if it doesn't exist
        os.makedirs(results_dir, exist_ok=True)
        
        # Load test data
        test_companies = []
        test_categories = []
        
        print(f"Loading test data from {test_file_path}")
        with open(test_file_path, 'r', encoding='utf-8') as f:
            reader = csv.reader(f)
            header = next(reader)  # Skip header
            for row in reader:
                if len(row) >= 2:  # Ensure the row has at least two columns
                    test_companies.append(row[0])
                    test_categories.append(row[1])
        
        print(f"Loaded {len(test_companies)} test cases")
        
        # Prepare for tracking metrics
        predicted_categories = []
        prediction_times = []
        prediction_confidences = []
        prediction_results = []
        
        # Run predictions for each company
        for i, company in enumerate(test_companies):
            print(f"Processing {i+1}/{len(test_companies)}: {company}")
            
            # Measure prediction time
            start_time = time.time()
            prediction = self.predict(company, use_enrichment=use_enrichment, **kwargs)
            elapsed_time = time.time() - start_time
            
            # Store results
            prediction_times.append(elapsed_time)
            
            # Get top prediction
            if prediction["predictions"]:
                top_prediction = prediction["predictions"][0]
                predicted_cat = top_prediction["category"]
                confidence = top_prediction["probability"]
            else:
                predicted_cat = "Unknown"
                confidence = 0.0
                
            predicted_categories.append(predicted_cat)
            prediction_confidences.append(confidence)
            
            # Store full prediction for later analysis
            prediction_results.append(prediction)
            
            # Print result
            true_cat = test_categories[i]
            correct = (predicted_cat == true_cat)
            print(f"  Predicted: {predicted_cat} ({confidence:.2%}), Actual: {true_cat}, Correct: {correct}, Time: {elapsed_time:.2f}s")
        
        # Calculate evaluation metrics
        unique_categories = sorted(list(set(test_categories)))
        
        # Calculate accuracy
        accuracy = accuracy_score(test_categories, predicted_categories)
        
        # Calculate precision, recall, and F1 score - handle potential errors with try/except
        try:
            precision = precision_score(test_categories, predicted_categories, average='weighted', zero_division=0)
            recall = recall_score(test_categories, predicted_categories, average='weighted', zero_division=0)
            f1 = f1_score(test_categories, predicted_categories, average='weighted', zero_division=0)
            
            # Generate classification report
            report = classification_report(test_categories, predicted_categories, output_dict=True)
            
            # Calculate confusion matrix
            cm = confusion_matrix(test_categories, predicted_categories, labels=unique_categories)
            
        except Exception as e:
            print(f"Warning: Error calculating some metrics: {e}")
            precision = recall = f1 = 0.0
            report = {}
            cm = np.zeros((len(unique_categories), len(unique_categories)))
        
        # Print overall metrics
        print("\nEvaluation Results:")
        print(f"Accuracy: {accuracy:.4f}")
        print(f"Precision: {precision:.4f}")
        print(f"Recall: {recall:.4f}")
        print(f"F1 Score: {f1:.4f}")
        print(f"Average prediction time: {np.mean(prediction_times):.2f}s")
        
        # Generate plots
        
        # 1. Confusion Matrix
        plt.figure(figsize=(12, 10))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=unique_categories, yticklabels=unique_categories)
        plt.title('Confusion Matrix')
        plt.xlabel('Predicted Category')
        plt.ylabel('True Category')
        plt.xticks(rotation=90)
        plt.yticks(rotation=0)
        plt.tight_layout()
        plt.savefig(f"{results_dir}/confusion_matrix.png")
        plt.close()
    
        # Category-wise Performance
        category_accuracy = defaultdict(lambda: {'correct': 0, 'total': 0})
        for true_cat, pred_cat in zip(test_categories, predicted_categories):
            category_accuracy[true_cat]['total'] += 1
            if true_cat == pred_cat:
                category_accuracy[true_cat]['correct'] += 1
        
        categories = []
        accuracies = []
        sample_counts = []
        
        for cat, counts in category_accuracy.items():
            categories.append(cat)
            accuracies.append(counts['correct'] / counts['total'] if counts['total'] > 0 else 0)
            sample_counts.append(counts['total'])
        
        # Sort by number of samples
        sorted_indices = np.argsort(sample_counts)[::-1]  # Descending order
        sorted_categories = [categories[i] for i in sorted_indices]
        sorted_accuracies = [accuracies[i] for i in sorted_indices]
        sorted_counts = [sample_counts[i] for i in sorted_indices]
        
        plt.figure(figsize=(12, 8))
        bars = plt.bar(range(len(sorted_categories)), sorted_accuracies, color='skyblue')
        
        # Add data value labels on top of bars
        for i, (acc, count) in enumerate(zip(sorted_accuracies, sorted_counts)):
            plt.text(i, acc + 0.02, f"{acc:.2f}\n(n={count})", ha='center', va='bottom', fontsize=8, rotation=0)
        
        plt.title('Category-wise Accuracy')
        plt.xlabel('Category')
        plt.ylabel('Accuracy')
        plt.xticks(range(len(sorted_categories)), sorted_categories, rotation=90)
        plt.ylim(0, 1.1)  # Leave space for text
        plt.grid(True, axis='y', alpha=0.3)
        plt.tight_layout()
        plt.savefig(f"{results_dir}/category_accuracy.png")
        plt.close()
        
        # 5. Top Misclassifications
        misclassification_pairs = defaultdict(int)
        for true_cat, pred_cat in zip(test_categories, predicted_categories):
            if true_cat != pred_cat:
                misclassification_pairs[(true_cat, pred_cat)] += 1
        
        # Save evaluation results as JSON
        evaluation_results = {
            "metrics": {
                "accuracy": accuracy,
                "precision": precision,
                "recall": recall,
                "f1_score": f1,
                "avg_prediction_time": float(np.mean(prediction_times))
            },
            "category_accuracy": {cat: {"accuracy": acc, "sample_count": count} 
                                 for cat, acc, count in zip(sorted_categories, sorted_accuracies, sorted_counts)},
            "classification_report": report
        }
        
        with open(f"{results_dir}/evaluation_results.json", 'w') as f:
            json.dump(evaluation_results, f, indent=2)
        
        return evaluation_results


# Example usage:
if __name__ == "__main__":
    # Initialize the classifier
    classifier = BusinessClassifier(
        model_names=["all-MiniLM-L6-v2", "BAAI/bge-m3"],
        business_db_path="./chroma_db_biz",
        yelp_category_db_path="./chroma_db_yelp_cat",
        forbes_category_db_path="./chroma_db_forbes_cat",
        business_collection_name="businesses_multi",
        category_collection_name="categories_multi",
        forbes_path="./resources/train_companies.json",
        yelp2forbes_path="./resources/yelp2forbes_business_data.json",
        yelp_graph_path="./resources/yelp2forbes_categories.graphml",
        forbes_graph_path="./resources/forbes_categories.graphml",
        enrichment_db_path="resources/test_companies.json"
    )
    
    # Evaluation with raw business names
    #evaluation_results = classifier.evaluate(
    #    test_file_path="resources/test.csv",
    #    results_dir="results",
    #    use_enrichment=False,
    #    top_k_businesses=3,
    #    top_k_yelp=10,
    #    top_k_forbes=10,
    #    beta=0.2
    #)

    # Evaluation with enriched business descriptions
    #evaluation_results = classifier.evaluate(
    #    test_file_path="resources/test.csv",
    #    results_dir="results",
    #    use_enrichment=False,
    #    top_k_businesses=3,
    #    top_k_yelp=10,
    #    top_k_forbes=10,
    #    beta=0.2
    #)