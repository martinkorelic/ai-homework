from transformers import BertForSequenceClassification, BertTokenizer
import torch
from scripts.utils import load_json
import pandas as pd
from sklearn.preprocessing import LabelEncoder
import pickle, json
from sklearn.metrics import classification_report

def load_model_and_predict(checkpoint_dir: str, company_name: str, enriched_data: dict, label_encoder) -> str:
    """
    Loads model from checkpoint directory and predicts the category for a company.
    If enriched data is available, uses it in the input.

    Args:
        checkpoint_dir (str): Path to trained model checkpoint
        company_name (str): Company name to classify
        enriched_data (dict): Dict of enriched entries (name → metadata)
        label_encoder (LabelEncoder): Fitted label encoder from training

    Returns:
        str: Predicted category
    """
    tokenizer = BertTokenizer.from_pretrained(checkpoint_dir)
    model = BertForSequenceClassification.from_pretrained(checkpoint_dir)
    model.eval()

    def build_input(company, enriched_data):
        # Normalize the company name for consistent matching
        company_lower = company.lower()
        
        # Find the corresponding entry in the enriched_data
        entry = enriched_data.get(company_lower, None)
        
        if entry:
            # If we find the entry, collect the relevant fields
            tags = ", ".join([t for t in entry.get("tags", []) if t not in {"B2B", "B2C", "B2G"}])
            description = entry.get("description", "")
            
            # Return a combined string of company name, tags, and description
            return f"{company}. {tags}. {description}"
        
        # If no entry is found, return just the company name (fallback)
        return company

    input_text = build_input(company_name, enriched_data)
    inputs = tokenizer(input_text, return_tensors="pt", truncation=True)

    with torch.no_grad():
        outputs = model(**inputs)
        probs = torch.softmax(outputs.logits, dim=-1)
        pred_label = torch.argmax(probs, dim=1).item()
        return label_encoder.inverse_transform([pred_label])[0]

def create_label_encoder(train_csv_path):

    train_df = pd.read_csv(train_csv_path)

    train_df["CATEGORY"] = train_df["CATEGORY"].str.lower().str.strip()

    le = LabelEncoder()
    le.fit(train_df["CATEGORY"])

    return le


def evaluate_on_test_set(train_csv_path, test_csv_path, enriched_json_path, checkpoint_dir):
    """
    Evaluates model on test set using test.csv and enriched test_companies.json

    Args:
        train_csv_path (str): Path to train.csv
        test_csv_path (str): Path to test.csv
        enriched_json_path (str): Path to enriched test JSON
        checkpoint_dir (str): Path to model checkpoint
    """

    le = create_label_encoder(train_csv_path)

    df_test = pd.read_csv(test_csv_path)
    df_test.dropna(inplace=True)
    df_test["COMPANY"] = df_test["COMPANY"].str.strip()
    df_test["CATEGORY"] = df_test["CATEGORY"].str.strip().str.lower()

    with open(enriched_json_path) as f:
        enriched_data = {entry["name"].strip().lower(): entry for entry in json.load(f) if entry["name"] != None}

    y_true = []
    y_pred = []

    i = 0
    for _, row in df_test.iterrows():
        company = row["COMPANY"]
        true_label = row["CATEGORY"]
        pred_label = load_model_and_predict(checkpoint_dir, company, enriched_data, le)

        print(f"Evaluating {i} - {company}")
        y_true.append(true_label)
        y_pred.append(pred_label)
        i += 1
    cls_report = classification_report(y_true, y_pred)

    print(cls_report)

    return (y_true, y_pred)

#if __name__ == "__main__":
#    evaluate_on_test_set('resources/train.csv', 'resources/test.csv', 'resources/test_companies.json', 'bert_company_classifier/checkpoint-870', le)