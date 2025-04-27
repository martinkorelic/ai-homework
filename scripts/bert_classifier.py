import pandas as pd
import json
from transformers import BertTokenizer, BertForSequenceClassification, Trainer, TrainingArguments
from transformers import DataCollatorWithPadding
from sklearn.preprocessing import LabelEncoder
from torch.utils.data import Dataset
import torch

df = pd.read_csv("resources/train.csv")
df.dropna(inplace=True)
df["COMPANY"] = df["COMPANY"].str.strip()
df["CATEGORY"] = df["CATEGORY"].str.strip().str.lower()

with open("resources/train_companies.json") as f:
    enriched_entries = {entry["name"].strip().lower(): entry for entry in json.load(f)}

def build_input_text(company_name: str):
    entry = enriched_entries.get(company_name.lower())
    if entry:
        desc = entry.get("description", "")
        tags = ", ".join([t for t in entry.get("tags", []) if t not in {"B2B", "B2C", "B2G"}])
        return f"{company_name}. {tags}. {desc}"
    return company_name  # fallback to just company name

df["input_text"] = df["COMPANY"].apply(build_input_text)

le = LabelEncoder()
df["label"] = le.fit_transform(df["CATEGORY"])

class CompanyDataset(Dataset):
    def __init__(self, texts, labels, tokenizer):
        self.encodings = tokenizer(texts, truncation=True, padding=True)
        self.labels = labels

    def __getitem__(self, idx):
        item = {key: torch.tensor(val[idx]) for key, val in self.encodings.items()}
        item["labels"] = torch.tensor(self.labels[idx])
        return item

    def __len__(self):
        return len(self.labels)

tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")
dataset = CompanyDataset(df["input_text"].tolist(), df["label"].tolist(), tokenizer)

num_labels = df["label"].nunique()
model = BertForSequenceClassification.from_pretrained("bert-base-uncased", num_labels=num_labels)

training_args = TrainingArguments(
    output_dir="./bert_company_classifier",
    evaluation_strategy="no",
    save_strategy="epoch",
    logging_dir="./logs",
    num_train_epochs=2,
    per_device_train_batch_size=8,
    per_device_eval_batch_size=8,
    learning_rate=2e-5,
    weight_decay=0.01,
    save_total_limit=2,
    load_best_model_at_end=True,
)

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=dataset,
    eval_dataset=dataset,
    tokenizer=tokenizer,
    data_collator=DataCollatorWithPadding(tokenizer)
)

trainer.train()