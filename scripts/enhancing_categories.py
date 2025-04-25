import json
import os
import random
import torch
from transformers import pipeline
from scripts.utils import load_json
from dotenv import load_dotenv

load_dotenv('.env')

hf_token = os.environ["HF_TOKEN"]

# Load dataset
def load_yelp_dataset(path):
    return load_json(path)

# Extract unique categories
def extract_unique_categories(data):
    unique = set()
    for entry in data:
        unique.update([cat.strip() for cat in entry.get("categories", [])])
    return sorted(list(unique))

# Find example businesses for a given category
def get_example_businesses(data, category, max_samples=3):
    matches = [entry['name'] for entry in data if category in entry.get("categories", [])]
    return random.sample(matches, min(len(matches), max_samples))

# Prompt format for the LLM
def build_prompt(category, example_names):
    return (
        f"Describe the domain of the business category '{category}' "
        f"Some example of companies in this business category: {', '.join(example_names)}. "
        "Do not mention any business names."
        "Start your description with 'Focus on' and list only 5 specific relevant service or product keywords to the business category."
        "Provide only a brief single sentence with comma seperated keywords."
    )

# Generate with local or HF pipeline
def generate_descriptions(categories, data, model_name="tiiuae/falcon-rw-1b", max_tokens=50, tags=True):
    pipe = pipeline("text-generation", model=model_name, device=0, max_new_tokens=max_tokens, do_sample=False, temperature=0.0, num_return_sequences=1, token=hf_token, model_kwargs={"torch_dtype": torch.float16})

    results = {}
    for i, cat in enumerate(categories):
        if tags:
            examples = get_example_businesses_by_tag(data, cat)
        else:
            examples = get_example_businesses(data, cat)
        if not examples:
            continue

        prompt = build_prompt(cat, examples)
        print(f"Generating description for category: {cat} - {i}/{len(categories)}")

        output = pipe(prompt)[0]["generated_text"]
        cleaned = output.split("Start your description with")[0].strip()
        generated = output.replace(prompt, "").strip().split('\n')[0]

        print(generated)

        generated = generated if generated.lower().startswith("focus on") else f"Focus on {generated.strip('.')}"
        results[cat] = f'{cat}: {generated.replace("Focus on", "").strip()}'

    return results

def extract_unique_tags(data):
    unique_tags = set()
    for entry in data:
        unique_tags.update([tag.strip() for tag in entry.get("tags", [])])
    return sorted(list(unique_tags))


def get_example_businesses_by_tag(data, tag, max_samples=3):
    matches = [entry["name"] for entry in data if tag in entry.get("tags", [])]
    return random.sample(matches, min(len(matches), max_samples))

# Save to JSON
def save_results(results, path="yelp_category_descriptions.json"):
    with open(path, "w") as f:
        json.dump(results, f, indent=2, ensure_ascii=False)
    print(f"Saved descriptions to {path}")

#if __name__ == "__main__":
#    dataset_path = "resources/train_companies.json"
#    data = load_yelp_dataset(dataset_path)
#    yelp_categories = extract_unique_categories(data)
#    forbes_tags = extract_unique_tags(data)
#    descriptions = generate_descriptions(forbes_tags, data, model_name="meta-llama/Meta-Llama-3.1-8B-Instruct", tags=True)
#    save_results(descriptions, path="forbes_category_descriptions.json")
