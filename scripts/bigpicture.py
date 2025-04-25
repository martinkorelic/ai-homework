import os
import csv
import json
import requests
from typing import Optional, Dict, Any, List
import time
from dotenv import load_dotenv

load_dotenv('.env')

API_KEY = os.getenv("BIGPICTURE_API_KEY")
if not API_KEY:
    raise RuntimeError("Please set the BIGPICTURE_API_KEY environment variable.")

HEADERS = {
    "Authorization": API_KEY,
    "Accept": "application/json"
}

def get_best_domain_hit(company_name: str) -> Optional[Dict[str, Any]]:
    resp = requests.get("https://company.bigpicture.io/v2/companies/search", headers=HEADERS, params={"name": company_name})
    if resp.status_code != 200:
        print(f"[ERROR] Search failed for {company_name}: {resp.status_code}")
        return None
    data = resp.json().get("data", [])
    if not data:
        return None
    return max(data, key=lambda x: x.get("confidence", 0))  # best confidence

def get_company_details(domain: str) -> Optional[Dict[str, Any]]:
    resp = requests.get("https://company.bigpicture.io/v1/companies/find", headers=HEADERS, params={"domain": domain})
    if resp.status_code != 200:
        return None
    return resp.json()

def extract_relevant_fields(search_result: Dict[str, Any], find_result: Dict[str, Any]) -> Dict[str, Any]:
    return {
        "name": search_result.get("name"),
        "domain": search_result.get("domain"),
        "confidence": search_result.get("confidence"),
        "tags": find_result.get("tags"),
        "category": find_result.get("category"),
        "linkedin": find_result.get("linkedin"),
        "description": find_result.get("description"),
        "id": find_result.get("id")
    }

def enrich_companies_from_csv(csv_path: str, output_json_path: str):
    enriched = []

    with open(csv_path, newline='', encoding='utf-8') as csvfile:
        reader = csv.DictReader(csvfile)
        for row in reader:
            company_name = row.get("COMPANY")
            if not company_name:
                continue

            print(f"Querying for: {company_name}")
            search_hit = get_best_domain_hit(company_name)
            if not search_hit:
                print(f"No search results for {company_name}")
                continue

            domain = search_hit.get("domain")
            if not domain:
                print(f"No domain found for {company_name}")
                continue

            find_result = get_company_details(domain)
            if not find_result:
                print(f"Failed to fetch details for domain: {domain}")
                continue

            record = extract_relevant_fields(search_hit, find_result)
            enriched.append(record)

    with open(output_json_path, "w", encoding="utf-8") as f:
        json.dump(enriched, f, indent=2, ensure_ascii=False)
    print(f"Saved {len(enriched)} enriched companies to '{output_json_path}'")


def test_bigpicture_api(company_name: str = "Google") -> dict:
    search_hit = get_best_domain_hit(company_name)
    if not search_hit:
        print(f"No search results for {company_name}")
        return {}

    time.sleep(1)

    domain = search_hit.get("domain")
    if not domain:
        print(f"No domain found for {company_name}")
        return {}

    find_result = get_company_details(domain)
    if not find_result:
        print(f"Failed to fetch details for domain: {domain}")
        return {}

    record = extract_relevant_fields(search_hit, find_result)
    print(f"✅ Success: {record['name']} ({record['domain']})")
    return record


def enrich_companies_from_csv(csv_path: str, output_json_path: str, progress_path: str = "progress.txt"):
    enriched = []
    start_index = 0

    # Resume from progress if exists
    if os.path.exists(progress_path):
        with open(progress_path, "r") as pf:
            try:
                start_index = int(pf.read().strip())
                print(f"🔁 Resuming from row {start_index}")
            except ValueError:
                print("⚠️ Progress file corrupted. Starting from scratch.")

    # Load existing data if JSON already exists
    if os.path.exists(output_json_path):
        with open(output_json_path, "r", encoding="utf-8") as jf:
            enriched = json.load(jf)

    with open(csv_path, newline='', encoding='utf-8') as csvfile:
        reader = list(csv.DictReader(csvfile))
        for i, row in enumerate(reader[start_index:], start=start_index):
            company_name = row.get("COMPANY")
            if not company_name:
                continue

            print(f"[{i}] Querying: {company_name}")
            search_hit = get_best_domain_hit(company_name)
            time.sleep(1)

            if not search_hit:
                print(f"No search results for {company_name}")
                continue

            domain = search_hit.get("domain")
            if not domain:
                print(f"No domain found for {company_name}")
                continue

            find_result = get_company_details(domain)
            if not find_result:
                print(f"Failed to fetch details for domain: {domain}")
                continue

            record = extract_relevant_fields(search_hit, find_result)
            enriched.append(record)

            # Save progress every 10 records
            if len(enriched) % 10 == 0:
                with open(output_json_path, "w", encoding="utf-8") as jf:
                    json.dump(enriched, jf, indent=2, ensure_ascii=False)
                with open(progress_path, "w") as pf:
                    pf.write(str(i + 1))
                print(f"💾 Progress saved at row {i + 1}")

    # Final save
    with open(output_json_path, "w", encoding="utf-8") as jf:
        json.dump(enriched, jf, indent=2, ensure_ascii=False)
    with open(progress_path, "w") as pf:
        pf.write(str(len(reader)))

#if __name__ == "__main__":
    # Process full CSV
    #fetch_and_save_all("test.csv", "bigpicture_results.json")
    #enrich_companies_from_csv('resources/train.csv', 'train_companies.json')
