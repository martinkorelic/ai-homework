# Company category classifier

## Overview of repository

- `.env-example` - environment variables needed to run
- `requirements.txt` - python packages, however might need to be adjusted based on local machine
- `company_classification_task_report.ipynb` - Full report of models and explanation of the system
- `scripts` - all the scripts used in the project

Data and models:
- `bert_classifier/` - BERT fine-tuned model
- `chroma_db_biz/` - Vector database of embedded business names
- `chroma_db_forbes_cat/` - Vector database of embedded Forbes subcategories
- `chroma_db_yelp_cat/` - Vector database of embedded Yelp subcategories
- `resources/` - the rest of datasets, training and test sets, graphs...

Model results:
- `results/` - Graph classifier results
- `results_bert/` - BERT classifier results
- `results_enriched/` - Enriched graph classifier results
- `results_rf/` - Random forest classifier results