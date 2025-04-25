# Setup

To set up the environment:

- Use python `requirements.txt` to get all the modules. You might need to consider that some packages are specific, such as `torch` which has to installed based on machine.
- Create `.env` file from `.env-example` (you just need Huggingface token (`HF_TOKEN`), but also not completely necessary)
- To get the models `git lfs pull` or `git clone --recursive`
- To run models and test, the code examples for running are in `company_classification_task_report.ipynb`