# Project Setup

This guide outlines the steps to set up the environment and download the necessary datasets for this project.

## 1. Install Dependencies

First, you need to install the required Python packages listed in the `requirements.txt` file.

```bash
pip install -r requirements.txt
```

---

## 2. Download Datasets

Next, download the required datasets from Hugging Face Hub. These commands will download the `worldmodel_raw_data` and `worldmodel_tokenized_data` datasets into a local directory named `data`.

Make sure you have `huggingface-cli` installed and you are logged in (`huggingface-cli login`).

```bash
# Download the raw data
huggingface-cli download 1x-technologies/worldmodel_raw_data --repo-type dataset --local-dir data

# Download the processed world model data
huggingface-cli download 1x-technologies/worldmodel --repo-type dataset --local-dir data
```

After completing these steps, your environment will be ready and the required data will be available in the `data/` directory.