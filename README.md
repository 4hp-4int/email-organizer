Below is a suggested **README.md** reflecting the current state of the project and its files. Feel free to modify any section to fit your environment, deployment strategy, or specific instructions.

---

# Email Classification & Organization

This repository contains a system for classifying and organizing emails using a **BERTopic** model (potentially enhanced with custom embeddings). It integrates with Microsoft Graph to read emails from a mailbox (e.g. Outlook/Office 365), automatically classify them into topics, and move them to corresponding folders.

## Table of Contents

1. [Overview](#overview)  
2. [Key Features](#key-features)  
3. [Project Structure](#project-structure)  
4. [Setup & Installation](#setup--installation)  
5. [Configuration](#configuration)  
6. [Usage](#usage)  
   - [1. Collect and Store Emails](#1-collect-and-store-emails)  
   - [2. Train the Topic Model](#2-train-the-topic-model)  
   - [3. Create Inbox Folders](#3-create-inbox-folders)  
   - [4. Organize Emails](#4-organize-emails)  
7. [Technical Details](#technical-details)  
   - [Topic Modeling with BERTopic](#topic-modeling-with-bertopic)  
   - [Embedding Models](#embedding-models)  
   - [Microsoft Graph Integration](#microsoft-graph-integration)  
   - [Data Storage](#data-storage)  
8. [Credits & License](#credits--license)

---

## Overview

This project automates the process of **retrieving, classifying, and categorizing emails**. After emails are fetched from a mailbox, they are embedded and stored in MongoDB for analysis or future model training. A **BERTopic** model determines the most likely “topic” (e.g., Promotions, Orders, Finance, etc.), and the system then moves the email to a corresponding folder in Outlook.

### Workflow Summary

1. **Collect Emails**: Pull emails from Outlook/Office 365 via Microsoft Graph.  
2. **Store**: Embeddings and metadata are saved to a MongoDB collection.  
3. **Train/Re-train**: Use the stored data to train or refine a BERTopic model.  
4. **Create Folders**: Based on discovered or predefined topics, create matching folders in Outlook.  
5. **Organize**: Classify new emails and move them into the relevant folder automatically.

---

## Key Features

- **Automated Email Retrieval**: Uses Microsoft Graph to fetch emails.  
- **Topic Modeling**: Leverages [BERTopic](https://github.com/MaartenGr/BERTopic) for clustering/classification.  
- **Custom Embeddings**: Integrates with:
  - [Sentence Transformers](https://github.com/UKPLab/sentence-transformers)
  - [Voyage AI](https://voyage.ml/) client for custom embeddings
  - [LlamaCPP-based representation model](https://github.com/abetlen/llama-cpp-python) (optional)
- **MongoDB Integration**: Persists email data and embeddings for analysis or retraining.  
- **CLI Interface**: Built with [Typer](https://typer.tiangolo.com/) for easy command-line usage.

---

## Project Structure

Below is an overview of the key Python files:

```
.
├── app.py
├── src
│   ├── commands
│   │   ├── collect_and_store.py
│   │   ├── create_inbox_folders.py
│   │   ├── review_and_categorize.py
│   │   └── train_topic_model.py
│   ├── agent.py
│   ├── embedders.py
│   ├── message.py
│   ├── topic_labels.py
│   ├── topic_model_factory.py
│   └── util.py
├── xconfig.py (not shown, but contains environment variable logic)
└── ...
```

**Main Files & Their Roles**:

- **app.py**  
  A CLI entry-point (Typer application). Exposes commands:
  - `collect_and_store`
  - `create_inbox_folders`
  - `organize`
  - `train_topic_model`

- **src/agent.py**  
  The `EmailOrganizerAgent` class connects to Microsoft Graph, retrieves emails, preprocesses and encrypts them, and can move them between folders.

- **src/commands/**  
  - **collect_and_store.py**: Fetches emails from Outlook and stores them in MongoDB.  
  - **create_inbox_folders.py**: Creates Outlook inbox subfolders based on discovered topics.  
  - **review_and_categorize.py**: Pulls today’s (or unread) emails, classifies them, and moves them to the appropriate folder.  
  - **train_topic_model.py**: Trains or retrains the BERTopic model based on data in the MongoDB collection or local pickles.

- **src/topic_model_factory.py**  
  Builds a BERTopic instance with custom or default embeddings, UMAP/HDBSCAN parameters, and optional Llama-based representation.

- **src/embedders.py**  
  Defines custom embedding classes, e.g. `VoyageEmbedder`, suitable for plugging into BERTopic’s “backend” embeddings.

- **src/util.py**  
  General utility functions (async wrappers, HTML processing, chunking, etc.).

- **src/message.py**  
  A dataclass for representing messages in a standardized format.

- **src/topic_labels.py**  
  Example dictionaries for mapping numeric topic IDs to user-friendly strings.

---

## Setup & Installation

1. **Clone the repository**:
   ```bash
   git clone https://github.com/<your_org>/email-classification.git
   cd email-classification
   ```

2. **Create a virtual environment** (recommended):
   ```bash
   python -m venv venv
   source venv/bin/activate  # On macOS/Linux
   # or venv\Scripts\activate on Windows
   ```

3. **Install dependencies**:
   ```bash
   pip install -r requirements.txt
   ```

4. **Set up environment variables**:  
   This project uses `xconfig.py` (not shown here) to read your environment variables. Required variables might include:
   - `AZURE_TENANT_ID`
   - `AZURE_CLIENT_ID`
   - `AZURE_CLIENT_SECRET`
   - `AZURE_GRAPH_SCOPES`
   - `MODEL` (e.g., `"all-MiniLM-L6-v2"` for sentence-transformers)
   - `SPACY_LIBRARY` (e.g., `"en_core_web_sm"`)
   - `ENCRYPTION_KEY` (for encrypting email content)
   - `EMAIL_DATA_DATABASE` / `EMAIL_DATA_COLLECTION` / `EMAIL_RUN_LOG_COLLECTION`
   - `connection_string` for MongoDB
   - And any references for the `Voyage AI` client or Llama-based embeddings

   You can set these in an `.env` file or in your shell.

5. **Install spaCy model** (if needed), e.g.:
   ```bash
   python -m spacy download en_core_web_sm
   ```

---

## Configuration

The **topic model** is configured via a JSON file (default: `model_config.json`). Example fields could include:

```json
{
  "model_name": "models/my_bertopic_model",
  "voyage_model_name": "voyage-3",
  "representation_model": {
    "model_path": "path/to/llama2.bin",
    "n_gpu_layers": 32,
    "n_ctx": 4096,
    "stop": ["###"]
  },
  "umap_params": {
    "n_neighbors": 15,
    "n_components": 5,
    "min_dist": 0.0,
    "metric": "cosine"
  },
  "hdbscan_params": {
    "min_cluster_size": 10,
    "metric": "euclidean",
    "cluster_selection_method": "eom"
  },
  "nr_topics": null,
  "min_topic_size": 10,
  "zeroshot_topic_list": [],
  "training_limit": 5000
}
```

- **`model_name`**: Directory where the BERTopic model will be saved or loaded.  
- **`voyage_model_name`**: If using Voyage AI, references the model name to fetch embeddings.  
- **`representation_model`**: If using the Llama-based representation for topic labeling.  
- **`umap_params`, `hdbscan_params`**: Parameters for the underlying dimensionality reduction and clustering.  
- **`training_limit`**: Maximum emails to fetch from MongoDB for training.

---

## Usage

This project uses [Typer](https://typer.tiangolo.com/). You can run commands directly via `python app.py <command>`.

### 1. Collect and Store Emails

Fetches the user’s inbox emails from Outlook and stores them (with embeddings) in MongoDB:

```bash
python app.py collect-and-store --user-id "user@domain.com"
```

- **`--user-id`**: The mailbox user you want to retrieve emails from. (Default is set in the code.)

### 2. Train the Topic Model

Trains (or retrains) the BERTopic model from the emails that have been stored in MongoDB. You can optionally provide pickled documents/embeddings, or let the script pull from your DB:

```bash
python app.py train-topic-model \
  --config-path "model_config.json" \
  --retrain False \
  --docs-pickle-path "docs.pickle" \
  --embeddings-pickle-path "embeddings.pickle"
```

**Arguments**:
- `--config-path`: Path to your JSON configuration for the model.  
- `--retrain`: If `True`, loads an existing model from `model_name` and refits it.  
- `--docs-pickle-path`: Optional, if you have pre-stored documents as a pickle file.  
- `--embeddings-pickle-path`: Optional, if you have pre-stored embeddings as a pickle file.

### 3. Create Inbox Folders

Creates subfolders in your Outlook mailbox’s Inbox for each discovered topic in the loaded model:

```bash
python app.py create-inbox-folders \
  --user-id "user@domain.com" \
  --config-path "model_config.json"
```

### 4. Organize Emails

Fetches today’s unread emails, infers their topics, and **moves** them to the appropriate folder (unless `--dry-run` is used):

```bash
python app.py organize \
  --user-id "user@domain.com" \
  --config-path "model_config.json" \
  --dry-run False
```

- If `--dry-run` is `True`, it only **logs** what it would do, without moving emails.

---

## Technical Details

### Topic Modeling with BERTopic

The system uses **BERTopic** to cluster emails by subject/body text, then optionally label them with a zero-shot or Llama-based approach. The pipeline is:

1. Preprocessing via spaCy (removing stopwords, punctuation).  
2. Vectorizing either with a local **Sentence Transformer** or **Voyage AI** embedder.  
3. Dimensionality reduction via **UMAP** and clustering via **HDBSCAN**.  
4. Optional re-labeling or zero-shot classification.

### Embedding Models

- **Sentence Transformers**  
  Default approach using a model specified by `MODEL` environment variable.  
- **VoyageEmbedder** (in `embedders.py`)  
  Uses the Voyage AI client to embed text.  
- **LlamaCPP**  
  If specified, the model can also use Llama-based representation for interpretability or text-based topic labeling.

### Microsoft Graph Integration (Sorry GMail'ers)

`EmailOrganizerAgent` (in `agent.py`) handles:

- **Authentication**: Uses `ClientSecretCredential` from Azure Identity.  
- **Fetching**: Emails are retrieved via `GraphServiceClient`, optionally filtered (e.g., by `isRead == false`).  
- **Moving Emails**: The `.move()` call moves an email to another folder.  

### Data Storage

- **MongoDB**  
  - **Emails**: Embedded data, subject, body, etc., are encrypted and stored in a single collection.  
  - **Log**: The script can store “operation logs” (what got moved, etc.) in another collection.

---

## Credits & License

- [Typer](https://typer.tiangolo.com/) – Command-line interface.  
- [BERTopic](https://github.com/MaartenGr/BERTopic) – Topic modeling library.  
- [Voyage AI](https://voyage.ml/) – Optional embedding provider.  
- [Microsoft Graph Python SDK](https://github.com/microsoftgraph/msgraph-sdk-python) – For mailbox operations.  

You may distribute or modify under your chosen license. Please see `LICENSE` (if present) or contact the repository owner for details.

---  

**Enjoy automated email organization!** If you have any questions or encounter issues, feel free to open an issue or reach out.