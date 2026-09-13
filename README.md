# 🎫 SmartTicket AI: ML-Powered Support Classifier

SmartTicket AI is an end-to-end asynchronous machine learning system that classifies customer support tickets. It leverages a modern cloud-native stack to handle high-volume ticket processing without blocking the user interface.

---

## 🏗️ System Architecture

The system is designed for high availability and decoupling:

* **Frontend:** Streamlit Dashboard for ticket submission and history tracking.
* **Backend API:** FastAPI for ticket ingestion and database management.
* **Messaging:** AWS SQS for reliable, asynchronous task queuing.
* **Storage:** AWS S3 for hosting large Model artifacts (`.pkl` files).
* **ML Engine:** LinearSVC classifiers for multi-output (Category & Issue Type) prediction.

---

## 🛠️ Setup & Installation

### 1. Environment Configuration

```bash
# Clone the repository
git clone [https://github.com/muralikrishna1729/support-ticket-ai.git](https://github.com/muralikrishna1729/support-ticket-ai.git)
cd support-ticket-ai

# Create and activate virtual environment
python -m venv venv
source venv/bin/activate  # Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt
```

## 2. AWS Credentials (.env)
Create a .env file in the root directory and add your credentials:

```
AWS_ACCESS_KEY_ID=your_access_key
AWS_SECRET_ACCESS_KEY=your_secret_key
AWS_REGION=ap-south-1
SQS_QUEUE_URL=[https://sqs.ap-south-1.amazonaws.com/your_account_id/smartticket-queue](https://sqs.ap-south-1.amazonaws.com/your_account_id/smartticket-queue)
S3_BUCKET_NAME=your_bucket_name
```

## 🚀 Execution Guide
Follow these steps in order to run the full pipeline:

### Step 1: Model Training
Processes raw data, fits the TF-IDF vectorizer, and trains the classifiers.

```
python -m src.pipeline.train_pipeline --source_path notebook/data/tickets-dataset.csv
```

### Step 2: S3 Model Synchronization
Uploads the locally trained .pkl files to your S3 bucket for production use.

```
python scripts/upload_model.py
```

### Step 3: Launch FastAPI Backend
Starts the REST API to handle incoming tickets and database operations.

```
uvicorn main:app --reload
```

### Step 4: Run the ML Worker
Starts the background process that polls SQS and classifies tickets.

```
python worker.py
```

### Step 5: Start Streamlit App
Launch the web interface to interact with the system.

```
streamlit run streamlit_app.py
```

## 🔌 API Endpoints

| Method | Endpoint | Description |
| :--- | :--- | :--- |
| **POST** | `/tickets` | Submits a new ticket for classification. |
| **GET** | `/tickets/{id}` | Polls for the status and result of a specific ticket. |
| **GET** | `/history` | Retrieves a list of all processed tickets. |

---

## 📊 Performance Metrics

The models were evaluated using weighted F1-scores to account for class imbalance
(source of truth: `models/model_scores.json`, current version `v3.0-negation-tfidf`):

* **Category Classification:** Negation-tagged TF-IDF + LinearSVC (GridSearch) ~ **0.53**
* **Issue Type Prediction:** LinearSVC ~ **0.81**

**Model selection:** feature-extraction techniques were benchmarked on the same
train/test split with `benchmark_techniques.py` (results in
`experiment_results.json`): plain TF-IDF (0.529), frozen sentence embeddings
`all-MiniLM-L6-v2` (0.340), TF-IDF ⊕ embeddings concatenation (0.364) and
**negation-tagged TF-IDF (0.532 — winner)**. Generic embeddings lost badly on
this domain-specific vocabulary, so the pipeline stays pure scikit-learn;
negation tagging gives "cannot access" (`not NEG_access`) a distinct token
signature from "can access" (`access`). Low-confidence predictions (max
`decision_function` score < 0.60) are flagged `needs_review` for human triage.

**Taxonomy diagnosis** (`diagnose_taxonomy.py`, full numbers in
`taxonomy_results.json`):

* Near-duplicate *templates* (TF-IDF cosine ≥ 0.7, 17% of the corpus) are
  **99.6% label-consistent** — the label noise is not copy-paste relabeling.
* At topic level (similarity ≥ 0.5 transitive groups covering 43% of the
  corpus), **8.2% of tickets carry conflicting labels** (7.3% in heavily
  inconsistent groups) — the noise is paraphrase-level boundary overlap.
* The production model's confusion is concentrated in one four-way cluster:
  Technical / Product / IT / Customer Support bleed 6–24% into each other
  pairwise (e.g. Product → Technical 21.9%, IT → Technical 23.9%).
* Merging those four into one category lifts **macro F1 0.5207 → 0.6116**
  (retrained on 7 labels; 0.5979 if the same 10-cat model's predictions are
  merged post-hoc). This quantifies how much of the error ceiling is
  taxonomy, not model. The merged bucket is 75% of the corpus — too coarse
  for real team routing — so the 10-category model ships with the confusion
  cluster documented and `needs_review` prioritized at that boundary.
* A stratified 249-row hand-labeling subset
  (`make_validation_subset.py` → `artifacts/validation_subset.csv`) is
  prepared to separate model error from label error on held-out data.

---

## 📁 Project Structure

```plaintext
├── artifacts/           # Processed CSV data (Git Ignored)
├── aws/                 # S3 & SQS Client configurations
├── models/              # Saved .pkl artifacts (Git Ignored)
├── src/
│   ├── components/      # Ingestion, Transformation, Trainer
│   ├── db/              # SQLAlchemy models & DB connection
│   ├── pipeline/        # Train & Predict logic
│   └── services/        # Business logic for background tasks
├── streamlit_app.py     # UI Dashboard (Streamlit)
├── worker.py            # SQS Listener/Processor
├── main.py              # FastAPI Entry Point
└── requirements.txt     # Project Dependencies
```
## 📝 Key Features

* **Asynchronous Processing:** Users receive a "Ticket ID" immediately while the ML engine works in the background.
* **Smart Auto-Response:** Generates tailored responses (e.g., "Technical Support" tickets trigger faster response expectations).
* **Class Balance Handling:** Uses `compute_class_weight` during training to ensure rare ticket types are classified correctly.
* **Decoupled Architecture:** The worker can be scaled horizontally to handle higher SQS loads.
* **Cloud Integrated:** Uses `Boto3` for seamless communication with AWS S3 and SQS.
