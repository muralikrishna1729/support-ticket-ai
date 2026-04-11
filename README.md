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

The models were evaluated using weighted F1-scores to account for class imbalance:

* **Category Classification:** LinearSVC (GridSearch) ~ **0.68**
* **Issue Type Prediction:** LinearSVC ~ **0.89**

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
