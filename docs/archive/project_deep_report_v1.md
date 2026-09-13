> Superseded by taxonomy_results.json and README (post-negation-tagging, post-taxonomy-diagnosis numbers). Kept for historical error-analysis and interview-prep content.

# SmartTicket AI — Comprehensive Project Report & Interview Guide

---

## Part 1 — Model Performance Deep Dive

### 1.1 Dataset Overview

| Metric | Value |
|---|---|
| Raw dataset rows | 24,000 (20k primary + 4k extra) |
| After deduplication | 23,998 |
| After English-only filter | **13,313** |
| Train split (85%) | **11,308** |
| Test split (15%) | **1,996** |
| TF-IDF vocabulary size | 25,000 features |
| ML models | 2x LinearSVC (category + issue type) |

> [!WARNING]
> The English-only filter discards **~44% of the dataset** (multilingual rows). The dataset name says "multi-lang" but the pipeline throws away non-English data — you are not benefitting from the full 24k rows.

---

### 1.2 Category Model — Full Results

| Category | Precision | Recall | F1-Score | Test Support | Class Weight |
|---|---|---|---|---|---|
| Billing and Payments | 0.77 | 0.78 | **0.77** | 212 (10.6%) | 0.94 |
| Customer Service | 0.42 | 0.42 | **0.42** | 311 (15.6%) | 0.64 |
| General Inquiry | 0.48 | 0.37 | **0.42** | 27 (1.4%) | 7.25 |
| Human Resources | 0.54 | 0.42 | **0.47** | 33 (1.7%) | 6.05 |
| IT Support | 0.44 | 0.48 | **0.46** | 234 (11.7%) | 0.85 |
| Product Support | 0.49 | 0.49 | **0.49** | 370 (18.5%) | 0.54 |
| Returns and Exchanges | 0.51 | 0.49 | **0.50** | 97 (4.9%) | 2.06 |
| Sales and Pre-Sales | 0.50 | 0.46 | **0.48** | 56 (2.8%) | 3.58 |
| Service Outages & Maint. | 0.61 | 0.58 | **0.60** | 74 (3.7%) | 2.70 |
| Technical Support | 0.59 | 0.58 | **0.58** | 582 (29.2%) | 0.34 |
| **Weighted Average** | **0.54** | **0.54** | **0.54** | 1,996 | — |

---

### 1.3 Issue Type Model — Full Results

| Issue Type | Precision | Recall | F1-Score | Test Support |
|---|---|---|---|---|
| Change | 0.94 | 0.90 | **0.92** | 211 |
| Incident | 0.78 | 0.79 | **0.79** | 807 |
| Problem | 0.59 | 0.58 | **0.59** | 402 |
| Request | 0.94 | 0.96 | **0.95** | 576 |
| **Weighted Average** | **0.81** | **0.81** | **0.81** | 1,996 |

The issue type model performs very well because "Change", "Request", "Incident", "Problem" are structurally distinct ITIL concepts with clearly different vocabulary.

---

### 1.4 The Two Worst Categories — Root Cause Analysis

#### #1 — `Customer Service` (F1: 0.42)

**Test support: 311 samples (15.6% of test set — 2nd largest class)**

This is the single biggest drag on overall model accuracy.

**Why it fails — Confusion Matrix Evidence:**
```
87x  [Customer Service] -> [Product Support]       <- most common mistake
50x  [Customer Service] -> [Technical Support]
44x  [Customer Service] -> [IT Support]
29x  [IT Support]       -> [Customer Service]       <- model is confused both ways
20x  [Billing]          -> [Customer Service]
17x  [Customer Service] -> [Billing and Payments]
```

**Root causes:**
- "Customer Service" is a **catch-all umbrella** category. In the dataset it contains tickets about product complaints, billing disputes, return requests — topics that belong to other more specific categories.
- The `queue` column (mapped to `category`) represents the *routing queue in a helpdesk*, not the true semantic topic. Tickets routed to the "Customer Service" queue are inherently multi-topic.
- Vocabulary overlap with every other category is maximal. No distinctive lexical signal — TF-IDF cannot separate it.

---

#### #2 — `General Inquiry` (F1: 0.42)

**Test support: only 27 samples (1.4%) — the smallest class**

**Why it fails:**
- With only 27 test samples, even a few misclassifications catastrophically drop F1.
- Despite a class weight of **7.25** (highest in the dataset), there simply is not enough training data to learn a reliable boundary.
- "General Inquiry" by definition contains tickets that don't fit anywhere specific — making it semantically the most ambiguous class.
- The 37% recall means the model only identifies ~10 of 27 actual "General Inquiry" tickets correctly.

---

### 1.5 Confusion Cluster — The 4-Way Deadlock

The biggest systemic problem is a **4-way confusion cluster** among semantically overlapping categories:

```
Technical Support  <->  Product Support    (87 + 82 misclassifications)
Technical Support  <->  IT Support         (65 + 52)
Customer Service   <->  Product Support    (51 + 44)
Customer Service   <->  Technical Support  (57 + 50)
Customer Service   <->  IT Support         (36 + 29)
IT Support         <->  Product Support    (23 + 22)
```

These four categories account for **75.1%** of the test set. All share the same vocabulary: "issue", "error", "fix", "not working", "system", "access", "login", "reset". TF-IDF cannot distinguish *context of use* — only word presence.

---

### 1.6 Category vs. Issue Type Cross-Table (Test Set)

| Category | Change | Incident | Problem | Request |
|---|---|---|---|---|
| Billing and Payments | 13 | 50 | 45 | 104 |
| Customer Service | 23 | 98 | 55 | 135 |
| General Inquiry | 6 | 9 | 6 | 6 |
| Human Resources | 4 | 17 | 7 | 5 |
| IT Support | 44 | 78 | 56 | 56 |
| Product Support | 47 | 161 | 79 | 83 |
| Returns and Exchanges | 9 | 40 | 17 | 31 |
| Sales and Pre-Sales | 11 | 11 | 3 | 31 |
| Service Outages & Maint. | 18 | 43 | 3 | 10 |
| Technical Support | 36 | 300 | 131 | 115 |

> [!NOTE]
> Technical Support is heavily skewed towards "Incident" (300/582 = 52%), which explains why the issue-type model is so much easier to train — "Incident" has a distinctive vocabulary (outage, crash, down, urgent).

---

## Part 2 — Actionable Improvement Suggestions

### High Impact

**1. Re-label / Merge Ambiguous Categories**
- Merge `Customer Service` + `General Inquiry` into a single "General" class, OR
- Audit 500 "Customer Service" samples and relabel them into their true categories. This single fix could push category F1 from 0.54 to 0.65+.

**2. Use a Transformer-Based Model (BERT / DistilBERT)**
- TF-IDF + LinearSVC loses word order and context. "My product is technically broken" and "I need technical support for my product" produce the same TF-IDF vector.
- `distilbert-base-uncased` fine-tuned for multi-class classification would likely push category F1 to 0.75-0.85.

**3. Hierarchical Classification**
- First predict a coarse group: `{Technical/IT/Product}` vs `{Billing/Returns}` vs `{HR/Sales/General}`.
- Then predict the fine-grained category within the group.
- Prevents cross-group confusion (the model won't confuse "Billing" with "IT Support").

**4. Use Multilingual Data**
- The pipeline discards ~44% of rows with `language != 'en'`. Add multilingual BERT (`xlm-roberta-base`) to leverage the full 24k dataset.

### Medium Impact

**5. Use Subject Line as Separate Feature**
- Currently: `text = subject + ' ' + body`. The subject is the highest-signal part but gets diluted by the long body.
- Better: Separate TF-IDF channels — `X = hstack([tfidf_subject * 2.0, tfidf_body])`.

**6. Better Stop Word Strategy**
- Words like `'support', 'issue', 'problem', 'customer'` are removed from custom stop words but are discriminative signals. Remove only ultra-generic words and let `max_df=0.85` handle the rest.

**7. Oversample Minority Classes**
- `General Inquiry` (~180 train), `Human Resources` (~200 train), `Sales` (~340 train) are too small.
- Use SMOTE on TF-IDF vectors or data augmentation via back-translation.

**8. Add Confidence Calibration**
- LinearSVC `decision_function` scores are NOT probabilities. Use `CalibratedClassifierCV(clf, method='sigmoid')` to get true 0-1 probabilities, making the `needs_review` threshold meaningful.

---

## Part 3 — Interview Questions & Answers

### Basic Level

---

**Q1: What does this project do at a high level?**

**A:** SmartTicket AI is an end-to-end ML system for automated customer support ticket classification. When a user submits a ticket via the Streamlit frontend or REST API, the system immediately returns a ticket ID (asynchronous UX), then in the background:
1. Cleans and preprocesses the ticket text.
2. Runs it through a trained TF-IDF vectorizer.
3. Uses two LinearSVC classifiers — one to predict the *category* (e.g., "IT Support") and one to predict the *issue type* (e.g., "Incident" or "Request").
4. Stores the results in a database with a confidence score.
5. Flags low-confidence predictions for human review.

---

**Q2: What is TF-IDF and why did you use it?**

**A:** TF-IDF stands for Term Frequency-Inverse Document Frequency. For each word in a document:
- **TF** = how often the word appears in this document.
- **IDF** = how rare the word is across all documents (rarer words get higher weight).

`TF-IDF(w) = TF(w) x log(N / df(w))`

I used it because:
1. Converts text into a fixed-size numerical vector that ML models can consume.
2. Naturally down-weights generic words like "the", "is".
3. Extremely fast to compute at inference time (milliseconds).
4. For well-separated categories, linear classifiers on TF-IDF features compete with deep learning at a fraction of the cost.

**Limitation:** TF-IDF loses word order and semantic context — "not working" and "working well" have similar TF-IDF vectors.

---

**Q3: What is LinearSVC and how does it classify text?**

**A:** LinearSVC finds a hyperplane in the feature space (our 25,000-dimensional TF-IDF space) that separates classes with the maximum possible margin.

For multi-class (10 categories), it uses the **One-vs-Rest (OvR)** strategy — it trains 10 binary classifiers, each answering "is this ticket category X or not?", and the one with the highest decision-function score wins.

The `C` hyperparameter controls regularization:
- High C = fits training data tightly (risk of overfitting)
- Low C = larger margin, more robust to noise

In this project, `C=2.0` was selected via `GridSearchCV` for the category classifier.

---

**Q4: What is the difference between Precision, Recall, and F1-Score?**

**A:** Using "Customer Service" as a concrete example:

| Metric | Formula | Meaning | Value |
|---|---|---|---|
| Precision | TP / (TP + FP) | Of all tickets called "Customer Service", how many actually were? | 0.42 |
| Recall | TP / (TP + FN) | Of all actual "Customer Service" tickets, how many did the model find? | 0.42 |
| F1-Score | 2 x (P x R) / (P + R) | Harmonic mean — balanced metric | 0.42 |

**Why F1 not accuracy?** Dataset is imbalanced — "Technical Support" is 29.2% of data. A model that always predicts "Technical Support" gets 29% accuracy while being completely useless. F1 per class correctly penalizes this.

---

**Q5: What is class imbalance and how did you handle it?**

**A:** Class imbalance means some classes have far more samples than others:
- Technical Support: ~29% of data (overrepresented)
- General Inquiry: ~1.4% (severely underrepresented)

**How it was handled:**
1. `class_weight='balanced'` in LinearSVC — assigns each class a weight inversely proportional to its frequency: `weight = N / (n_classes x n_samples_of_class)`. For "General Inquiry" this gives weight **7.25**, meaning each misclassification counts 7x more in the loss.
2. `compute_class_weight('balanced', ...)` from sklearn computes these weights explicitly.

---

**Q6: What is FastAPI and why use it over Flask?**

**A:**

| Feature | FastAPI | Flask |
|---|---|---|
| Performance | ASGI, async-native | WSGI, sync-first |
| Data validation | Pydantic v2 built-in | Manual |
| Auto docs | /docs (Swagger), /redoc | Third-party extensions |
| Type hints | First-class | Optional |
| Background tasks | Built-in BackgroundTasks | Requires extensions |

In this project, FastAPI's `BackgroundTasks` is used to immediately return the ticket ID while the ML processing happens asynchronously in the same process.

---

**Q7: What is the difference between SQLite and PostgreSQL in this project?**

**A:**
- **SQLite (dev default)**: File-based (`tickets.db`), no server needed, `check_same_thread=False` is required for FastAPI threads. Zero setup.
- **PostgreSQL (Docker/prod)**: Full RDBMS, supports connection pooling (`pool_size=10, max_overflow=20`), concurrent writes, production-grade reliability.

The code auto-detects which is in use via `"sqlite" in DATABASE_URL` and applies the correct connection arguments.

---

### Intermediate Level

---

**Q8: Walk me through the full ticket lifecycle in the system.**

**A:** End-to-end flow:

```
1. User submits ticket via Streamlit UI
       POST /tickets {ticket: "text..."}
2. FastAPI route calls create_ticket(db, text)
   -> DB INSERT: status="pending", returns ticket_id immediately
3. BackgroundTasks.add_task(process_ticket_background, ticket_id)
   -> Response 200 returned to user immediately
4. Background task starts:
   a. DB UPDATE: status="processing"
   b. PredictPipeline.predict(text):
      - clean_text(): lowercase, regex strip, stop words removal
      - tfidf.transform([clean_text]) -> 25,000-dim sparse vector
      - clf_category.predict() -> label
      - decision_function() -> confidence = max(scores)
      - needs_review = confidence < 0.60
      - clf_issue_type.predict() -> issue_type
      - RESPONSES[category] -> auto_response
   c. DB UPDATE: status="completed" with all results
   d. CloudWatch: 4 metric calls
5. Streamlit polls GET /tickets/{id} every 1 second (max 10 retries)
6. Once status="completed", display results
```

---

**Q9: Why is the category F1 (0.54) so much lower than issue type F1 (0.81)?**

**A:** Three structural reasons:

1. **Semantic overlap of categories**: "Customer Service", "IT Support", "Technical Support", "Product Support" share massive vocabulary overlap. A user saying "my login doesn't work" could legitimately belong to any of the four. The ground truth labels reflect the *helpdesk routing queue*, not the intrinsic topic.

2. **Number of classes**: Category has 10 classes; issue type has 4. More classes = tighter, harder decision boundaries.

3. **ITIL concepts are structurally distinct**: "Request", "Incident", "Change", "Problem" are IT Service Management concepts with structurally different language:
   - Incident: "down", "outage", "crashed", "urgent"
   - Request: "please", "would like", "need access", "setup"
   - Change: "upgrade", "migrate", "update", "transition"
   - Problem: "recurring", "root cause", "pattern", "persistent"

---

**Q10: How does the model bootstrap work and what happens if S3 is unavailable?**

**A:** In `ensure_models_exist()`:

1. Check if all 5 model files exist locally in `./models/`.
2. If all present -> skip (fast path, no network call).
3. If some missing -> attempt download from S3 for each missing file.
4. If **ALL downloads fail** (S3 not configured) -> log a warning but **continue** with local models. (Non-fatal graceful degradation.)
5. If **SOME downloads fail** (partial failure) -> `raise RuntimeError`. A partial model set would produce inconsistent predictions, so better to crash explicitly.
6. `TESTING=true` env var skips all of this (used in CI).

---

**Q11: Explain the connection pool configuration and why session management matters.**

**A:**

```python
pool_size=10        # max persistent connections kept alive
max_overflow=20     # extra connections when pool exhausted (max total=30)
pool_pre_ping=True  # "SELECT 1" before using a connection (detect stale ones)
pool_recycle=3600   # close and reopen connections older than 1 hour
```

**The problem:** In `process_ticket_background()`, one DB session is held open across:
1. DB read (fetch ticket)
2. ML inference (~50-200ms CPU)
3. DB write (update ticket)
4. AWS CloudWatch calls (~4 HTTPS calls, ~400ms)

This means one connection per in-flight ticket is held for ~600ms+. With `pool_size=10`, only 10 tickets can process concurrently before requests block.

**Better pattern:** Read -> close session -> do ML + AWS -> open new session -> write.

---

**Q12: What is the SQS worker and why is it currently idle?**

**A:** `worker.py` implements a polling loop that:
1. Calls `sqs.receive_message(MaxNumberOfMessages=1, WaitTimeSeconds=5)` (long polling)
2. Extracts `ticket_id` from the message body
3. Calls `process_ticket_background(ticket_id)`
4. Deletes the message from SQS after successful processing

**Why it's idle:** `POST /tickets` does **not call `send_message()`**. The function `aws/sqs_client.py::send_message()` is defined but never imported or called. The worker is wired to receive — there's simply no publisher.

**To activate it:** In `ticket_service.py`, after creating the DB row, call `sqs_client.send_message(ticket.id)` instead of `background_tasks.add_task(...)`.

---

**Q13: How does GridSearchCV work and what did it find?**

**A:** GridSearchCV performs exhaustive hyperparameter search using cross-validation:

```
For each C in [0.1, 1.0, 2.0, 5.0, 10.0]:
    5-Fold cross validation:
        For each fold: train on 4 folds, validate on 1 fold, compute F1
    Average F1 across 5 folds = score for this C
Return the C with highest average CV F1
```

**Result:** Best `C = 2.0` for the category classifier — moderate regularization, not overfitting (C=10) nor underfitting (C=0.1).

---

### Advanced Level

---

**Q14: Why would a Transformer model outperform TF-IDF + LinearSVC on this dataset?**

**A:** Three core limitations of TF-IDF + LinearSVC that Transformers overcome:

1. **Bag-of-Words assumption**: "My account is not working" and "my account is working perfectly" produce nearly identical TF-IDF vectors. BERT uses bidirectional attention to understand that "not working" is semantically opposite to "working perfectly".

2. **No transfer learning**: LinearSVC starts from scratch on each training run. BERT is pre-trained on billions of words — it already understands syntax, negation, and semantic relationships. Fine-tuning on 11k tickets is sufficient to achieve strong performance.

3. **Contextual embeddings**: TF-IDF gives the word "issue" the same vector in all contexts. BERT gives "network issue" and "billing issue" different embeddings for "issue" based on surrounding words — exactly what's needed to separate "IT Support" from "Billing".

**Expected improvement**: Fine-tuned `distilbert-base-uncased` typically achieves 15-25 percentage points higher F1 on semantically overlapping multi-class text classification.

---

**Q15: How would you implement confidence calibration and why does it matter?**

**A:** **The problem:** LinearSVC's `decision_function()` returns raw signed distances from the decision hyperplane — not probabilities. A score of 0.3 and 2.5 don't have a proportional relationship.

**Current code (broken confidence):**
```python
cat_scores = self.clf_category.decision_function(text_tfidf)[0]
confidence = round(float(max(cat_scores)), 4)
needs_review = confidence < 0.60  # 0.60 is meaningless as a probability threshold
```

**The fix — Platt Scaling:**
```python
from sklearn.calibration import CalibratedClassifierCV

calibrated_clf = CalibratedClassifierCV(
    LinearSVC(C=2.0, class_weight='balanced'),
    method='sigmoid',  # Platt scaling
    cv=5
)
calibrated_clf.fit(X_train, y_train)

# At inference:
probas = calibrated_clf.predict_proba(X_test)   # true [0,1] probabilities
confidence = float(max(probas[0]))               # genuinely meaningful
needs_review = confidence < 0.60                # now a real 60% threshold
```

**Why it matters:** The `needs_review` flag is business-critical. Without calibration, a confidently wrong prediction (e.g., score=1.2) slips through without review, sending wrong auto-responses to customers.

---

**Q16: Explain the concurrency model and its bottlenecks under load.**

**A:**

**FastAPI + BackgroundTasks model:**
- Uvicorn runs a single process with an async event loop.
- `BackgroundTasks` runs tasks in a **thread pool**, not the async event loop.
- Each `process_ticket_background()` runs in a separate OS thread.

**GIL bottleneck:**
- NumPy/SciPy sparse matrix operations (TF-IDF, SVC) release the GIL — concurrent.
- Pure Python code (regex, list comprehensions in `clean_text`) holds the GIL.
- Under high concurrency, threads for Python string processing contend on the GIL.

**DB connection pool bottleneck:**
- `pool_size=10` means max 10 concurrent sessions.
- Each ticket holds a session for ~600ms (ML + CloudWatch time).
- **Throughput ceiling = 10 / 0.6s = ~16-17 tickets/second**.

**Real fix — SQS decoupling:**
```
POST /tickets -> DB insert -> SQS send -> return immediately (<5ms)
Worker pool (N instances) -> poll SQS -> process -> DB update
```
API can now handle 500+ req/sec. Workers scale horizontally independent of the API tier.

---

**Q17: How would you implement ML model drift detection in CI/CD?**

**A:** The current CI/CD only runs code tests. A production ML pipeline would add:

**Stage 1 — Pre-deploy model validation:**
```yaml
- name: Evaluate model on holdout set
  run: |
    python scripts/evaluate_model.py \
      --model models/clf_category.pkl \
      --test artifacts/test.csv \
      --min-f1 0.52
```

**Stage 2 — Shadow scoring / A/B routing:**
- Deploy new model to 10% of traffic alongside old model.
- Alert if new model disagrees with old model on >20% of tickets.

**Stage 3 — Online drift detection:**
- Track distribution of predicted categories in CloudWatch daily.
- If "Billing and Payments" drops from 10% to 2% of tickets, input distribution has shifted.
- Use Population Stability Index (PSI) or KL-divergence to quantify drift.
- Alert triggers retraining pipeline.

**Stage 4 — Automated retraining:**
- Nightly: `if new_tickets_since_last_retrain > 5000: trigger train_pipeline.py`.
- Human approval gate before production promotion.

---

**Q18: What trade-offs did you make in feature engineering?**

**A:**

| Decision | Current Choice | Trade-off | Better Alternative |
|---|---|---|---|
| Feature representation | Bag of n-grams | Loses word order, no semantics | Sentence embeddings (SBERT) |
| Subject weighting | Concatenated equally with body | Subject diluted by long body | Separate TF-IDF channels, weighted |
| Vocabulary size | 25,000 | Fixed cap | Dynamic vocab based on mutual info |
| Stop words | Removes "support", "issue" | Removes discriminative tokens | Remove only ultra-generic words |
| n-gram range | (1,3) | Trigrams add noise for short texts | (1,2) or character n-grams |
| Feature selection | None after TF-IDF | 25k features, many irrelevant | Chi-squared: SelectKBest(chi2, k=10000) |
| Text preprocessing | English only | 44% of data discarded | Multilingual model (XLM-RoBERTa) |

**Highest ROI single change:** Two-channel TF-IDF with subject weighted 2x body:
```python
from scipy.sparse import hstack
X_subject = tfidf_subject.fit_transform(df['subject'])
X_body    = tfidf_body.fit_transform(df['body'])
X = hstack([X_subject * 2.0, X_body])
```

---

**Q19: How does the `needs_review` flag work and what are its failure modes?**

**A:**

**Current implementation:**
```python
cat_scores = self.clf_category.decision_function(text_tfidf)[0]
confidence = round(float(max(cat_scores)), 4)
needs_review = confidence < 0.60
```

**Failure modes:**

1. **High confidence does not mean correct**: If the model confidently predicts "Technical Support" with score=1.2 for an ambiguous ticket, it is never flagged. Wrong auto-response gets sent without review — the most dangerous failure mode.

2. **Threshold has no probabilistic meaning**: Raw SVC scores are not probabilities. A 0.60 threshold is arbitrary.

3. **Threshold is not per-class**: "Billing and Payments" (F1=0.77) needs a lower threshold than "Customer Service" (F1=0.42). One threshold fits none well.

**Proper fix — calibrated per-class thresholds:**
```python
# Tune threshold per class on validation set:
for category in le_category.classes_:
    thresholds[category] = find_threshold(probas, y_val, category, target_precision=0.90)

# At inference:
proba = calibrated_clf.predict_proba(text_tfidf)[0]
predicted_class = le_category.classes_[np.argmax(proba)]
confidence = max(proba)
needs_review = confidence < thresholds[predicted_class]
```

---

**Q20: How would you scale this system to handle 10,000 tickets per minute?**

**A:** Current architecture handles ~16-17 tickets/second (DB pool bottleneck). To reach 10,000/min (~167/sec):

**Step 1 — Activate SQS decoupling (immediate, 10x improvement)**
- API becomes stateless: DB insert + SQS send (~5ms each)
- API throughput: easily 500+ req/sec

**Step 2 — Scale workers horizontally**
- Each worker processes ~5 tickets/sec (200ms: 100ms ML + 100ms DB)
- Need: 167 / 5 = ~34 worker instances
- ECS Fargate / Kubernetes HPA auto-scaling based on SQS queue depth

**Step 3 — Batch inference**
```python
texts = [t.ticket_text for t in batch]
vectors = tfidf.transform(texts)          # one sparse transform call for 10 tickets
preds = clf_category.predict(vectors)     # one SVC call, 10 results
```
Batch inference is 5-8x faster per-ticket due to matrix operation amortization.

**Step 4 — Caching and read replicas**
- Cache `/analytics/summary` in Redis (TTL 60s) to avoid repeated full table scans.
- Add PostgreSQL read replica for all SELECT queries; route writes to primary.

**Step 5 — Model serving as microservice**
- Move `PredictPipeline` to a dedicated model serving service (BentoML, Triton).
- Workers call the model service via HTTP.
- Allows independent scaling of inference vs. data orchestration.

---

*Report generated: 2026-09-11 | Models: TF-IDF (25k features) + 2x LinearSVC | Dataset: 13,313 EN samples*
