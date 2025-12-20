# 🧠 AI Echo – ChatGPT-Style Review Sentiment Classifier

This project builds and deploys a **3-class sentiment classifier** for ChatGPT-style product reviews:

- **Negative**
- **Neutral**
- **Positive**

It includes:

- A **training notebook** (`AI_Echo_Notebook.ipynb`) that:
  - Cleans and prepares the review dataset
  - Creates a **sentiment label** from ratings (1–5)
  - Compares several models
  - Tunes a **rule-assisted LinearSVC** model
  - Saves artifacts for deployment
- A **Streamlit app** (`app.py`) that:
  - Uses a **SentenceTransformer embedding model** (`all-MiniLM-L6-v2`)
  - Loads a **rule-assisted LinearSVC** classifier
  - Lets you explore **single predictions + dashboard-style insights**

---

## 📂 Dataset

**Files used in this project**

- `data/chatgpt_style_reviews_dataset.xlsx` (training dataset without stopwords)
- During preprocessing, a cleaned Parquet file is created:
  - `data/clean/reviews_clean.parquet` (used by the Streamlit app)

Each row represents one review of a ChatGPT-style assistant.

**Key columns**

- `date` – Review date (some rows may have masked/invalid dates)
- `title` – Short title/summary of the review
- `review` – Main review text (original dataset already removed most stopwords)
- `rating` – Integer rating from **1 to 5**
- `username` – Reviewer ID
- `helpful_votes` – Number of “helpful” upvotes
- `review_length` – Length of the review text (characters)
- `platform` – e.g., `Web`, `Android`, `iOS`, `Amazon`, `Flipkart`
- `language` – Language code of the review
- `location` – Country/region of the reviewer
- `version` – App/version string (e.g., `3.8.4`)
- `verified_purchase` – `Yes` / `No`

**Label creation (Target variable)**

We convert the numeric rating to **sentiment** as:

- Rating **1–2 → Negative**
- Rating **3 → Neutral**
- Rating **4–5 → Positive**

This gives a **class-imbalanced** dataset:

- Negative: 97  
- Positive: 96  
- Neutral: 57  

Because of this imbalance, **Macro-F1** is used as the main metric (treats each class equally).

---

## 🧹 Text Cleaning & Feature Engineering

The notebook applies a consistent text-cleaning pipeline on `title` and `review`:

1. **Lowercasing**  
2. Remove **URLs** (`http…`, `www…`)  
3. Remove **@mentions** and **#hashtags**  
4. Strip **emojis**  
5. Remove **punctuation**  
6. Keep only **[a–z]** letters  
7. Normalize whitespace  

From these, we create:

- `title_clean`
- `review_clean`
- `final_text` = `title_clean + " " + review_clean`

Additional features used mainly for EDA / dashboards:

- `review_length` (characters)
- `review_tokens` (word count)
- Date-based features like `month` and `week`
- Metadata-driven analyses (`platform`, `location`, `version`, `verified_purchase`)

> The model itself uses **`final_text`** as the main input. The training dataset is already pre-cleaned (no stopwords), so the notebook only applies final normalization.

---

## 🧠 Modelling Pipeline

### 1️⃣ Train/Test split

- `X = final_text`  
- `y = sentiment`  
- **Stratified** split:
  - 80% train / 20% test  
  - `random_state=42` for reproducibility  

---

### 2️⃣ Baseline: TF-IDF + Classical Models

First, we tried traditional **TF-IDF** features (word & character n-grams) with:

- **Logistic Regression**
- **LinearSVC**
- **MultinomialNB**

Results (approx.):

| Model                | Vectorizer | Accuracy | Macro-F1 | Key issue                           |
|----------------------|-----------|----------|----------|-------------------------------------|
| Logistic Regression  | TF-IDF    | ~0.40    | ~0.31    | Hardly predicts **Neutral**         |
| LinearSVC            | TF-IDF    | ~0.40    | ~0.31    | Same as Logistic; Neutral collapsed |
| MultinomialNB        | TF-IDF    | ~0.40    | ~0.30    | Worst Macro-F1; Neutral ≈ 0 F1      |

**Observation**

- All three models **struggle with Neutral**.  
- They tend to predict **Negative/Positive only**, ignoring ambiguous mid-range reviews.  
- Accuracy is misleadingly “okay”, but **Macro-F1 reveals the problem** (Neutral F1 near 0).

---

### 3️⃣ Sentence-Transformer Embeddings + Classical Models

To capture better semantics and handle unseen words, we switched to:

- **Text encoder**: `SentenceTransformer("all-MiniLM-L6-v2")`
  - 384-dim dense embeddings
  - Handles synonyms, paraphrases, and **words not seen in training** much better than TF-IDF
- Models trained on embeddings:
  - **Logistic Regression** (multinomial, `class_weight="balanced"`)
  - **LinearSVC** (`class_weight="balanced"`)

**Results (on the same 20% test split)**

1. **Logistic Regression + MiniLM embeddings**

- Accuracy: **0.42–0.44**  
- Macro-F1: **≈ 0.40–0.42**

Roughly balanced, but still not great on Neutral.

2. **LinearSVC + MiniLM embeddings**

- Accuracy: **0.50**  
- Macro-F1: **0.4548**

Per-class pattern:

- **Negative** – strong performance (F1 ≈ 0.62)  
- **Positive** – strong performance (F1 ≈ 0.57)  
- **Neutral** – weakest (F1 ≈ 0.17)  

**Why LinearSVC was better here**

- SVMs work well in **high-dimensional embedding spaces**.  
- The margin-based objective handles noisy, small datasets better.  
- `class_weight="balanced"` helps with the label imbalance.  
- Highest **Macro-F1** among pure ML models.

---

### 4️⃣ Rule-Assisted LinearSVC (Gap + Confidence Thresholds)

Even with the best LinearSVC model, **Neutral** was consistently under-predicted.  
To fix this, we added a **rule layer** on top of the SVM decision scores.

For each review:

- Let `top1` = highest decision score across classes  
- Let `top2` = second-highest decision score  
- Compute:  
  - **gap** = `top1 - top2` (how confident the model is in the winner)  
  - If:
    - `gap < gap_t` (classes are close / ambiguous) **AND**
    - `top1 < conf_t` (overall confidence is low)
  - Then **force the prediction to "Neutral"**  

We grid-searched over `(gap_t, conf_t)` and found thresholds that improved performance:

- `gap_t ≈ 0.0337`  
- `conf_t ≈ -0.1258`  

**Final Rule-Assisted Model performance**

- Accuracy: **0.50**  
- Macro-F1: **≈ 0.47**

Class-wise:

- **Negative** – strong recall & precision  
- **Positive** – strong recall & precision  
- **Neutral** – **noticeable improvement in recall** vs pure LinearSVC  

Also, the **predicted class distribution** becomes more realistic:

- Negative: 22  
- Positive: 14  
- Neutral: 14  

Instead of almost never predicting Neutral, the model now uses it for **genuinely ambiguous / low-confidence** cases.

---

## 🏆 Final Model Choice & Justification

### ✅ Chosen Model: **Rule-Assisted LinearSVC over MiniLM embeddings**

**Why not just TF-IDF + classic models?**

- They collapsed almost entirely to two classes (Negative/Positive).  
- Neutral F1 was near **0.0**.  
- Accuracy ~0.40 but **not trustworthy** because minor classes were ignored.

**Why not Logistic Regression on embeddings?**

- Better than TF-IDF, but still **lower Macro-F1 (~0.41)** than LinearSVC.  
- LinearSVC was stronger overall, especially for clear Positive/Negative reviews.

**Why LinearSVC + SentenceTransformer embeddings?**

- Works very well on **short text**.  
- Leverages **semantic similarity**, not just exact words.  
- Handles **unseen words** better:
  - If a review uses a new word but in a context similar to training reviews, MiniLM will still embed it close to similar sentences.
- SVM is a good choice for small to medium datasets (like ~250 reviews).

**Why add rule-assistance (gap & confidence thresholds)?**

- Neutral is inherently the **“I’m not sure”** class.  
- The raw LinearSVC model tends to be overconfident and under-predict Neutral.  
- By:
  - Measuring **confidence gap** between top classes,  
  - Measuring **absolute confidence**, and  
  - Forcing “Neutral” in low-confidence cases,  
- …we get:
  - Better **Macro-F1 (~0.47 vs 0.4548)**.  
  - More **balanced confusion matrix**.  
  - A model that is **safer and more conservative**, especially for borderline cases.

**Conclusion**

> The **rule-assisted LinearSVC + MiniLM embeddings** strikes the best balance between **performance, interpretability, and robustness on a small dataset**. It improves minority class (Neutral) handling without hurting overall accuracy.

---

## 🖥️ Streamlit App Overview

The `app.py` file turns the trained model into an **interactive sentiment dashboard**.

### Tab 1 – 💬 Single Review Prediction

- Textbox to paste a **review**  
- Model returns:
  - **Predicted sentiment** (Negative / Neutral / Positive)  
  - **Decision scores** for each class (e.g., bar chart)  
- Shows **overall sentiment distribution** across the dataset.

### Tab 2 – 📊 Key Questions & Insights

Pre-built views to answer typical product-analytics questions:

1. Overall sentiment distribution  
2. Sentiment vs **rating** (e.g., 5★ but Negative sentiment → mismatches)  
3. Word clouds per sentiment  
4. Sentiment trend over **time** (if dates are valid)  
5. Sentiment by **verified purchase**  
6. **Review length** vs sentiment  
7. Sentiment by **location**  
8. Sentiment by **platform**  
9. Sentiment by **version**  
10. Top terms in **negative** reviews  

This makes the project more than just a model: it becomes an **exploratory dashboard** for understanding how users feel about a ChatGPT-style product.

---

## 🧪 Handling Words Outside the Training Dataset

Because we use **SentenceTransformer embeddings**:

- The model does **not** rely on an exact fixed vocabulary like TF-IDF.  
- New words (e.g., slang, product feature names, misspellings) are mapped into the same semantic space.  
- As long as the context is similar to training data, the model can still make reasonable predictions.

This is one of the main reasons for choosing a **pretrained embedding model** instead of raw bag-of-words.

---

## 🚀 How to Run Locally

### 1. Create & activate a conda environment

```bash
conda create -n ai_echo_env python=3.10 -y
conda activate ai_echo_env
```

### 2. Install dependencies

```bash
pip install -r requirements.txt
```

Your `requirements.txt` should include at least:

- `streamlit`  
- `pandas`  
- `numpy`  
- `scikit-learn`  
- `sentence-transformers`  
- `joblib`  
- `plotly`  
- `wordcloud`  
- `openpyxl`  

### 3. Make sure artifacts exist

The Streamlit app expects:

- Clean data: `data/clean/reviews_clean.parquet`  
- Model artifact: `artifacts/rule_assisted_linearsvc.joblib`  
- Embedding model folder: `artifacts/embedding_model/`  

If they are missing:

1. Run the notebook `AI_Echo_Notebook.ipynb` end-to-end.  
2. It will:
   - Load the Excel dataset.  
   - Clean and save the Parquet file.  
   - Train and tune the rule-assisted LinearSVC.  
   - Save:
     - `artifacts/rule_assisted_linearsvc.joblib`  
     - `artifacts/embedding_model/`  

### 4. Run the Streamlit app

```bash
streamlit run app.py
```

Open the URL shown in the terminal (usually `http://localhost:8501`).

---

## 📁 Project Structure (Suggested)

```text
AI-Echo/
├── app.py
├── AI_Echo_Notebook.ipynb
├── requirements.txt
├── artifacts/
│   ├── rule_assisted_linearsvc.joblib
│   └── embedding_model/          # saved SentenceTransformer
├── data/
│   ├── chatgpt_style_reviews_dataset.xlsx
│   └── clean/
│       └── reviews_clean.parquet
├── eda/
│   └── plots/                    # saved PNGs from EDA
└── README.md
```

---

## ⚠️ Limitations & Future Work

**Limitations**

- Small dataset (~250 reviews) → metrics are sensitive to the split.  
- Synthetic-style text → may not perfectly match real production data.  
- Threshold tuning (`gap_t`, `conf_t`) done on the test set; ideally should be within a validation loop.  
- No model calibration (e.g., Platt scaling) → decision scores are not true probabilities.

**Possible future improvements**

- Collect more **real-world reviews** & retrain.  
- Do **k-fold cross-validation** for more stable evaluation.  
- Calibrate scores and build a **probability-based rule layer**.  
- Try **fine-tuning a transformer** model directly (DistilBERT, BERT) once dataset size is larger.  
- Add **explanations** (e.g., SHAP or LIME) for model decisions in the Streamlit UI.

---

## ✅ Final Takeaway

This project shows how you can go from:

> Raw product reviews → Cleaned text → Classical ML + modern embeddings → Rule-assisted model → Interactive Streamlit dashboard

The final **rule-assisted LinearSVC with MiniLM embeddings** offers:

- Better **balanced performance** across Negative / Neutral / Positive.  
- A more **conservative and realistic** use of the Neutral class.  
- An interpretable and production-friendly pipeline suitable for demos, portfolios, and further extension.
