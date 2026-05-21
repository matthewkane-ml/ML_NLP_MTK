# NLP Spam Classifier

> A URL spam detector built with TF-IDF vectorization and a tuned Support Vector Machine — classifies whether a URL is spam or legitimate.

---

## Problem

Spam URLs are one of the most common vectors for phishing, malware, and unwanted content. Manually reviewing URLs at scale is impractical, so this project builds an NLP-based binary classifier that analyzes the textual content of a URL and predicts whether it is spam. The pipeline covers the full NLP workflow: text cleaning, lemmatization, vectorization, and model tuning.

## Dataset

- **Source:** [4GeeksAcademy NLP Project Tutorial](https://raw.githubusercontent.com/4GeeksAcademy/NLP-project-tutorial/main/url_spam.csv)
- **Size:** Publicly available CSV of labeled URLs
- **Target:** `is_spam` (0 = Legitimate, 1 = Spam)
- **Key features:** Raw URL text, preprocessed and vectorized using TF-IDF (top 5,000 features)

## Approach

1. **Text cleaning:** Stripped non-alphabetic characters, removed single-character tokens, collapsed whitespace, and lowercased all text using regex.
2. **Lemmatization & stop word removal:** Applied NLTK's `WordNetLemmatizer`, removed English stop words, and dropped tokens shorter than 3 characters.
3. **Vectorization:** Applied `TfidfVectorizer` with `max_features=5000`, `max_df=0.8`, and `min_df=5` to convert the cleaned URL tokens into a sparse feature matrix.
4. **Baseline model:** Trained an initial `SVC` with a linear kernel to establish a performance baseline.
5. **Hyperparameter tuning:** Ran `GridSearchCV` (5-fold CV) over kernel type, `C`, `degree`, and `gamma`. Best configuration: `kernel=poly`, `C=1000`, `degree=1`, `gamma=auto`.
6. **Model export:** Saved the optimized model using `pickle` for deployment.

## Results

The optimized SVM (polynomial kernel, C=1000) outperformed the linear baseline. Exact held-out accuracy scores were evaluated via `accuracy_score` on the 20% test split. SVM with TF-IDF is a well-proven combination for text classification — the polynomial kernel with low degree effectively captures token co-occurrence patterns without overfitting.

## Tech stack

`Python` · `pandas` · `scikit-learn` · `NLTK` · `regex` · `TF-IDF` · `pickle`

## Run it locally

```bash
git clone https://github.com/matthewkane-ml/ML_NLP_MTK.git
cd ML_NLP_MTK
pip install -r requirements.txt
jupyter notebook src/explore.ipynb
```

## What I'd do next

- Evaluate additional metrics beyond accuracy — precision, recall, and F1 matter more than accuracy for imbalanced spam datasets
- Try a Naive Bayes baseline, which is often a surprisingly strong performer on bag-of-words text tasks
- Experiment with character-level n-grams or domain-specific tokenization to better capture URL structure (e.g., subdomains, TLDs, path segments)
- Add a deployment layer — a Flask or Streamlit interface for live URL checking

---

**Author:** Matthew Kane — [LinkedIn](https://www.linkedin.com/in/thomas-k-392094410/) · [GitHub portfolio](https://github.com/matthewkane-ml)
