# Fake News Classification using Naive Bayes & TF-IDF

This repository contains a Data Science project designed to detect and classify news articles as either **Real** or **Fake**. The implementation uses the **Naive Bayes** machine learning algorithm combined with **Term Frequency-Inverse Document Frequency (TF-IDF)** feature extraction techniques.

## 📌 Project Overview
Misinformation and fake news have grown exponentially with the rise of online social platforms. This project builds a reliable text classification pipeline that cleans text data, removes generic stop words, translates core content into numerical vectors via TF-IDF, and predicts article authenticity using a Naive Bayes classifier.

## 📊 Dataset Detail
- **Source:** Kaggle Fake News Dataset (`news_articles.csv`)
- **Size:** ~11 MB
- **Total Records:** 2,096 raw records (2,040 post-cleaning)
- **Attributes Available:** `author`, `published`, `title`, `text`, `language`, `site_url`, `type`, `label`, `hasImage`, and pre-processed textual segments.

## 🛠️ Tech Stack & Libraries
- **Language:** Python 3
- **Environment:** Jupyter Notebook / Google Colab
- **Data Manipulation:** `pandas`, `numpy`
- **Visualization:** `matplotlib`, `seaborn`
- **Text Preprocessing:** `gensim`
- **Machine Learning:** `scikit-learn` (Naive Bayes & TF-IDF Vectorizer)

## 🚀 Key Implementation Workflow

### 1. Data Collection & Inspection
- Mounted file resources natively to access local archives.
- Profiled datatypes (11 `object` text structures, 1 numeric identifier).
- Analyzed null counts, discovering text gaps in 46 core entries alongside 10 duplicates.

### 2. Exploratory Data Analysis (EDA)
- **Imbalance Check:** Verified that more than half of the corpus was explicitly classified as **Fake**.
- **Feature Dependencies:** Evaluated how auxiliary variables (such as the presence of an image feature `hasImage`) shift authenticity probabilities (e.g., presence of an image increased fake news probability by 35.78% within the set).
- Identified and plotted the *Top 10 Fake News Publishers* across 491 unique metadata authors.

### 3. Data Preprocessing & Cleaning
- Dropped non-contributing dimensions like `main_img_url`.
- Re-engineered date markers (`published`) from generic objects into standard Python datetime format.
- Replaced target class string labels (`Fake` / `Real`) with standard machine-readable boolean parameters (`1` / `0`) into a new target vector `is_fake`.
- Resolved text formatting alignment using `gensim.parsing.preprocessing.remove_stopwords` to dynamically construct missing token sets.
- Balanced text article categories by condensing categorical variance (e.g., re-mapping scattered labels like generic `fake` to fixed tags like `state`).

### 4. Feature Extraction & Modeling
- Transformed processed headlines (`title_without_stopwords`) and articles (`text_without_stopwords`) into unique weights using parallel **TF-IDF Vectorizers**.
- Extracted and flattened auxiliary layout properties like `language`, `hasImage`, and emotional bias metrics into a dense configuration row matrix.
- Horizontally stacked text vectors with contextual metadata arrays via structural `scipy.sparse.hstack` components.
- Trained a **Naive Bayes** model optimized to output discrete boolean target lists (`0` for Real, `1` for Fake).

## 🔮 Sample Evaluation
The final segment of the notebook extracts sample rows to print definitive structural inferences:
```python
# Apply the trained TF-IDF vectorizers & stack dense layout attributes
sample_combined = sp.hstack([sample_text_tfidf, sample_title_tfidf, sample_other_features_dense])

# Execute authenticity prediction
predictions = model.predict(sample_combined)
print("Predictions for the sample news articles:", predictions)

📂 How to Run
Clone this repository:

Bash
git clone [https://github.com/your-username/your-repo-name.git](https://github.com/your-username/your-repo-name.git)
Download the source dataset directly from Kaggle and place it within your path environment.

Run the notebook DS-Project-FakeNewsClassification-usingNaiveBayesModel.ipynb in Jupyter Notebook or upload it directly to Google Colab.
