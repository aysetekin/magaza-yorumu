# Sentiment Analysis on Movie & Store Reviews (NLP Project)

## Description
This project applies Natural Language Processing (NLP) techniques to perform sentiment analysis on textual reviews from movie (IMDb) and store platforms. The goal is to classify reviews as positive or negative using multiple machine learning models and compare their accuracy across different feature extraction methods.

## Features
- Text cleaning: lowercasing, punctuation & number removal, stopword filtering (Turkish)
- Lemmatization using `TextBlob` and NLTK
- Vectorization:
  - CountVectorizer
  - TF-IDF (word-level, n-gram, character-level)
  - HashingVectorizer
- Model comparison:
  - Logistic Regression
  - Naive Bayes
  - Random Forest
  - Decision Tree
  - SVM

## Technologies
- Python
- pandas, numpy
- sklearn (scikit-learn)
- matplotlib, seaborn
- nltk
- textblob

## Results
- All models were evaluated using 10-fold cross-validation
- Accuracy comparisons were made across feature extraction methods
- WordCloud and bar plots visualized term frequencies

## How to Run
1. Install required packages:
```bash
pip install -r requirements_yorum.txt
