# IMDB Sentiment Analysis with RNN

This project performs sentiment analysis on the IMDB movie reviews dataset using a Recurrent Neural Network (RNN) built with PyTorch.

## Project Overview
The goal is to classify movie reviews as either **positive** or **negative**. The project covers the full NLP pipeline, including:
- **Data Pre-processing**: Lowercasing, URL removal, punctuation removal, HTML tag removal, stopword removal, and stemming.
- **Vectorization**: Converting text into numerical form using `TfidfVectorizer` with a limit of 5,000 features.
- **Deep Learning Model**: A custom RNN architecture implemented in PyTorch.
- **Evaluation**: The model achieved an accuracy of approximately 85.65% on the test set.

## Installation
Ensure you have Python installed, then install the necessary dependencies:
```bash
pip install -r requirements.txt
```

**Author**
KRISHNA YADAV
B.Tech in CSE | Machine Learning Enthusiasm
