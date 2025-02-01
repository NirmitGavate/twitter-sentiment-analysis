# Sentiment Analysis on U.S. Airlines - Twitter Data (2015)

## Overview

This project focuses on sentiment analysis of Twitter data from February 2015, where the objective is to classify the sentiment of tweets about major U.S. airlines and categorize the reasons for negative sentiment. The project utilizes an **LSTM (Long Short-Term Memory) neural network** to perform the analysis and achieve high accuracy in both sentiment and reason classification.

## Goal

The main goals of the project were:
1. **Sentiment Classification**: Classify tweets as positive, negative, or neutral.
2. **Reason Classification**: Identify the reasons for negative sentiment, such as "late flight" or "rude service."

## Dataset

The dataset consists of **Twitter data** scraped from February 2015. The tweets are related to various issues and experiences with major U.S. airlines. The data includes:
- Tweets with sentiment labels (positive, negative, neutral)
- Negative reason labels (e.g., "late flight," "rude service," etc.)

## Approach

1. **Data Preprocessing:**
   - Cleaned the raw text (removed stop words, special characters, and URLs).
   - Tokenized the text and padded sequences for LSTM input.
   - Label encoding for sentiment and reason labels.

2. **Model Building:**
   - **LSTM Neural Network** was chosen for this task due to its effectiveness in handling sequential data like text.
   - The network was composed of an embedding layer, LSTM layers, and fully connected layers to predict sentiment and negative reasons.
   - The model was trained using a **multi-task learning approach** for sentiment and reason classification.

3. **Evaluation Metrics:**
   - **Sentiment Accuracy**: 97.59%
   - **Reason Accuracy**: 94%

4. **Model Tuning and Hyperparameters:**
   - Optimized the LSTM architecture with hyperparameters like learning rate, batch size, and number of LSTM units.
   - Used dropout layers to prevent overfitting and to improve model generalization.

## Results

The model achieved impressive performance on both sentiment and reason classification:

- **Sentiment Accuracy**: 97.59%
- **Reason Accuracy**: 94%

The model demonstrated strong capabilities in understanding customer sentiments towards airlines and identifying specific issues mentioned in the tweets.

## Conclusion

- The **LSTM neural network** provided high accuracy for both sentiment and reason classification tasks, making it suitable for large-scale sentiment analysis tasks on social media data.
- The model's performance could be further enhanced by using more advanced techniques like **BERT** or **GPT** for sentiment analysis.

---

## Dependencies

- Python 3.x
- TensorFlow / Keras
- Scikit-learn
- Pandas
- Numpy
- Matplotlib
- NLTK (for text preprocessing)
