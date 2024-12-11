# Sentiment-Analysis-Using-Text

A Python-based sentiment analysis project that predicts the sentiment of text as **positive**, **neutral**, or **negative**. This project leverages advanced Natural Language Processing (NLP) techniques and a deep learning model built with TensorFlow and Keras.

**Features**

- **Text Preprocessing**: Includes stopword removal, lemmatization, and tokenization using NLTK.
- **Deep Learning Model**: LSTM-based architecture for robust sentiment prediction.
- **User Interaction**: A command-line interface for real-time text sentiment prediction.
- **Customizable**: Easily extendable for other datasets or additional functionalities.

**Dataset** 
- Trained on Twitter sentiment data (Tweets.csv).

**Prerequisites**
1. Python 3.7+
2. Libraries: Install the required dependencies using the command:
   - pip install tensorflow nltk scikit-learn pandas

**Command to run**
- python predict_sentiment.py

**How It Works**
- **Data Preprocessing:**
  - Removes URLs, mentions, hashtags, and punctuation.
  - Converts text to lowercase.
  - Tokenizes and lemmatizes the text.
- **Model Architecture:**
  - Embedding layer for text vectorization.
  - LSTM layer for capturing context.
  - Fully connected dense layer with softmax activation for sentiment classification.
- **Prediction:**
  - Processes user input using the same tokenizer as the training process.
  - Predicts sentiment using the trained LSTM model.

**Output**

![Output](https://github.com/user-attachments/assets/d955f7c1-69fc-4073-8924-62dc21276cf5)
