import nltk
import re
import math
import pandas as pd
import numpy as np
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize, sent_tokenize
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.model_selection import train_test_split
from sklearn.ensemble import GradientBoostingClassifier
from heapq import nlargest
from gensim.models import Word2Vec
from definations import *  # assuming you still use sentence_score() and word_freq_counter()

nltk.download('punkt')
nltk.download('stopwords')

# ********************************************************************Input*********************************************************************************
def intro():
    print("""\n\n**********************************************Choice Menu************************************************
        *************************************Text summarizer and cyber bullying detection*************************************
        1. Check the summary of the Text Input
        2. Check whether the Language was Offensive or not
    \n\n """)

intro()
choice = int(input("Enter choice 1 or 2: "))
test_data = input("Enter the text: ")

# ************************************************************Offensiveness and sentiment analysis***********************************************************
stopword = set(stopwords.words('english'))

# Read and label data
df = pd.read_csv("Cyberbullying-Detection-and-Summarization\twitter data.csv")
df['labels'] = df['class'].map({0: "Hate Speech Detected", 1: "Offensive language Detected", 2: "No hate and offensive speech"})
df = df[['tweet', 'labels']]

# Clean the text
def clean(text):
    text = str(text).lower()
    text = re.sub(r'\[.*?\]', '', text)
    text = re.sub(r'https?://\S+|www\.\S+', '', text)
    text = re.sub(r'[%s]' % re.escape(string.punctuation), '', text)
    text = re.sub(r'\n', '', text)
    text = re.sub(r'\w*\d\w*', '', text)
    words = [word for word in word_tokenize(text) if word not in stopword]
    return " ".join(words)

df["tweet"] = df["tweet"].apply(clean)

x = np.array(df["tweet"])
y = np.array(df["labels"])

# Train model
cv = CountVectorizer()
x = cv.fit_transform(x)
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.33, random_state=42)
clf = GradientBoostingClassifier()
clf.fit(x_train, y_train)

# ***********************************************************************Text summarizer***************************************************************************

# Clean and tokenize input
cleaned_text = clean(test_data)
sentences = sent_tokenize(test_data)
words = word_tokenize(cleaned_text)

# Train a small Word2Vec model on the input if no pretrained model is loaded
w2v_model = Word2Vec([word_tokenize(s) for s in sentences], vector_size=100, window=5, min_count=1, workers=2)

# Word frequency using pretrained Word2Vec model
def word_freq_counter_w2v(words, model):
    word_freq = {}
    for word in words:
        if word in model.wv:
            word_freq[word] = np.linalg.norm(model.wv[word])
    return word_freq

word_freq = word_freq_counter_w2v(words, w2v_model)

# Normalize frequency
if word_freq:
    max_freq = max(word_freq.values())
    for word in word_freq.keys():
        word_freq[word] = word_freq[word] / max_freq

# Sentence scoring using same helper
sent_score = {}
sent_score = sentence_score(sentences, sent_score, word_freq)

# **************************************************************************Output************************************************************************************

# Summarization
if choice == 1:
    num_lines = math.ceil(len(sent_score) * 0.3)
    summary = nlargest(n=num_lines, iterable=sent_score, key=sent_score.get)
    print("\n\n Summary of the text is : \n\n")
    for sent in summary:
        print(sent)

# Offensiveness detection
elif choice == 2:
    df_vec = cv.transform([cleaned_text]).toarray()
    print("\n\n Sentiments/Offensiveness of the text is : \n\n")
    print(clf.predict(df_vec))
