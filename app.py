import spacy
import re
import math
import pandas as pd
import numpy as np
from spacy.lang.en.stop_words import STOP_WORDS
from string import punctuation
nlp = spacy.load("en_core_web_sm")
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from heapq import nlargest
import string
from definations import *


#********************************************************************Input*********************************************************************************
# Choice menu and choice and text input from user
intro()
choice = int(input("Enter choice 1 or 2"))

test_data = input("Enter the text")


#************************************************************Offensiveness and sentiment analysis***********************************************************
# Intializing stopwords and punctuation
stopword =list(STOP_WORDS)
punctuation = punctuation + '\n'

# Read csv file from system and creating dataframe and labelling it.
df = pd.read_csv("twitter_data.csv")
df['labels'] = df['class'].map({0 : "Hate Speech Detected" , 1: "Offensive language Detected" , 2: "No hate and offesive speech" })
df = df[['tweet', 'labels']]


# applying cleaning to the text
#Cleaning the data
def clean(text):
    text = str(text).lower()
    text = re.sub('\[.*?\]','',text)
    text = re.sub('https?://S+|www\.\S+','',text)
    text = re.sub('[%s]'% re.escape(string.punctuation),'',text)
    text = re.sub('\n','',text)
    text = re.sub('\w*\d\w*','', text)
    text = [word for word in text.split(' ') if word not in stopword]
    text = " ".join(text)
    return text

df["tweet"] = df["tweet"].apply(clean)


# creating x and y
x = np.array(df["tweet"])
y = np.array(df["labels"])


# Fit and train the model using tweet data
cv = CountVectorizer()
x = cv.fit_transform(x)
x_train,x_test,y_train,y_test = train_test_split(x , y , test_size=.33 , random_state= 42)
clf = DecisionTreeClassifier()
clf.fit(x_train,y_train)



#***********************************************************************Text summarizer***************************************************************************
doc = nlp(test_data)
tokens = [token.text for token in doc]

# Text cleaning providing word frequency counter
word_freq = {}
stop_words = list(STOP_WORDS)
word_freq = word_freq_counter(doc , word_freq)

#Normalizing the word counter 
max_freq = max(word_freq.values())
for word in word_freq.keys():
    word_freq[word] = word_freq[word] / max_freq
  
# sentence tokenizer
sent_token = [sent for sent in doc.sents]
sent_score = {}
sent_score = sentence_score(sent_token , sent_score , word_freq)



#**************************************************************************Output************************************************************************************

# selecting 30% sentence with max score and summarizing it
if choice==1:
    num_lines = math.ceil(len(sent_score) * .3)
    summary = nlargest(n = num_lines, iterable = sent_score , key = sent_score.get)
    print("\n\n Summary of the text is : \n\n")
    print(summary)


# Predicting the speech sentiments and offensiveness
if choice==2:
    df = cv.transform([test_data]).toarray()
    print("\n\n sentiments/Offensiveness of the text is : \n\n")
    print(clf.predict(df))
            
