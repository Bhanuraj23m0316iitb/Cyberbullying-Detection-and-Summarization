import re
from spacy.lang.en.stop_words import STOP_WORDS
from string import punctuation
import string


#Introduction to menu
def intro():
    print("""\n\n**********************************************Choice Menu************************************************
                                    
        *************************************Text speech summurizer and its cyber Bullyeing detection*************************************
                                        
                                        1. Check the summary of the Text Input
                                        2. Check whether the Lnaguge was Offensive or not
                                        
               \n\n """)

#Cleaning the data
def clean(text , stopword):
    text = str(text).lower()
    text = re.sub('\[.*?\]','',text)
    text = re.sub('https?://S+|www\.\S+','',text)
    text = re.sub('[%s]'% re.escape(string.punctuation),'',text)
    text = re.sub('\n','',text)
    text = re.sub('\w*\d\w*','', text)
    text = [word for word in text.split(' ') if word not in stopword]
    text = " ".join(text)
    return text



#word requency counter for summarizer
def word_freq_counter(doc , word_freq):
    stop_words = list(STOP_WORDS)
    for word in doc:
        if word.text.lower() not in stop_words:
            if word.text.lower() not in punctuation:
                if word.text not in word_freq.keys():
                    word_freq[word.text] = 1
                else:
                    word_freq[word.text] +=1
    return word_freq


# Calculating sentence score 
def sentence_score(sent_token , sent_score , word_freq):
    for sent in sent_token:
        for word in sent:
            if word.text.lower() in word_freq.keys():
                if sent not in sent_score.keys():
                    sent_score[sent] = word_freq[word.text.lower()]
                else:
                    sent_score[sent] += word_freq[word.text.lower()]  
    return sent_score                  
