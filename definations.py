import re
import string
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize, sent_tokenize
from gensim.models import Word2Vec

# Download required NLTK data (only once)
nltk.download('punkt')
nltk.download('stopwords')

# Introduction to menu
def intro():
    print("""\n\n********************************************** Choice Menu ************************************************
                                    
        ************************************* Text Speech Summarizer and Cyber Bullying Detection *************************************
                                        
                                        1. Check the summary of the Text Input
                                        2. Check whether the Language was Offensive or not
               \n\n """)

# Clean the data
def clean(text):
    text = text.lower()
    text = re.sub(r'\[.*?\]', '', text)
    text = re.sub(r'https?://\S+|www\.\S+', '', text)
    text = re.sub(r'[%s]' % re.escape(string.punctuation), '', text)
    text = re.sub(r'\n', ' ', text)
    text = re.sub(r'\w*\d\w*', '', text)
    return text

# Word frequency counter for summarizer
def word_freq_counter(text):
    stop_words = set(stopwords.words('english'))
    words = word_tokenize(text)
    freq = {}

    for word in words:
        if word not in stop_words and word not in string.punctuation:
            freq[word] = freq.get(word, 0) + 1
    return freq

# Sentence scoring using word frequencies
def sentence_score(text, word_freq):
    sentences = sent_tokenize(text)
    sent_scores = {}

    for sent in sentences:
        words = word_tokenize(sent.lower())
        for word in words:
            if word in word_freq:
                sent_scores[sent] = sent_scores.get(sent, 0) + word_freq[word]
    return sent_scores

# Word2Vec training (optional enhancement for future use)
def train_word2vec(text):
    sentences = sent_tokenize(text)
    tokenized_sentences = [word_tokenize(sent.lower()) for sent in sentences]
    model = Word2Vec(sentences=tokenized_sentences, vector_size=100, window=5, min_count=1, workers=2)
    return model

# Example usage
if __name__ == "__main__":
    intro()
    input_text = """Social media platforms are often used to spread negativity. 
                    Some people use offensive language and bully others online. 
                    However, AI can be used to detect and prevent such behavior."""
    
    cleaned_text = clean(input_text)
    freq = word_freq_counter(cleaned_text)
    scores = sentence_score(input_text, freq)
    w2v_model = train_word2vec(input_text)

    top_sentences = sorted(scores.items(), key=lambda kv: kv[1], reverse=True)[:2]
    summary = ' '.join([sent for sent, score in top_sentences])
    
    print("\n--- Summary ---\n", summary)
