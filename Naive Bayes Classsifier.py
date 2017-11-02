# -*- coding: utf-8 -*-

import sklearn
from sklearn.model_selection import train_test_split
import nltk
from nltk import FreqDist
from nltk import classify
from nltk.classify import apply_features
import numpy as np
from html.parser import HTMLParser
import re
from nltk.corpus import stopwords
from nltk.tokenize import TweetTokenizer
from autocorrect import spell
from textblob import TextBlob
from nltk.stem import PorterStemmer
from nltk.tokenize import TreebankWordTokenizer
import nltk.classify.util
from nltk.classify import NaiveBayesClassifier
import collections
import numpy as np
from sklearn.metrics import confusion_matrix
from nltk import (precision,recall,f_measure,confusionmatrix)

def read_training_data(filename):
    with open(filename,'r') as tsv:
        trainTweet = [line.strip().split('\t') for line in tsv]
        return trainTweet

data= read_training_data('TaskA_modified.txt')

##Splitting training and test data
sentences=[]
sentiment=[]


##Preprocessing tweets
def preprocessing(original_tweet):
    #Removing URLs
    result1 = re.sub(r"http\S+", "", original_tweet)
    result1 = re.sub(r"https\S+", "", result1)
    #print(" After Removing any links in the tweet:\n")
    #print(result1)
    ##Escaping HTML characters
    html_parser = HTMLParser()
    result2 = html_parser.unescape(result1)
    #print("After removing html characters:\n")
    #print(result2)
    ##TreebankTokenizer
    result3=TreebankWordTokenizer().tokenize(result2)
    #print("With TreebankWordTokenizer:\n")
    #print(result3)
    ##Apostrophe lookup
    Appost_dict={"'s":"is","'re":"are","'ve":"have","n't":"not","d":"had","'ll":"will","'m":"am",}
    reformed=[Appost_dict[word] if word in Appost_dict else word for word in result3]
    result4=" ".join(reformed)
    #print(result4)
    ##Remove special characters
    result5=re.sub(r"[!@#$%^&*()_+-=:;?/~`'’]",' ',result4)
    #print("After removing special characters:\n")
    #print(result5)
    ##Tweet tokenizer
    tkznr=TweetTokenizer(reduce_len=True,strip_handles=True,)
    result6=tkznr.tokenize(result5)
    #print("After Tweet Tokenizing:\n")
    #print(result6)
    result7=" ".join(result6)
    result8=TextBlob(result7)
    corr_tweet=result8.correct()
    return corr_tweet

tweets=[]
for cols in data:
    preprocessed_sentence=preprocessing(cols[2])
    words_filtered=[e.lower() for e in preprocessed_sentence.split() if len(e) >=3]
    tweets.append((words_filtered,cols[1]))
    
# Function to get all words
def get_words_from_tweets(tweets):
    all_words=[]
    for (words_filtered,sentiment) in tweets:
        only_words=[e for e in words_filtered]
        all_words.extend(only_words)
    return all_words

words_list=get_words_from_tweets(tweets)

##To count the frequency
def get_word_freq(words_list):
    worddict=nltk.FreqDist(words_list)
    return worddict

word_dict=get_word_freq(words_list)
##Listing all words in order of occurence
word_features=word_dict.keys()
###Relevant features
def extract_features(tweets):
    tweets_words=set(tweets)
    features={}
    for word in word_features:
       features['contains(%s)' %word]=word in tweets_words
    return features
##This will test every tweet with our corpus if that word is present or not
train_tweets,test_tweets=train_test_split(tweets,test_size=0.3)
training_set=nltk.classify.apply_features(extract_features,train_tweets)
test_set=nltk.classify.apply_features(extract_features,test_tweets)
classifier=nltk.NaiveBayesClassifier.train(training_set) 
accuracy=nltk.classify.accuracy(classifier,test_set)
print("The overall Accuracy of the classifier is: %.3f\n" %accuracy)

actual = collections.defaultdict(set)
predicted = collections.defaultdict(set)
actual_label=[]
predicted_label=[]

for i, (feats, label) in enumerate(test_set):
    actual[label].add(i)
    observed = classifier.classify(feats)
    predicted[observed].add(i)
    actual_label.append(label)
    predicted_label.append(label)
    
conf_matrix=nltk.ConfusionMatrix(actual_label,predicted_label)
print("Confusion Matrix:\n")
print(conf_matrix)
        
print("*****************Classification Report**************************")
class_values=actual.keys()
for i in class_values:
    print("For sentiment:" +str(i))
    print("***********************************************")
    print("Precision is:")
    print(nltk.precision(actual[str(i)],predicted[str(i)]))
    print("Recall is :")
    print(nltk.recall(actual[str(i)],predicted[str(i)]))
    print("F measure is :")
    print(nltk.f_measure(actual[str(i)],predicted[str(i)]))
    print("***********************************************")




