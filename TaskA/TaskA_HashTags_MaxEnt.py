# -*- coding: utf-8 -*-
"""
Created on Tue Oct 31 14:07:02 2017

@author: Sahithi
"""
import re
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
from nltk.classify import MaxentClassifier
import collections
import numpy as np
from sklearn.metrics import confusion_matrix
from nltk import (precision,recall,f_measure,confusionmatrix)
from sklearn.naive_bayes import GaussianNB
from sklearn.tree import DecisionTreeClassifier

"""
tweets=[["@Microsoft 2nd computer with same error!!! #Windows10fail Guess we will shelve this until SP1! http://t.co/QCcHlKuy8Q","Negative"],
         ["#teens @BillGates 1st company failed miserably. When Gates &amp; @PaulGAllen tried to sell the product it wouldn't work #nevergiveup @Microsoft","Negative"],
          ["Top 5 most searched for Back-to-School topics -- the list may surprise you http://t.co/Xj21uMVo0p  @bing @MSFTnews #backtoschool @Microsoft","Positive"],
           ["@ForbesRussia #MBA #casestudy Namaste 2 #google and @Microsoft's CEOs, but #Multiculturals mttr!May B the era of Bad Translations wd B Over?","Neutral"],
            ["We're excited to learn about #cloud #analytics from @Microsoft tomorrow! Join us https://t.co/p0bMREBBHC #tech #rva http://t.co/1XHmPdSvzq","Positive"]]
             
 """
def read_training_data(filename):
    with open(filename,'r') as tsv:
        trainTweet = [line.strip().split('\t') for line in tsv]
        return trainTweet

data= read_training_data('TaskA_modified.txt')


         
sentences=[]
sentiment=[]
tweets=[]
for cols in data:
        sentences.append(cols[2])
        sentiment.append(cols[1])
        tweets.append((cols[2],cols[1]))

hash_words=[]
hash_word=[]
def hash_tag_removal(word):
    for i in word:
        if(i=="#"):
           word=word.replace(i,'')
    return word
           
###Extracting only hash tag words
for i in range(len(tweets)):
    hash_word.append([hash_tag_removal(j) for j in tweets[i][0].split() if j.startswith("#")])

for i in range(len(hash_word)):
    for j in range(len(hash_word[i])):
        hash_words.append([hash_word[i][j],sentiment[i]])
        

#train_words,test_words,train_senti,test_senti=train_test_split(words,sentiment,test_size=0.1)
# Function to get all words
def get_words_from_tweets(hash_words):
    all_words=[]
    for i in range(len(hash_words)):
        all_words.append(hash_words[i][0])
    return all_words

words_list=get_words_from_tweets(hash_words)

##To count the frequency
def get_word_freq(words_list):
    worddict=nltk.FreqDist(words_list)
    return worddict

word_dict=get_word_freq(words_list)
##Listing all words in order of occurence
word_features=word_dict.keys()
###Relevant features
def extract_features(train_tweets):
    features={}
    for word in word_features:
       features['contains(%s)' %word]=word in train_tweets
    return features
##This will test every tweet with our corpus if that word is present or not
train_tweets,test_tweets=train_test_split(hash_words,test_size=0.1)
training_set=nltk.classify.apply_features(extract_features,train_tweets)
test_set=nltk.classify.apply_features(extract_features,test_tweets)
classifier=nltk.MaxentClassifier.train(training_set) 
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





                

    

