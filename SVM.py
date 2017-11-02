from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
import time
from sklearn import svm
from sklearn.metrics import classification_report
from sklearn.metrics import accuracy_score
import re
from html.parser import HTMLParser
from nltk.corpus import stopwords
from nltk.tokenize import TweetTokenizer
from autocorrect import spell
from textblob import TextBlob
from nltk.stem import PorterStemmer
from nltk.tokenize import TreebankWordTokenizer
"""
tweets=[("I love this Laptop","Positive"),("The dish tastes amazing","Positive"),
            ("I feel great this morning","Positive"),
           ("I am so excited about the show","Positive"),("He is my best friend","Positive"),
           ("I do not like this laptop","Negative"),("This dish is horrible","Negative"),
            ("I feel tired this morning","Negative"),
            ("I am not looking forward to the show","Negative"),("He is my enemy","Negative"),
            ]
"""
def read_training_data(filename):
    with open(filename,'r') as tsv:
        trainTweet = [line.strip().split('\t') for line in tsv]
        return trainTweet

data= read_training_data('TaskC_modified.txt')
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
    return result7

for cols in data:
    sentences.append(cols[3])
    sentiment.append(cols[2])
    
processed_sentences=[]
for i in sentences:   
     processed_sentences.append(preprocessing(i))
     
train_tweet,test_tweet,train_senti,test_senti=train_test_split(processed_sentences,sentiment,test_size=0.3)
vectorizer = TfidfVectorizer(sublinear_tf=True,
                             use_idf=True)
train_vectors = vectorizer.fit_transform(train_tweet)
test_vectors = vectorizer.transform(test_tweet)

##Classification with SVM and kernel is rbf
classifier_rbf = svm.SVC()
t0 = time.time()
classifier_rbf.fit(train_vectors, train_senti)
t1 = time.time()
prediction_rbf = classifier_rbf.predict(test_vectors)
t2 = time.time()
time_rbf_train = t1-t0
time_rbf_predict = t2-t1
#print("Predicted value: %s and actual value:%s" %(prediction_rbf,test_senti))
print("******************SVM kernel=rbf*******************")
print("Accuracy is: %.3f \n" %((accuracy_score(test_senti,prediction_rbf))*100))
print("Classification report:\n")
print(classification_report(test_senti,prediction_rbf))
print("Total time taken: %ds" %(time_rbf_train+time_rbf_predict))
print("***************************************************")

##Classification with SVM and kernel is linear
classifier_linear = svm.SVC(kernel='linear')
t0 = time.time()
classifier_linear.fit(train_vectors, train_senti)
t1 = time.time()
prediction_linear = classifier_linear.predict(test_vectors)
t2 = time.time()
time_linear_train = t1-t0
time_linear_predict = t2-t1
#print("Predicted value: %s and actual value:%s" %(prediction_linear,test_senti))
print("******************SVM kernel=linear*******************")
print("Accuracy is: %.3f \n" %((accuracy_score(test_senti,prediction_linear))*100))
print("Classification report:\n")
print(classification_report(test_senti,prediction_linear))
print("Total time taken: %ds" %(time_linear_train+time_linear_predict))
print("***************************************************")

##Classification with Linear SVM
classifier_linearsvm = svm.LinearSVC()
t0 = time.time()
classifier_linearsvm.fit(train_vectors, train_senti)
t1 = time.time()
prediction_linearsvm = classifier_linearsvm.predict(test_vectors)
t2 = time.time()
time_linearsvm_train = t1-t0
time_linearsvm_predict = t2-t1
#print("Predicted value: %s and actual value:%s" %(prediction_linearsvm,test_senti))
print("******************SVM kernel=linearsvc*******************")
print("Accuracy is: %.3f \n" %((accuracy_score(test_senti,prediction_linearsvm))*100))
print("Classification report:\n")
print(classification_report(test_senti,prediction_linearsvm))
print("Total time taken: %ds" %(time_linearsvm_train+time_linearsvm_predict))
print("***************************************************")
    
