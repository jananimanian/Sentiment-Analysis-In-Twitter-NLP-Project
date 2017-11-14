import matplotlib.pyplot as plt # used for plotting graphs
from plotly.offline import download_plotlyjs, init_notebook_mode,  plot # used for plotting graphs
from plotly.graph_objs import *   # used for plotting graphs
import re # used for pre-processing 
from nltk.tokenize import TweetTokenizer # used for tokenization
from html.parser import HTMLParser # used to remove the html tags
from nltk.tokenize import TreebankWordTokenizer #used for tokenization
from sklearn.feature_extraction.text import TfidfVectorizer # used for TF-IDF vector generation
#from sklearn.model_selection import train_test_split # used for cross validation
import numpy as np # used for columnstack
import pandas as pd # used for dataframe constuction
import time # used to calculate time of the classification
from sklearn import svm # used to invoke classifier
from collections import Counter
import math
from sklearn.metrics import mean_absolute_error # to calculate mean absolute error

#read training data
def read_training_data(filename):
    with open(filename,'r') as tsv:
        trainTweet = [line.strip().split('\t') for line in tsv]
        return trainTweet

# Plotting initial SubTask B - Training Data
def plotlabels(positive,negative):
    datas = [{'label':'positive', 'color': 'g', 'height': positive},
             {'label':'negative', 'color': 'b', 'height': negative}]
    i = 0
    for data in datas:
        plt.bar(i, data['height'],align='center',color=data['color'])
        i += 1
    labels = [data['label'] for data in datas]
    pos = [i for i in range(len(datas)) ]
    plt.xticks(pos, labels)
    plt.xlabel('Emotions')
    plt.title('Sentiment Analysis')
    plt.show();
    
#Reading training data    
training_data=read_training_data('2016-part2-subtaskBD.tsv')

#Calculating the count of sentiment classes in the training data
positive=0;
negative=0;
posna=0;
negna=0;

for cols in training_data:
    if cols[2]=="positive":
        if(cols[3])=="Not Available":
            posna=posna+1;
        positive=positive+1;
    elif cols[2]=="negative":
        if(cols[3])=="Not Available":
            negna=negna+1;
        negative=negative+1;
        
#calling plotlabels to plot the given tweet w.r.t sentiment labels
plotlabels(positive,negative);

#plotting to visuvalise the count of "Not Available" tweets Vs Actual Tweets 
pos_tweet=positive-posna;
neg_tweet=negative-negna;

trace0 = Bar(
    x=['positive','negative'],
    y=[pos_tweet,neg_tweet],
    name='Tweets',
    text='Tweets',
    textposition='auto'
    
)
trace1 = Bar(
    x=['positive','negative'],
    y=[posna,negna],
    name='Missing Tweets',
    text='Missing Tweets',
    textposition='auto'  
)
data1 = [trace0,trace1]
layout1 = Layout(
    showlegend=False,
    height=600,
    width=800,
    barmode='stack'
)

fig1 = dict( data=data1, layout=layout1 )
plot(fig1,filename='stacked_graph.html') 

#Removing the missing tweets and storing the actual tweets for further processing
actual_training_data=[]
for cols in training_data:  
    if cols[3]!="Not Available":
        actual_training_data.append(cols)

#recalculating the counts of sentiment classes
positive=0;
negative=0;
for cols in actual_training_data:
    if cols[2]=="positive":
          positive=positive+1;
    elif cols[2]=="negative":
          negative=negative+1;

#plotting the actual positive and negative classes of training data
plotlabels(positive,negative);

#Pre-Processing Function
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
    ##Contractions Removal  
    Appost_dict={"'s":"is","'re":"are","'ve":"have","n't":"not","d":"had","'ll":"will","'m":"am",}
    reformed=[Appost_dict[word] if word in Appost_dict else word for word in result3]
    result4=" ".join(reformed)
    #print(result4)
    ##Remove special characters
    result5=re.sub(r"[!@#$%^&*()_+-=:;?/~`'’]",' ',result4)
    #print("After removing special characters:\n")
    #print(result5)
    ##Tweet tokenizer
    tkznr=TweetTokenizer(reduce_len=True,strip_handles=True,preserve_case=False)
    result6=tkznr.tokenize(result5)
    #print("After Tweet Tokenizing:\n")
    #print(result6)
    result7=" ".join(result6)
    corr_tweet=result7
    #print(corr_tweet)
    return corr_tweet

def KLD(true, pred):
    epsilon = 0.5 / len(pred)
    countsTrue, countsPred = Counter(true), Counter(pred)
    #print("KLD",countsTrue.keys())
    
    if(len(countsTrue)==2):
        [actual_pos, actual_neg]=countsTrue.values()
        if(len(countsPred)==2):
            [Predicted_pos, Predicted_neg]=countsPred.values()
        elif(len(countsPred)==1):
            [p_pos_or_p_neg]=countsPred.keys()
            if(p_pos_or_p_neg=="negative"):
                Predicted_pos=1
                [Predicted_neg]=countsPred.values()
            elif(p_pos_or_p_neg=="positive"):
                Predicted_neg=1
                [Predicted_pos]=countsPred.values()               
    elif(len(countsTrue)==1):
        [pos_or_neg]=countsTrue.keys()
        if(pos_or_neg=="negative"):
            #print("inside negative")
            [actual_neg]=countsTrue.values()
            actual_pos=1
            if(len(countsPred)==2):
                [Predicted_pos, Predicted_neg]=countsPred.values()
            elif(len(countsPred)==1):
                [p_pos_or_p_neg]=countsPred.keys()
                if(p_pos_or_p_neg=="negative"):
                    Predicted_pos=1
                    [Predicted_neg]=countsPred.values()
                elif(p_pos_or_p_neg=="positive"):
                    Predicted_neg=1
                    [Predicted_pos]=countsPred.values()           
        elif(pos_or_neg=="positive"):
            #print("inside positive")
            [actual_pos]=countsTrue.values()
            actual_neg=1
            if(len(countsPred)==2):
                [Predicted_pos, Predicted_neg]=countsPred.values()
            elif(len(countsPred)==1):
                [p_pos_or_p_neg]=countsPred.keys()
                if(p_pos_or_p_neg=="negative"):
                    Predicted_pos=1
                    [Predicted_neg]=countsPred.values()
                elif(p_pos_or_p_neg=="positive"):
                    Predicted_neg=1
                    [Predicted_pos]=countsPred.values()      
    p_pos = actual_pos/len(true)
    p_neg = actual_neg/len(true)
    est_pos = Predicted_pos/len(true)
    est_neg = Predicted_neg/len(true)
    p_pos_s = (p_pos + epsilon)/(p_pos+p_neg+2*epsilon)
    p_neg_s = (p_neg + epsilon)/(p_pos+p_neg+2*epsilon)
    est_pos_s = (est_pos+epsilon)/(est_pos+est_neg+2*epsilon)
    est_neg_s = (est_neg+epsilon)/(est_pos+est_neg+2*epsilon)
    return p_pos_s*math.log10(p_pos_s/est_pos_s)+p_neg_s*math.log10(p_neg_s/est_neg_s)

#Reading the data from file

sentences=[]
sentiment=[]
topic=[]
for cols in actual_training_data:
    topic.append(cols[1])
    sentences.append(cols[3])
    sentiment.append(cols[2])
    
#Pre-Processing the data 
processed_sentences=[]
for i in sentences:   
     processed_sentences.append(preprocessing(i))
tweets=np.column_stack((topic,processed_sentences,sentiment)).tolist()

#Initialising TF-IDF Vector
vectorizer = TfidfVectorizer(sublinear_tf=True,use_idf=True)

#Creating dataframe for picking each topic's data
traintweet_df=pd.DataFrame(tweets)
traintweet_df.columns=['topic','tweet','sentiment']
tweets_per_topic=traintweet_df.groupby('topic')
topic_count=0;
topic_average=0;
KLD_Topic=[]
#AE_Topic=[]
for cols in tweets_per_topic:
    topic_count=topic_count+1;
    topicwise_df=cols[1]
    topic_labels=np.unique(topicwise_df['sentiment']);
    topic_name=np.unique(topicwise_df['topic']);
    if (len(topic_labels)==1) and (topic_labels=="positive"):
        topicwise_df.loc[-1]=[topic_name,'not available','negative']
        train_vectors = vectorizer.fit_transform(topicwise_df['tweet'].tolist())
        classifier_linearsvm = svm.LinearSVC()  
        t0 = time.time()
        classifier_linearsvm.fit(train_vectors, topicwise_df['sentiment'].tolist())
        t1 = time.time()
        test_topic_count=0;
        test_accuracy=0;
        time_rbf_train = t1-t0
        s=[]
    #    AE_Test=[]
        for testcols in tweets_per_topic:
            if(testcols[0]!=cols[0]):    
                test_topic_count=test_topic_count+1
                topicwise_test_df=testcols[1]
                test_vectors = vectorizer.transform(topicwise_test_df['tweet'].tolist())
                prediction_rbf = classifier_linearsvm.predict(test_vectors)
                s.append(KLD(topicwise_test_df['sentiment'].tolist(),prediction_rbf.tolist()))
   #             AE_Test.append(mean_absolute_error(KLD(topicwise_test_df['sentiment'].tolist(),prediction_rbf.tolist())))
                t2 = time.time()
                time_rbf_train = t1-t0
                time_rbf_predict = t2-t1
    elif (len(topic_labels)==1) and (topic_labels=="negative"):
        topicwise_df.loc[-1]=[cols[0],'not available', 'positive']        
        train_vectors = vectorizer.fit_transform(topicwise_df['tweet'].tolist())
        classifier_linearsvm = svm.LinearSVC()  
        t0 = time.time()
        classifier_linearsvm.fit(train_vectors, topicwise_df['sentiment'].tolist())
        t1 = time.time()
        test_topic_count=0;
        test_accuracy=0;
        time_rbf_train = t1-t0
        s=[]
  #      AE_Test=[]
        for testcols in tweets_per_topic:
            if(testcols[0]!=cols[0]):
                test_topic_count=test_topic_count+1
                topicwise_test_df=testcols[1]
                test_vectors = vectorizer.transform(topicwise_test_df['tweet'].tolist())
                prediction_rbf = classifier_linearsvm.predict(test_vectors)
                s.append(KLD(topicwise_test_df['sentiment'].tolist(),prediction_rbf.tolist()))
 #               AE_Test.append(mean_absolute_error(KLD(topicwise_test_df['sentiment'].tolist(),prediction_rbf.tolist())))
                t2 = time.time()
                time_rbf_predict = t2-t1
    else:
        train_vectors = vectorizer.fit_transform(topicwise_df['tweet'].tolist())
        ##Classification with SVM and kernel is rbf
        classifier_linearsvm = svm.LinearSVC()  
        t0 = time.time()
        classifier_linearsvm.fit(train_vectors, topicwise_df['sentiment'].tolist())
        t1 = time.time()
        test_topic_count=0;
        test_accuracy=0;
        time_rbf_train = t1-t0
        s=[]
 #       AE_Test=[]
        for testcols in tweets_per_topic:
            if(testcols[0]!=cols[0]):
                test_topic_count=test_topic_count+1
                topicwise_test_df=testcols[1]
                test_vectors = vectorizer.transform(topicwise_test_df['tweet'].tolist())
                prediction_rbf = classifier_linearsvm.predict(test_vectors)
                s.append(KLD(topicwise_test_df['sentiment'].tolist(),prediction_rbf.tolist()))
                #AE_Test.append(mean_absolute_error(topicwise_test_df['sentiment'].tolist(),prediction_rbf.tolist()))
                t2 = time.time()
                time_rbf_train = t1-t0
                time_rbf_predict = t2-t1
    KLD_test_avg=sum(s)/test_topic_count
    KLD_Topic.append(KLD_test_avg)
    #AE_test_avg=sum(AE_Test)/test_topic_count
    #AE_Topic.append(AE_test_avg)

SVM_KLD_Topic_Average=sum(KLD_Topic)/topic_count;
#SVM_AE_Topic_Average=sum(AE_Topic)/topic_count;

print("Subtask-D Kullback Leibler Divergence for Topicwise Tweet Distribution is :",SVM_KLD_Topic_Average)       
#print("Subtask-D Absolute Error for Topicwise Tweet Distribution is :",SVM_AE_Topic_Average)       