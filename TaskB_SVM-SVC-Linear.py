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
from sklearn.metrics import classification_report # used for classification report
from sklearn.metrics import accuracy_score # used for accuracy score calculation

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
MAR_macro_average=0;

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
        test_MAR=0;
        time_rbf_train = t1-t0
        print("Modelling Time for a Topic:",cols[0],"\n")
        print("Total time taken to train %ds" %(time_rbf_train))
        for testcols in tweets_per_topic:
            if(testcols[0]!=cols[0]):
                test_topic_count=test_topic_count+1
                topicwise_test_df=testcols[1]
                test_vectors = vectorizer.transform(topicwise_test_df['tweet'].tolist())
                prediction_rbf = classifier_linearsvm.predict(test_vectors)
                accuracy=((accuracy_score(topicwise_test_df['sentiment'].tolist(),prediction_rbf)))
                MAR = ((recall_score(topicwise_test_df['sentiment'].tolist(),prediction_rbf, average='macro')))
                test_accuracy=test_accuracy+accuracy
                test_MAR=test_MAR+MAR;
                print("Classification report for Topic:",cols[0],"\n")
                print(classification_report(topicwise_test_df['sentiment'],prediction_rbf))
                t2 = time.time()
                time_rbf_train = t1-t0
                time_rbf_predict = t2-t1
                print("Prediction Time for a Test Topic:",testcols[0],"\n")
                print("Total time taken to predict %ds" %(time_rbf_predict))
    elif (len(topic_labels)==1) and (topic_labels=="negative"):
        topicwise_df.loc[-1]=[cols[0],'not available', 'positive']        
        train_vectors = vectorizer.fit_transform(topicwise_df['tweet'].tolist())
        classifier_linearsvm = svm.LinearSVC()  
        t0 = time.time()
        classifier_linearsvm.fit(train_vectors, topicwise_df['sentiment'].tolist())
        t1 = time.time()
        test_topic_count=0;
        test_accuracy=0;
        test_MAR=0;
        time_rbf_train = t1-t0
        print("Modelling Time for a Topic:",cols[0],"\n")
        print("Total time taken to train %ds" %(time_rbf_train))
        for testcols in tweets_per_topic:
            if(testcols[0]!=cols[0]):
                test_topic_count=test_topic_count+1
                topicwise_test_df=testcols[1]
                test_vectors = vectorizer.transform(topicwise_test_df['tweet'].tolist())
                prediction_rbf = classifier_linearsvm.predict(test_vectors)
                accuracy=((accuracy_score(topicwise_test_df['sentiment'].tolist(),prediction_rbf)))
                MAR = ((recall_score(topicwise_test_df['sentiment'].tolist(),prediction_rbf, average='macro')))
                test_accuracy=test_accuracy+accuracy  
                test_MAR=test_MAR+MAR;
                print("Classification report for topic:",cols[0],"\n")
                print(classification_report(topicwise_test_df['sentiment'],prediction_rbf))
                t2 = time.time()
                time_rbf_predict = t2-t1
                print("Prediction Time for a Topic:",testcols[0],"\n")
                print("Total time taken to predict %ds" %(time_rbf_predict))
    else:
        train_vectors = vectorizer.fit_transform(topicwise_df['tweet'].tolist())
        ##Classification with SVM and kernel is rbf
        classifier_linearsvm = svm.LinearSVC()  
        t0 = time.time()
        classifier_linearsvm.fit(train_vectors, topicwise_df['sentiment'].tolist())
        t1 = time.time()
        test_topic_count=0;
        test_accuracy=0;
        test_MAR=0;
        time_rbf_train = t1-t0
        print("Modelling Time for a Topic:",cols[0],"\n")
        print("Total time taken to train %ds" %(time_rbf_train))
        for testcols in tweets_per_topic:
            if(testcols[0]!=cols[0]):
                test_topic_count=test_topic_count+1
                topicwise_test_df=testcols[1]
                test_vectors = vectorizer.transform(topicwise_test_df['tweet'].tolist())
                prediction_rbf = classifier_linearsvm.predict(test_vectors)
                accuracy=((accuracy_score(topicwise_test_df['sentiment'].tolist(),prediction_rbf)))
                MAR = ((recall_score(topicwise_test_df['sentiment'].tolist(),prediction_rbf, average='macro')))
                test_accuracy=test_accuracy+accuracy;
                test_MAR=test_MAR+MAR;
                print("Classification Report for Topic:",cols[0],"\n")
                print(classification_report(topicwise_test_df['sentiment'],prediction_rbf))
                t2 = time.time()
                time_rbf_train = t1-t0
                time_rbf_predict = t2-t1
                print("Prediction Time for a Topic:",testcols[0],"\n")
                print("Total time taken to predict %ds" %(time_rbf_predict))
    test_accuracy_average=test_accuracy/test_topic_count;
    topic_average=topic_average+test_accuracy_average
    
    test_MAR_average=test_MAR/test_topic_count;
    MAR_macro_average=MAR_macro_average+test_MAR_average
    
SVM_Accuracy=topic_average/topic_count;
print("Subtask-B Topic Averaged Accuracy for SVM - SVCLinear is :",SVM_Accuracy)  

SVM_MAR=MAR_macro_average/topic_count;
print("Subtask-B Macro Averaged Recall SVM Linear Kernel is :",SVM_MAR)     