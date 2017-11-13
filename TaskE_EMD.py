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

def read_training_data(filename):
    with open(filename,'r') as tsv:
        trainTweet = [line.strip().split('\t') for line in tsv]
        return trainTweet
    
# Plotting initial SubTask C - Training Data
def plotlabels(highlypositive,positive,neutral,negative,highlynegative):
    datas = [{'label':'highly positive', 'color': 'm', 'height': h_positive},
             {'label':'positive', 'color': 'g', 'height': positive},
             {'label':'neutral', 'color': 'y', 'height': neutral},
             {'label':'negative', 'color': 'b', 'height': negative},
             {'label':'highly negative', 'color': 'r', 'height': h_negative}]
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
training_data=read_training_data('2016subtaskCEtrain.tsv')
h_positive=0;
positive=0;
negative=0;
neutral=0;
h_negative=0;

hposna=0;
posna=0;
neuna=0;
negna=0;
hnegna=0;

for cols in training_data:
    if cols[2]== '2':
        if(cols[3])=="Not Available":
            hposna=hposna+1;
        h_positive=h_positive+1;
    elif cols[2]=='1':
        if(cols[3])=="Not Available":
            posna=posna+1;
        positive= positive+1;
    elif cols[2]=='0':    
        if(cols[3])=="Not Available":
            neuna=neuna+1;
        neutral=neutral+1;
    elif cols[2]=='-1':
        if(cols[3])=="Not Available":
            negna=negna+1;
        negative=negative+1;
    elif cols[2]== '-2':
        if(cols[3])=="Not Available":
            hnegna=hnegna+1;
        h_negative=h_negative+1;
    

plotlabels(h_positive,positive,neutral,negative,h_negative);

pos_tweet=positive-posna;
neg_tweet=negative-negna;
neu_tweet=neutral-neuna;
hpos_tweet=h_positive-hposna;
hneg_tweet=h_negative-hnegna;

trace0 = Bar(
    x=['highly positive','positive','neutral','negative','highly negative'],
    y=[hpos_tweet,pos_tweet,neu_tweet,neg_tweet,hneg_tweet],
    name='Tweets',
    text='Tweets',
    textposition='auto'
    
)
trace1 = Bar(
    x=['highly positive','positive','neutral','negative','highly negative'],
    y=[hposna,posna,neuna,negna,hnegna],
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
plot(fig1,filename='stacked_graph') 

actual_training_data=[]
for cols in training_data:  
    if cols[3]!="Not Available":
        actual_training_data.append(cols)
positive=0;
negative=0;
neutral=0;
h_positive = 0;
h_negative = 0;
for cols in actual_training_data:
    if cols[2]=="2":
          h_positive=h_positive+1;
    elif cols[2]=="1":
          positive=positive+1;
    elif cols[2]=="0":
        neutral=neutral+1;
    elif cols[2]=="-1":
          negative=negative+1;
    elif cols[2]=="-2":
          h_negative=h_negative+1;
   
plotlabels(h_positive,positive,neutral,negative,h_negative);

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

#Function to calculate EMD
def EMD(true, pred):
    Actual_cls=[0,0,0,0,0]
    Predicted_cls=[0,0,0,0,0]
    countsTrue, countsPred = Counter(true), Counter(pred)
    Actual_cls_int=dict((int(k),int(v)) for k,v in countsTrue.items())
    for k in Actual_cls_int:
        Actual_cls[k]=Actual_cls_int[k]
        
    Predict_cls_int=dict((int(k),int(v)) for k,v in countsPred.items())
    for k in Predict_cls_int:
        Predicted_cls[k]=Predict_cls_int[k]
   
    p_pos = Actual_cls[1]/len(true)
    p_hpos = Actual_cls[2]/len(true)
    neu=Actual_cls[0]/len(true)
    n_neg=Actual_cls[-1]/len(true)
    n_hneg=Actual_cls[-2]/len(true)
    """
    print("p_pos",p_pos)
    print("p_hpos",p_hpos)
    print("neu",neu)
    print("n_neg",n_neg)
    print("n_hneg",n_hneg)
    """
    est_pos = Predicted_cls[1]/len(pred)
    est_hpos= Predicted_cls[2]/len(pred)
    est_neu= Predicted_cls[0]/len(pred)
    est_hneg= Predicted_cls[-2]/len(pred)
    est_neg =  Predicted_cls[-1]/len(pred)
    """
    print("est_pos",est_pos)
    print("est_hpos",est_hpos)
    print("est_neu",est_neu)
    print("est_neg",est_neg)
    print("est_hneg",est_hneg)
    """
    diff_hneg=abs(est_hneg-n_hneg)
    diff_neg=abs(est_neg-n_neg)
    diff_pos=abs(est_pos-p_pos)
    diff_hpos=abs(est_hpos-p_hpos)
    diff_neu=abs(est_neu-neu)
    emd_value=diff_hneg+diff_neg+diff_hpos+diff_pos+diff_neu;
    return emd_value

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
EMD_Topic=[]
for cols in tweets_per_topic:
    topic_count=topic_count+1;
    topicwise_df=cols[1]
    topic_labels=np.unique(topicwise_df['sentiment']);
    topic_name=np.unique(topicwise_df['topic']);
    train_vectors = vectorizer.fit_transform(topicwise_df['tweet'].tolist())
    ##Classification with SVM SVC-Linear
    classifier_linearsvm = svm.LinearSVC() 
    t0 = time.time()
    classifier_linearsvm.fit(train_vectors, topicwise_df['sentiment'].tolist())
    t1 = time.time()
    test_topic_count=0;
    test_accuracy=0;
    time_rbf_train = t1-t0
    s=[]
    for testcols in tweets_per_topic:
        if(testcols[0]!=cols[0]):
            test_topic_count=test_topic_count+1
            topicwise_test_df=testcols[1]
            test_vectors = vectorizer.transform(topicwise_test_df['tweet'].tolist())
            prediction_rbf = classifier_linearsvm.predict(test_vectors)
            s.append(EMD(topicwise_test_df['sentiment'].tolist(),prediction_rbf.tolist()))
            t2 = time.time()
            time_rbf_train = t1-t0
            time_rbf_predict = t2-t1
               
    test_EMD_average=sum(s)/test_topic_count;
    EMD_Topic.append(test_EMD_average)
    
SVM_EMD_Value=sum(EMD_Topic)/topic_count;
print("Subtask-E Earth Mover's Distance for Topicwise Tweet Distribution is :",SVM_EMD_Value)