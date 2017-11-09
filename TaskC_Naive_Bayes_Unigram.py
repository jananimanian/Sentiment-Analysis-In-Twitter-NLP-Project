import matplotlib.pyplot as plt #used for plotting graphs
from plotly.offline import download_plotlyjs, init_notebook_mode,  plot #used for plotting graphs
from plotly.graph_objs import *   #used for plotting graphs
import re #Used for pre-processing 
import nltk
from nltk.tokenize import TweetTokenizer # used for tokenization
from html.parser import HTMLParser # used to remove the html tags
from nltk.tokenize import TreebankWordTokenizer #used for tokenization
from sklearn.model_selection import train_test_split #used for cross validation
import pandas as pd #used for dataframe constuction
import numpy as np #used for columnstack
from sklearn.metrics import recall_score # to calculate recall score

def read_training_data(filename):
    with open(filename,'r') as tsv:
        trainTweet = [line.strip().split('\t') for line in tsv]
        return trainTweet
# Plotting initial SubTask A - Training Data
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

tweets=[]
for cols in actual_training_data:
    #print("Before Preprocessing:\n")
    topic=cols[1];
    sentiment=cols[2];
    sentence=cols[3];
    #print(sentence)
    preprocessed_sentence=preprocessing(sentence)
    #Converting the tweets into lowercase and eliminating words with size less than three
    words_filtered=[e.lower() for e in preprocessed_sentence.split() if len(e) >=3]
    tweets.append((topic,words_filtered,sentiment))

# Function to get all words
def get_words_from_tweets(tweets):
    all_words=[]
    for (topic,words_filtered,sentiment) in tweets:
        only_words=[e for e in words_filtered]
        all_words.extend(only_words)
    return all_words

#Building Corpus
words_list=get_words_from_tweets(tweets)

#Extracting Unigram Features - Word and its frequency
def get_word_freq(words_list):
    worddict=nltk.FreqDist(words_list)
    return worddict

#Building dictionary of unique features with its frequency
word_dict=get_word_freq(words_list)

##Listing all unigrams in the order of occurence
word_features=word_dict.keys()

###Relevant features
def extract_features(tweets):
    tweets_words=set(tweets)
    features={}
    for word in word_features:
       features['contains(%s)' %word]=word in tweets_words
    return features
##
#train_tweets,test_tweets=train_test_split(tweets,test_size=0.3)
traintweet_df=pd.DataFrame(tweets)
traintweet_df.columns=['topic','tweet','sentiment']
#topic_list=traintweet_df['topic'].unique();
tweets_per_topic=traintweet_df.groupby('topic')
#print(tweets_per_topic)
tweets_per_topic2=traintweet_df.groupby('topic').size();
print(tweets_per_topic2)
topic_count=0;
for cols in tweets_per_topic:
    topic_count=topic_count+1;
    #print("topic",cols[0])
    topicwise_df=cols[1]
    #print(topicwise_df)
    tweet_senti_pertopic=np.column_stack((topicwise_df['tweet'],topicwise_df['sentiment'])).tolist()
    #print(tweet_senti_pertopic)
    training_set=nltk.classify.apply_features(extract_features,tweet_senti_pertopic)
    #print("training_set",training_set)
    classifier=nltk.NaiveBayesClassifier.train(training_set)
    prediction=classifier.labels();
    print(prediction)
    test_accuracy=0
    test_topic_count=0
    topic_average=0
    for testcols in tweets_per_topic:
        #print(testcols[0])
        if(testcols[0]!=cols[0]):
            test_topic_count=test_topic_count+1
            topicwise_test_df=testcols[1]
            tweet_senti_test_pertopic=np.column_stack((topicwise_test_df['tweet'],topicwise_test_df['sentiment'])).tolist()
            testing_set=nltk.classify.apply_features(extract_features,tweet_senti_test_pertopic)
            accuracy=nltk.classify.accuracy(classifier,testing_set)
            test_accuracy=test_accuracy+accuracy
            print(accuracy)
    topic_average=topic_average+(test_accuracy/test_topic_count)

NB_Avg=topic_average/topic_count;
print("SubTask-C Naive Bayes Classifier with Unigram feature extraction Accuracy is :", NB_Avg)




