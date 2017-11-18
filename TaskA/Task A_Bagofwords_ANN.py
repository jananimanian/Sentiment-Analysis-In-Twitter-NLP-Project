#Team Sentiment Scrutiny: SemEval 2017 Task [4]
#Subramaniyan Janani M12484583
#Venkataramanan Archana M12511297
#Vemparala Sahithi M12484014
#Murali Nithya M12485228
# Subtask-A with Bag of Words Feature Extraction and Classification with Artificial Neural Network
# using the following preprocessing techniques
#Removing URLs
#Escaping HTML characters
#Contractions Removal 
#Remove special characters
#TweetTokenizer
#TreebankTokenizer
#Stoplist
#Eliminating words with size less than three

import matplotlib.pyplot as plt #used for plotting graphs
from plotly.offline import download_plotlyjs, init_notebook_mode,  plot #used for plotting graphs
from plotly.graph_objs import *   #used for plotting graphs
import numpy as np #used for columnstack
import re #used for pre-processing 
from html.parser import HTMLParser # used to remove the html tags
from nltk.corpus import stopwords # used to remove Stopwords
from nltk.tokenize import TweetTokenizer #used for tokenization
from nltk.tokenize import TreebankWordTokenizer #used for tokenization
from sklearn.metrics import * # used to calculate performance metrics

#Funtion to read Training data
def read_training_data(filename):
    with open(filename,'r') as tsv:
        trainTweet = [line.strip().split('\t') for line in tsv]
        return trainTweet
    
#Function to read testing data
def read_testing_data(filename):
    with open(filename,'r') as txt:
        testTweet = [line.strip().split('\t') for line in txt]
        return testTweet

# Plotting initial SubTask A - Training Data
def plotlabels(positive,neutral,negative):
    datas = [{'label':'positive', 'color': 'g', 'height': positive},
             {'label':'neutral', 'color': 'y', 'height': neutral},
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
training_data=read_training_data('2016downloaded4-subtask A.tsv')

#Reading testing data    
#testing_data=read_testing_data('twitter-2016devtest-A.txt')
testing_data=read_testing_data('twitter-2016test-A.txt')

#Extracting Test data tweets and sentiment
test_sentence=[]
test_sentiment=[]
for cols in testing_data:
    test_sentence.append(cols[2])
    test_sentiment.append(cols[1])
    
#Calculating the count of sentiment classes in the training data
positive=0;
negative=0;
neutral=0;
posna=0;
negna=0;
neuna=0;

for cols in training_data:
    if cols[1]=="positive":
        if(cols[2])=="Not Available":
            posna=posna+1;
        positive=positive+1;
    elif cols[1]=='neutral':    
        if(cols[2])=="Not Available":
            neuna=neuna+1;
        neutral=neutral+1;
    elif cols[1]=="negative":
        if(cols[2])=="Not Available":
            negna=negna+1;
        negative=negative+1;
        
#calling plotlabels to plot the given tweet w.r.t sentiment labels
plotlabels(positive,neutral,negative);

#plotting to visuvalise the count of "Not Available" tweets Vs Actual Tweets 
pos_tweet=positive-posna;
neg_tweet=negative-negna;
neu_tweet=neutral-neuna;

trace0 = Bar(
    x=['positive','neutral','negative'],
    y=[pos_tweet,neu_tweet,neg_tweet],
    name='Tweets',
    text='Tweets',
    textposition='auto'
    
)
trace1 = Bar(
    x=['positive','neutral','negative'],
    y=[posna,neuna,negna],
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
    if cols[2]!="Not Available":
        actual_training_data.append(cols)

#recalculating the counts of sentiment classes
positive=0;
negative=0;
neutral=0;
for cols in actual_training_data:
    if cols[1]=="positive":
          positive=positive+1;
    elif cols[1]=="neutral":
          neutral=neutral+1;
    elif cols[1]=="negative":
          negative=negative+1;

#plotting the actual positive and negative classes of training data
plotlabels(positive,neutral,negative);

##Preprocessing tweets
def preprocessing(original_tweet):
    #1.Removing URLs
    res1 = re.sub(r"http\S+", "", original_tweet)
    res1 = re.sub(r"https\S+", "", res1)
    ##2.Escaping HTML characters
    html_parser = HTMLParser()
    res2 = html_parser.unescape(res1)
    ##3.TreebankTokenizer
    res3=TreebankWordTokenizer().tokenize(res2)
    ##4.Contractions Removal  
    Appost_dict={"'s":"is","'re":"are","'ve":"have","n't":"not","d":"had","'ll":"will","'m":"am",}
    transformed=[Appost_dict[word] if word in Appost_dict else word for word in res3]
    ##5.Special Characters Removal 
    res4=" ".join(transformed)
    res5=re.sub(r"[!@#$%^&*()_+-=:;?/~`'’]",' ',res4)
    ##6.Tweet tokenizer
    tkznr=TweetTokenizer(reduce_len=True,strip_handles=True,preserve_case=False)
    res6=tkznr.tokenize(res5)
    ##7.Stopwords Removal
    remove_stopwords=[word for word in res6 if word not in stopwords.words('english')]
    res7= " ".join(remove_stopwords)
    corrected_tweet=res7
    return corrected_tweet

#Building corpus of words,classes and documents with tweets and associated sentiment
words=[]
classes=[]
docs=[]
for cols in training_data:
    if cols[1] not in classes:
        classes.append(cols[1]) #Corpus of Distinct classes
    preprocessed_sentence=preprocessing(cols[2]) # Preprocessing Training Data
    words_filtered=[e.lower() for e in preprocessed_sentence.split() if len(e) >=3] #Restricting data with Length less than 3
    words.extend(words_filtered) # Corpus of words
    docs.append((words_filtered,cols[1])) # Corpus of tweets with its associated sentiment

##Removing duplicate words
words=list(set(words))
training_bag=[] #Holds Unigrams for training data
output=[] #holds class of each sentence in an array

#Building Bag of words for Training data in terms of 0s and 1s
for doc in docs:
    output_class=[0 for i in classes] #Array indicating class of each input sentence
    bag=[]
    input_words=doc[0]
    i_words=[word for word in input_words]
    for word in words:
        # If words from Training Data is present in corpus
        if word in i_words:
            bag.append(1) #Indicating the presence of word with 1
        else:
            bag.append(0) #Indicating the absence of word with 0 
    training_bag.append(bag)
    output_class[classes.index(doc[1])]=1
    output.append(output_class)
   
#Funtion to create bag of words for test data
def bagofwords(sentence_words,words,showdetails="False"):
    bags=[0 for i in range(len(words))]
    for i in sentence_words:
        for j,k in enumerate(words):
            if(i==k):
                bags[j]=1
                #if showdetails:
                 #   print(" Word %s found in bag\n" %i)
    return bags
#Funtion to create non-linear exponential function to normalise values
def sigmoid(x):
    op=1/(1+np.exp(-x))
    return op

#Function to calculate error rate
def derivative(x):
    derv=x*(1-x)
    return derv

#Function to classify sentiment of test data
def classification(test_sentence,w0,w1):
    ret=[]
    class_return=[]
    for sentence in test_sentence:
        processed_sent=preprocessing(sentence) #Preprocessing Test Data
        test_words=TreebankWordTokenizer().tokenize(processed_sent)
        bag=bagofwords(test_words,words,"True") #step to create bag of words for test data based on training data in terms of 0s and 1s
        x=bag #Input is our bag of words
        L0=np.array(x) # layer 0 of Neural Network
        L1=sigmoid(np.dot(L0,w0)) #Layer 1 of Neural Network
        L2=sigmoid(np.dot(L1,w1)) # Layer 2 of Neural Network with probabilities of classes the input belongs to
        probability=[[i,r] for i,r in enumerate(L2)]
        probability.sort(key=lambda x:x[1],reverse=True)# Sorting the class probabilities to pick up the highest probability class
        ret=classes[probability[0][0]]
        class_return.append(ret)# return class with highest probability
    return class_return

#Function to train Neural Network
def train(X,y,hidden_neurons=10,alpha=0.1,iterations=10000):
    (m,n)=X.shape # Bag of words of training Data
    (a,b)=y.shape # Sentiment associated with training data
    #Initialising weights with random numbers
    np.random.seed(1)
    #Intialize weights
    w0=2*np.random.random((n,hidden_neurons))-1 # For Layer 0
    w1=2*np.random.random((hidden_neurons,b))-1 # For Layer 1
    #Building Layers of Neural Network
    for i in range(iterations):
        L_0=X # Bag of words as input
        L_1=sigmoid(np.dot(L_0,w0)) # Function to form layer 1
        L_2=sigmoid(np.dot(L_1,w1)) # Function to form Layer 2
        #To find Error at Layer 2
        L2_error=y-L_2 # determining how much our prediction is differing from actual value at Layer 2
        L2_delta=L2_error*derivative(L_2) #Direction in which prediction is missing
        L1_error=np.dot(L2_delta,w1.T) #Determining how much our prediction is differing from actual value due to Layer 1
        L1_delta=L1_error*derivative(L_1) #Direction in which prediction is missing
        #Updating weights as per error rate
        w0_upd=np.dot(L_0.T,L1_delta)
        w1_upd=np.dot(L_1.T,L2_delta)
        w0=w0+(alpha*w0_upd)
        w1=w1+(alpha*w1_upd)
    return w0,w1

##Inputs for training Neural Network
X=np.array(training_bag)
y=np.array(output)
# To obtain Weights by training Neural Network
w0,w1=train(X,y,hidden_neurons=20,alpha=0.1,iterations=5000)
print("Value of w0 is:")
print(w0)
print("Value of w1 is:")
print(w1) 

#Predicting sentiment of Test Data
predicted_sentiment=[]
predicted_sentiment=classification(test_sentence,w0,w1) # Classifying Input Test data
conf_matrix=confusion_matrix(test_sentiment,predicted_sentiment)
print("**********************Confusion Matrix*****************************")
print(conf_matrix)
accuracy=accuracy_score(test_sentiment,predicted_sentiment)
print("Accuracy of classifier is:")
print(accuracy)
print("**********************Classification report************************")
print(classification_report(test_sentiment,predicted_sentiment))



    



    



          
          
          
          
          
          
          
          
          
          
          
          
          
          
          
          
          
          
          
          
          
          
          
          
          
          
          
          
          
          
          
          
          
          
          
          
          
          
          
          
          
          
          
          
          
          
          
          
          
          
          
          
          
          
          
          
          
          
          
          
          
          
          
          
          
          
          
          
          
          
          
          
          
          
          
          
          
          
          
          
          
          
          
          
          
          
          
          
          
          
          
          
          
          
          
          
          
          
          
          
          
    

        

