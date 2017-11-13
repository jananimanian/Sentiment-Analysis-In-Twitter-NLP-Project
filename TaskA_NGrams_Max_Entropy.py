import matplotlib.pyplot as plt #used for plotting graphs
from plotly.offline import download_plotlyjs, init_notebook_mode,  plot #used for plotting graphs
from plotly.graph_objs import *   #used for plotting graphs
import re #Used for pre-processing 
import nltk
from nltk.tokenize import TweetTokenizer # used for tokenization
from html.parser import HTMLParser # used to remove the html tags
from nltk.tokenize import TreebankWordTokenizer #used for tokenization
from sklearn.model_selection import train_test_split #used for cross validation
from nltk import ngrams #used to assign ngrams

def read_training_data(filename):
    with open(filename,'r') as tsv:
        trainTweet = [line.strip().split('\t') for line in tsv]
        return trainTweet

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

n_gram=2;
tweets=[]
all_ngrams=[]

for cols in actual_training_data:
    ngram_sentence=[]
    preprocessed_sentence=preprocessing(cols[2])
    words_filtered=[e.lower() for e in preprocessed_sentence.split() if len(e) >=3]
    ngram_sentence.append((list(ngrams((words_filtered),n_gram))))
    #Storing the bigrams in a list to be used in building model
    for wordstags in ngram_sentence:
        all_ngrams.extend(wordstags) 
    tweets.append((ngram_sentence,cols[1]))
    
#Function to build Bigram Feature dictionary
def get_word_freq(words_list):
    worddict=nltk.FreqDist(words_list)
    return worddict

#Storing the Bigram Feature Dictionary
word_dict=get_word_freq(all_ngrams)

#Listing all unique Bigram words
uniq_ngram=word_dict.keys()

#Cross-Validation to split training and test data
train_tweets,test_tweets=train_test_split(tweets,test_size=0.3)

# Extracting Required features by cross chcking with POS Feature dictionary
def extract_features2(tweets):
    bigram_word=[] #to fetch the bigram tuple from input argument tweets    
    features={} # feature array variable
    for bigram_words in tweets: #seperating the bigram tuples into a list
        bigram_word.extend(bigram_words)
        
    matched=0;
    for i in uniq_ngram:
        for j in bigram_word:
            (a,b)=([(a == b) for a, b in zip(i,j)])
            if(a==True) and (b==True):
                matched=1
                break
            else:
                matched=0
        if(matched==1):
            features['contains(%s,%s)' %(i)]=True # setting features array to true if the combination is present in our POS Feature dictionary
        else:
            features['contains(%s,%s)' %(i)]=False # setting features array to False if the combination is not present in our POS Feature dictionary              
    return features

# Extracting Required features by cross chcking with POS Feature dictionary
def extract_features3(tweets):
    trigram_word=[] #to fetch the trigram tuple from input argument tweets    
    features={} # feature array variable
    for trigram_words in tweets: #seperating the trigram tuples into a list
        trigram_word.extend(trigram_words)
        
    matched=0;
    for i in uniq_ngram:
        for j in trigram_word:
            x,y,z=([(a == b) for a, b in zip(i,j)])
            if(x==True) and (y==True) and(z==True):
                matched=1
                break
            else:
                matched=0
        if(matched==1):
            features['contains(%s,%s,%s)' %(i)]=True # setting features array to true if the combination is present in our POS Feature dictionary
        else:
            features['contains(%s,%s,%s)' %(i)]=False # setting features array to False if the combination is not present in our POS Feature dictionary              
    return features

#This will check every tagged tweet with our buit POS Feature dictionary if the word and tag combination is present or not
if(n_gram==2):
    training_set=nltk.classify.apply_features(extract_features2,train_tweets)
elif(n_gram==3):
    training_set=nltk.classify.apply_features(extract_features3,train_tweets)
    
#Modelling the classifier
classifier=nltk.MaxentClassifier.train(training_set) 

#Testing the model to predict the accuracy
if(n_gram==2):
    test_set=nltk.classify.apply_features(extract_features2,test_tweets)
elif(n_gram==3):
    test_set=nltk.classify.apply_features(extract_features3,test_tweets)

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
print("Contingency table values:\n")
print(conf_matrix)
        
print("---------------------Classification Report------------------------")
class_values=actual.keys()
for i in class_values:
    print("For sentiment:" +str(i))
    print("-----------------------------------------------")
    print("Precision value:")
    print(nltk.precision(actual[str(i)],predicted[str(i)]))
    print("Recall value :")
    print(nltk.recall(actual[str(i)],predicted[str(i)]))
    print("F measure is :")
    print(nltk.f_measure(actual[str(i)],predicted[str(i)]))
    print("-----------------------------------------------")