#Team Sentiment Scrutiny: SemEval 2017 Task [4]
#Subramaniyan Janani M12484583
#Venkataramanan Archana M12511297
#Vemparala Sahithi M12484014
#Murali Nithya M12485228

import matplotlib.pyplot as plt #used for plotting graphs
from plotly.offline import download_plotlyjs, init_notebook_mode,  plot #used for plotting graphs
from plotly.graph_objs import *   #used for plotting graphs
import re #Used for pre-processing 
import nltk
from nltk.tokenize import TweetTokenizer # used for tokenization
from html.parser import HTMLParser # used to remove the html tags
from nltk.tokenize import TreebankWordTokenizer #used for tokenization
from sklearn.model_selection import train_test_split #used for cross validation
from nltk import pos_tag #used for tagging Parts of speech
import collections #used to collect the set of actual and predicted labels
#from nltk.classify import MaxentClassifier # used to invoke Maxentclassifier

#Function to read training data
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
    res1 = re.sub(r"http\S+", "", original_tweet)
    res1 = re.sub(r"https\S+", "", res1)
    #print(" After Removing any links in the tweet:\n")
    #print(result1)
    ##Escaping HTML characters
    html_parser = HTMLParser()
    res2 = html_parser.unescape(res1)
    #print("After removing html characters:\n")
    #print(result2)
    ##TreebankTokenizer
    res3=TreebankWordTokenizer().tokenize(res2)
    #print("With TreebankWordTokenizer:\n")
    #print(result3)
    ##Contractions Removal  
    Appost_dict={"'s":"is","'re":"are","'ve":"have","n't":"not","d":"had","'ll":"will","'m":"am",}
    reformed=[Appost_dict[word] if word in Appost_dict else word for word in res3]
    res4=" ".join(reformed)
    #print(result4)
    ##Remove special characters
    res5=re.sub(r"[!@#$%^&*()_+-=:;?/~`'’]",' ',res4)
    #print("After removing special characters:\n")
    #print(result5)
    ##Tweet tokenizer
    tkznr=TweetTokenizer(reduce_len=True,strip_handles=True,preserve_case=False)
    res6=tkznr.tokenize(res5)
    #print("After Tweet Tokenizing:\n")
    #print(result6)
    res7=" ".join(res6)
    corrected_tweet=res7
    #print(corr_tweet)
    return corrected_tweet

tweets=[]
tagged_train_sentences=[]
all_words_tags=[]
unique_word_tags=[]
for cols in actual_training_data:
    tag_sentence=[]
    preprocessed_sentence=preprocessing(cols[2])
    words_filtered=[e.lower() for e in preprocessed_sentence.split() if len(e) >=3]
    #tagged_train_sentences.append((pos_tag(words_filtered)))
    tag_sentence.append((pos_tag(words_filtered)))
    #Storing the words and tags in a list to be used in building model
    for wordstags in tag_sentence:
        all_words_tags.extend(wordstags) 
    #word_tag_df=pd.DataFrame(tagged_train_sentences)
    tweets.append((tag_sentence,cols[1]))

#Function to build POS Feature dictionary
def get_word_freq(words_list):
    worddict=nltk.FreqDist(words_list)
    return worddict

#Filtering only the required sentiment bearing POS tag 
required_word_tag=[]
for (word,tag) in all_words_tags:
    if(tag=='JJ') or (tag=='JJR') or (tag=='JJS') or (tag=='RB') or (tag=='RBR') or (tag=='RBS') or (tag=='NN') or (tag=='NNS')or (tag=='NNP') or (tag=='NNPS') or (tag=='UH') or (tag=='VB') or (tag=='VBD') or (tag=='VBG') or (tag=='VBN')or (tag=='VBP') or (tag=='VBZ'):
        required_word_tag.append((word,tag))

#Storing the POS Feature Dictionary
word_dict=get_word_freq(required_word_tag)

#Listing all unique POS tagged words
word_postag=word_dict.keys()

#Cross-Validation to split training and test data
train_tweets,test_tweets=train_test_split(tweets,test_size=0.3)

# Extracting Required features by cross chcking with POS Feature dictionary
def extract_features(tweets):
    wordtag=[] #to fetch the word and tag tuple from input argument tweets    
    features={} # feature array variable
    for wordstags in tweets: #seperating the word and tag tuples into a list
        wordtag.extend(wordstags)
    matched=0;
    for i in word_postag:
        for j in wordtag:
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

#This will check every tagged tweet with our buit POS Feature dictionary if the word and tag combination is present or not
training_set=nltk.classify.apply_features(extract_features,train_tweets)
#Modelling the classifier
classifier=nltk.MaxentClassifier.train(training_set) 
#Testing the model to predict the accuracy
test_set=nltk.classify.apply_features(extract_features,test_tweets)

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
    predicted_label.append(observed)
    
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