#!/usr/bin/env python
# coding: utf-8



from sklearn.naive_bayes import GaussianNB

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split  
from sklearn.metrics import classification_report,confusion_matrix,accuracy_score
from sklearn.svm import SVC

import os
import nltk
import string 
import re

import pymysql 



import sys
import tweepy
import numpy as np
import pandas as pd
import nltk
from sklearn.svm import SVC

print("HI")

raw=pd.read_csv("./data.csv")
raw['Tweet']=raw.Tweet.apply(lambda x:x.replace('.',''))
raw.head()


raw.shape[0]




#Create lists for tweets and label
Tweet = []
Labels = []
wpt = nltk.WordPunctTokenizer()
wordnet_lemmatizer = nltk.WordNetLemmatizer()


for row in raw["Tweet"]:
    #tokenize words
    words = wpt.tokenize(row)
    #remove punctuations
    clean_words = [word.lower() for word in words if word not in set(string.punctuation)]
    #remove stop words
    english_stops = nltk.corpus.stopwords.words('english')
    characters_to_remove = ["''",'``',"rt","https","’","“","”","\u200b","--","n't","'s","...","//t.c" ]
    clean_words = [word for word in clean_words if word not in english_stops]
    clean_words = [word for word in clean_words if word not in set(characters_to_remove)]
    #Lematise words
    lemma_list = [wordnet_lemmatizer.lemmatize(word) for word in clean_words]
    #print(lemma_list)
    Tweet.append(lemma_list)

    
    
for label in raw["Text Label"]:
    Labels.append(label)
    
c=[]
    
for x in Tweet:
    s=''
    for y in x:
            s=s+' '+y
    s = re.sub('[^A-Za-z0-9" "]+', '', s)
    s=s.lstrip()
    c.append(s)

    
print(c)


# from sklearn.feature_extraction.text import CountVectorizer
# 
# cv = CountVectorizer(min_df=0., max_df=1.)
# # have given min_document_freqn = 0.0 ->  which means ignore 
# # terms that appear in less than 1% of the documents 
# 
# # and max_document_freqn = 1.0 ->  which means ignore terms that appear 
# # in more than 100% of the documents".
# # In short nothing is to be ignored. all data values would be considered 
# 
# cv_matrix = cv.fit_transform(c)
# 
# cv_matrix = cv_matrix.toarray()
# 
# cv_matrix

# 
# 
# 
# # the final preprocessing step is to divide data into training and test sets
# from sklearn.model_selection import train_test_split  
# X_train, X_test, y_train, y_test = train_test_split(cv_matrix, Labels, test_size = 0.20,random_state=0)
# 
# # TYPE your Code here
# # Training the Algorithm. Here we would use simple SVM , i.e linear SVM
# #simple SVM
# 
# from sklearn.svm import SVC
# svclassifier=SVC(kernel='linear')
# 
# 
# 
# # classifying linear data
# 
# 
# # kernel can take many values like
# # Gaussian, polynomial, sigmoid, or computable kernel
# # fit the model over data
# 
# svclassifier.fit(X_train,y_train)
# 
# 
# # Making Predictions
# 
# y_pred=svclassifier.predict(X_test)
# 
# 
# # Evaluating the Algorithm
# 
# from sklearn.metrics import classification_report,confusion_matrix,accuracy_score
# 
# print(confusion_matrix(y_test,y_pred))
# 
# print(classification_report(y_test,y_pred))
# 
# print(accuracy_score(y_test,y_pred))
# 
# # Remember : for evaluating classification-based ML algo use  
# # confusion_matrix, classification_report and accuracy_score.


tv = TfidfVectorizer(min_df=0., max_df=1., use_idf=True)
tv_matrix = tv.fit_transform(c)
tv_matrix = tv_matrix.toarray()

vocab = tv.get_feature_names()

lol=pd.DataFrame(np.round(tv_matrix, 2), columns=vocab)


# In[ ]:





# the final preprocessing step is to divide data into training and test sets
X_train, X_test, y_train, y_test = train_test_split(tv_matrix, Labels, test_size = 0.20,random_state=0)

# TYPE your Code here
# Training the Algorithm. Here we would use simple SVM , i.e linear SVM
#simple SVM

svclassifier=SVC(kernel='linear')



# classifying linear data


# kernel can take many values like
# Gaussian, polynomial, sigmoid, or computable kernel
# fit the model over data

svclassifier.fit(X_train,y_train)


# Making Predictions

y_pred=svclassifier.predict(X_test)

# print(y_pred)


# Evaluating the Algorithm


# print(confusion_matrix(y_test,y_pred))

# print(classification_report(y_test,y_pred))

# print(accuracy_score(y_test,y_pred))

# Remember : for evaluating classification-based ML algo use  
# confusion_matrix, classification_report and accuracy_score.
# And for evaluating regression-based ML Algo use Mean Squared Error(MSE), ...


# In[ ]:




model = GaussianNB()


model.fit(X_train,y_train)



predicted=model.predict(X_test)


# print(confusion_matrix(y_test,y_pred))

# print(classification_report(y_test,y_pred))

# print(accuracy_score(y_test,y_pred))


# In[ ]:


results = model.predict_proba(X_test)[1]

results


svm = SVC(probability=True)
svm = SVC(kernel='linear', probability=True)
svm.fit(X_train, y_train)
LABEL=svm.predict(X_test)
# print(LABEL[1:10])
output_proba = svm.predict_proba(X_test)
# print(output_proba)




# credentials  --> put your credentials here
consumer_key = "EgjCgVxayfmnj68PXiVKu1PVj"
consumer_secret = "QFszM57LLN4GyIOmPBG9Q1bjWceDj7yGBIEuF2xWCyt93BbZ8Z"
access_token = "1199917326285426690-MWVeRP6oBVW7b6X5FEHJhwpNBB2bXf"
access_token_secret = "BqqWCmW1hUGlZhLLyXyjeFYVByDoJlPGIYwpGU949Y89x"


auth = tweepy.OAuthHandler(consumer_key, consumer_secret)
auth.set_access_token(access_token, access_token_secret)
api = tweepy.API(auth)

L  = []
author=[]


class CustomStreamListener(tweepy.StreamListener):
    def __init__(self):
        super().__init__()
        self.counter = 0
        self.limit = 10
        
    def on_status(self, status):
            #print(status.text)
            global L
            L.append(status.text) 
            author.append(status.user.screen_name)
            self.counter += 1
            if self.counter < self.limit:
                return True
            else:
                return False

    def on_error(self, status_code):
        print >> sys.stderr, 'Encountered error with status code:', status_code
        return True # Don't kill the stream

    def on_timeout(self):
        print >> sys.stderr, 'Timeout...'
        return False # Don't kill the stream

    
    
    
sapi = tweepy.streaming.Stream(auth, CustomStreamListener())    
sapi.filter(locations=[-74.36141,40.55905,-73.704977,41.01758])
main = pd.DataFrame(L,columns=['Tweet'])


# print(main)
# print(author)



posts=[]
for row in main["Tweet"]:
    #tokenize words
    words = wpt.tokenize(row)
    #remove punctuations
    clean_words = [word.lower() for word in words if word not in set(string.punctuation)]
    #remove stop words
    english_stops = nltk.corpus.stopwords.words('english')
    characters_to_remove = ["''",'``',"rt","https","’","“","”","\u200b","--","n't","'s","...","//t.c" ]
    clean_words = [word for word in clean_words if word not in english_stops]
    clean_words = [word for word in clean_words if word not in set(characters_to_remove)]
    #Lematise words
    lemma_list = [wordnet_lemmatizer.lemmatize(word) for word in clean_words]
    #print(lemma_list)
    posts.append(lemma_list)
    
    
refine=[]
    
for x in posts:
    s=''
    for y in x:
            s=s+' '+y
    s = re.sub('[^A-Za-z0-9" "]+', '', s)
    s=s.lstrip()
    refine.append(s)    

    
# print(refine)    



#re_tv = TfidfVectorizer(min_df=0., max_df=1., use_idf=True)
re_tv_matrix = tv.transform(refine)
re_tv_matrix = re_tv_matrix.toarray()

# print(re_tv_matrix)

OUTPUT=svm.predict(re_tv_matrix)


SEVERITY=svm.predict_proba(re_tv_matrix)


# print(OUTPUT)
# print(SEVERITY)



ind=list()
for i in main['Tweet']:
    index = list(main['Tweet']).index(i)
    if(OUTPUT[index]=="Bullying"):
        ind.append(author[index])
        
        
# print(ind)        

limit=0

timeline=[]

CRIM_SCORE=[]
CRIM_NAME=[]

for i in ind:
    #print(i)
    #print(limit)
    for po in tweepy.Cursor(api.user_timeline, screen_name=i, tweet_mode="extended",exclude='retweets').items():
        if(limit<10):
            if (not po.retweeted) and ('RT @' not in po.full_text):
                timeline.append(po.full_text)
                #print(po.full_text)
                limit=limit+1
        else:
            limit=0
            break
        
    final= pd.DataFrame(timeline,columns=['Tweet'])
    posts_culprit=[]
    for row in final["Tweet"]:
        #tokenize words
        words = wpt.tokenize(row)
        #remove punctuations
        clean_words = [word.lower() for word in words if word not in set(string.punctuation)]
        #remove stop words
        english_stops = nltk.corpus.stopwords.words('english')
        characters_to_remove = ["''",'``',"rt","https","’","“","”","\u200b","--","n't","'s","...","//t.c" ]
        clean_words = [word for word in clean_words if word not in english_stops]
        clean_words = [word for word in clean_words if word not in set(characters_to_remove)]
        #Lematise words
        lemma_list = [wordnet_lemmatizer.lemmatize(word) for word in clean_words]
        #print(lemma_list)
        posts_culprit.append(lemma_list)


    refine_culprit=[]

    for x in posts:
        s=''
        for y in x:
                s=s+' '+y
        s = re.sub('[^A-Za-z0-9" "]+', '', s)
        s=s.lstrip()
        refine_culprit.append(s)    


    #print(refine_culprit)    

    final_tv_matrix = tv.transform(refine)
    final_tv_matrix = final_tv_matrix.toarray()

    #print(re_tv_matrix)

    OUTPUT_FINAL=svm.predict(final_tv_matrix)


    SEVERITY_FINAL=svm.predict_proba(final_tv_matrix)

    
    #print(OUTPUT_FINAL)
    #print(SEVERITY_FINAL)
    
    for j in range(SEVERITY.shape[0]):
        x
        if(OUTPUT[j]=="Bullying"):    
            x=SEVERITY[j][0]
            #print(x,i)
            for k in range(SEVERITY_FINAL.shape[0]):
                if SEVERITY_FINAL[k][0]>0.5:
                    #print("ADD",SEVERITY_FINAL[k][0])
                    x=x+SEVERITY_FINAL[k][0]
                    #print("ADD")
                else:
                    #print("SUB",SEVERITY_FINAL[k][0])
                    x=x-SEVERITY_FINAL[k][1]
                    #print("SUB")
                #print(x)    
                break
            CRIM_SCORE.append(x)
            CRIM_NAME.append(i)    
    
            
    F_SCORE=[]
    global F_NAME
    F_NAME=[]
    for m in CRIM_SCORE: 
        if m not in F_SCORE: 
            F_SCORE.append(m) 
        
    for m in CRIM_NAME: 
        if m not in F_NAME: 
            F_NAME.append(m) 
        

host = os.getenv('MYSQL_HOST')
port = os.getenv('MYSQL_PORT')
user = os.getenv('MYSQL_USER')
password = os.getenv('MYSQL_PASSWORD')
database = os.getenv('MYSQL_DATABASE')

print((F_SCORE)) 
print((F_NAME))

conn=pymysql.connect(host="localhost",port=int(3306),user="root",password="",db="cyberproject",charset="latin1_swedish_ci")
curse=conn.cursor()


for i in range(len(F_SCORE)):
    curse.execute("INSERT INTO tweets(name,score) into VALUES(?,?)",F_SCORE[i],F_NAME[i])

conn.commit()








# posts_culprit=[]
# for row in final["Tweet"]:
#     #tokenize words
#     words = wpt.tokenize(row)
#     #remove punctuations
#     clean_words = [word.lower() for word in words if word not in set(string.punctuation)]
#     #remove stop words
#     english_stops = nltk.corpus.stopwords.words('english')
#     characters_to_remove = ["''",'``',"rt","https","’","“","”","\u200b","--","n't","'s","...","//t.c" ]
#     clean_words = [word for word in clean_words if word not in english_stops]
#     clean_words = [word for word in clean_words if word not in set(characters_to_remove)]
#     #Lematise words
#     lemma_list = [wordnet_lemmatizer.lemmatize(word) for word in clean_words]
#     #print(lemma_list)
#     posts_culprit.append(lemma_list)
    
    
# refine_culprit=[]
    
# for x in posts:
#     s=''
#     for y in x:
#             s=s+' '+y
#     s = re.sub('[^A-Za-z0-9" "]+', '', s)
#     s=s.lstrip()
#     refine_culprit.append(s)    

    
# # print(refine_culprit)    


# # In[ ]:


# final_tv_matrix = tv.transform(refine)
# final_tv_matrix = final_tv_matrix.toarray()

# # print(re_tv_matrix)

# # OUTPUT_FINAL=svm.predict(re_tv_matrix)


# # SEVERITY_FINAL=svm.predict_proba(re_tv_matrix)


# # print(OUTPUT_FINAL)
# # print(SEVERITY_FINAL)

# # for i in range(SEVERITY.shape[0]):
# #     # print(SEVERITY[i][0])


