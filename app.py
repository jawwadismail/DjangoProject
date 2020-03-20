from flask import Flask,render_template

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
from MODEL import * 



app = Flask(__name__)


@app.route('/')
def hello_world():
    return render_template('index.html')

if __name__ == '__main__':
    app.debug=True
    app.run(host='127.0.0.1',port=5000)