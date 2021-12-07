from flask import Flask, request, render_template
from tensorflow import keras
import numpy as np
import pandas as pd
import pickle

import requests
import json

import re
import nltk
from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer
from googlesearch import search

nltk.download('stopwords')

ps = PorterStemmer()

app = Flask(__name__)
model = keras.models.load_model("fake-news-analyzer.h5")
cv = pickle.load(open("cvectorizer", "rb"))

def preprocess_column(col_name, col_length):
    preprocessed_data = []
    for i in range(0,col_length):
        col_value = dataset[col_name][i]
        col_value = re.sub('[^a-zA-Z]', ' ', col_value)
        col_value = col_value.lower()
        col_value = col_value.split()
        col_value = [ps.stem(word) for word in col_value if not word in set(stopwords.words('english'))]            
        col_value = ' '.join(col_value)
        preprocessed_data.append(col_value)
    return preprocessed_data

@app.route('/')#home page
def home():
    return render_template("FakeNewsAnalyser.html")

@app.route('/predict', methods = ["POST", "GET"])
def predict():
    if request.method == "POST":
        News = request.form["News"]
        News = re.sub('[^a-zA-Z]', ' ', News) 
        News = News.lower()
        News = News.split() 
        News = [ps.stem(word) for word in News if not word in set(stopwords.words('english'))]
        News = ' '.join(News)
        #print(News)
        search_results = []
        for j in search(News, tld="com", stop=5, pause=1):
            #print(j)
            search_results.append(j)
        if (model.predict(cv.transform([News])))>0.5:
            #print("Real")
            return render_template('PositiveOutcome.html', search_results=search_results)
        else:
            #print("Fake")
            return render_template('NegativeOutcome.html', search_results=search_results)

if __name__ == "__main__":
    app.run(debug = True)

#headline1 = "Social gatherings of more than six people will be banned across England Wales and Scotland from tomorrow. But what are the new rules what happens if you break them and how do they differ across the nations? ðŸ‘‡"
#headline2 = "Korona virus, very new deadly form of virus, china is suffering, may come to India immediately, avoid any form of cold drinks, ice creams, koolfee, etc, any type of preserved foods, milkshake, rough ice, ice colas, milk sweets older then 48 hours, for atleast 90 days from today."
#headline3 = "As tuberculosis shaped modernism, so COVID-19 and our collective experience of staying inside for months on end will influence architectureâ€™s near future, @chaykak writes. https://t.co/ag34yZckbU"
#headline4 = "SUBHAN ALLAH: AFTER CORONA VIRUS CHINA GOVT LIFTED BAN ON HOLY QURAN & ALLOWED CHINESE MUSLIMS TO READ THEIR SACRED BOOK! SO WHICH OF THE FAVORS OF YOUR LORD WOULD YOU DENY?"zzzzz
#headline5 = "#IndiaFightsCorona Following the national trend 17 States/UTs have more new recoveries than new cases. https://t.co/aHWwlaimmb"