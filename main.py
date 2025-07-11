from fastapi import FastAPI
from fastapi.encoders import jsonable_encoder
from pydantic import BaseModel
import pandas as pd
import numpy as np
import csv as csv
import io
import matplotlib.pyplot as plt
import sklearn
import regex
import nltk
from nltk.tokenize import word_tokenize
nltk.download('punkt')
from nltk.corpus import stopwords
nltk.download('stopwords')
from nltk.stem import WordNetLemmatizer
nltk.download('wordnet')
import joblib
from sklearn.feature_extraction.text import TfidfVectorizer


joblib_in = open('regression_model.joblib','rb')
model=joblib.load(joblib_in)

app = FastAPI()




@app.post("/send")
async def send(data: str):
    data=data 
    data2 = pd.DataFrame([data], columns = ['Statement'])
    data2['Statement'] = preprocess(data2['Statement'])

    df = pd.read_csv('Lemm_df.csv',encoding='latin-1')
    df = df.dropna()
    vectorizer = TfidfVectorizer()
    tfidf_matrix = vectorizer.fit_transform(df['Statement']) #instead of transforming each time could load transformed one 

    new_tfidf = vectorizer.transform(data2['Statement'])
    
    prediction = model.predict(new_tfidf)
    probabilities = model.predict_proba(new_tfidf)
    predictions = model.predict(new_tfidf)

    probabilities=list(probabilities)


    Fake = probabilities[0][0]
    Real = probabilities[0][1]
    Fake = round(Fake*100,1)
    Real = round(Real*100,1)



    Real_Msg = f'REAL! We predicted that the probability this News article is Real is {Real} percent'
    Fake_Msg = f'FAKE! We predicted that the probability this News article is Fake is {Fake} percent'

    if predictions == [1]:
        return {"message": Real_Msg}
    else:
        return {"message": Fake_Msg}

    
    

def preprocess(text):
    text = text.str.replace(r'[^\x00-\x7f]_?', r'', regex=True)
    text = text.str.replace(r'https?://\S+|www\.\S+', r'', regex=True)
    text = text.str.replace(r'[^\w\s]', r'', regex=True)
    text = text.apply(lambda x: word_tokenize(x))
    stop_words = set(stopwords.words('english'))
    text = text.apply(lambda x: ' '.join([word for word in x if word.lower() not in stop_words]))
    lemmatizer = WordNetLemmatizer()
    text = text.apply(lambda x: ' '.join([lemmatizer.lemmatize(word.lower()) for word in x.split()]))
    return text





