from flask import Flask,request
from flask_restful import Resource, Api, reqparse
import pandas as pd
import ast
import nltk
nltk.download('punkt')
from nltk.stem.lancaster import LancasterStemmer
stemmer= LancasterStemmer()
from quantulum3 import parser
import tensorflow as tf
import keras
import random
import json
import numpy as np 
import spacy
import pickle

from nltk.tokenize.treebank import TreebankWordDetokenizer


def clean_up_sentence(sentence):
    # tokenizing the pattern
    sentence_words = nltk.word_tokenize(sentence)
    # stemming each word
    sentence_words = [stemmer.stem(word.lower()) for word in sentence_words]
    return sentence_words

# returning bag of words array: 0 or 1 for each word in the bag that exists in the sentence
def bow(sentence, words, show_details=False):
    # tokenizing the pattern
    sentence_words = clean_up_sentence(sentence)
    # generating bag of words
    bag = [0]*len(words)  
    for s in sentence_words:
        for i,w in enumerate(words):
            if w == s: 
                bag[i] = 1
                if show_details:
                    print ("found in bag: %s" % w)

    return(np.array(bag))


context = {}

ERROR_THRESHOLD = 0.25
def classify(sentence):
    # generate probabilities from the model
    results = model.predict(np.array([bow(sentence, words)]))[0]
    # filter out predictions below a threshold
    results = [[i,r] for i,r in enumerate(results) if r>ERROR_THRESHOLD]
    # sort by strength of probability
    results.sort(key=lambda x: x[1], reverse=True)
    return_list = []
    for r in results:
        return_list.append((classes[r[0]], r[1]))
    # return tuple of intent and probability
    return return_list

def whattoadd(text):
  output=[]
  output.append("add")
  ignore2=['of','from','me','with']
  modifiedtext=nltk.word_tokenize(text)
  modifiedtext = [w.lower() for w in modifiedtext if w not in ignore2]
  modifiedtext=TreebankWordDetokenizer().detokenize(modifiedtext)
  quants = parser.parse(modifiedtext)
  try:
    output.append(quants[0].value)
    output.append(quants[0].unit.name)
    start=quants[0].span[0]
    end=quants[0].span[1]
    for text1 in modifiedtext[start:end].split():
      ignore2.append(text1)
  except:
    print()
  
  nlp=spacy.load("en_core_web_sm")
  modifiedtext=nltk.word_tokenize(text)
  modifiedtext = [w.lower() for w in modifiedtext if w not in ignore2]
  modifiedtext=TreebankWordDetokenizer().detokenize(modifiedtext)
  doc=nlp(modifiedtext)
  for chunk in doc.noun_chunks:
    if chunk.root.dep_== 'dobj':
      output.append(chunk.text)
      
    #print(chunk.text,chunk.root.text,chunk.root.dep_,chunk.root.head.text)
  return output
#---------------------------------------------------------------------------------#
def whattodelete(text):
  output=[]
  output.append("delete")
  ignore2=['of','me','with']
  nlp=spacy.load("en_core_web_sm")
  modifiedtext=nltk.word_tokenize(text)
  modifiedtext = [w.lower() for w in modifiedtext if w not in ignore2]
  modifiedtext2=[]
  for string in modifiedtext:
    new_string = string.replace("delete", "remove")
    modifiedtext2.append(new_string)
  modifiedtext2=TreebankWordDetokenizer().detokenize(modifiedtext2)
  doc=nlp(modifiedtext2)
  for chunk in doc.noun_chunks:
    if chunk.root.dep_== 'dobj':
      output.append(chunk.text)
      print(chunk.text,chunk.root.text,chunk.root.dep_,chunk.root.head.text)
  return output


with open('intents.json') as json_data:
    intents = json.load(json_data)


data = pickle.load( open( "training_data", "rb" ) )
words = data['words']
classes = data['classes']
train_x = data['train_x']
train_y = data['train_y']
model = tf.keras.models.load_model('chatbot.h5')


  
app = Flask(__name__)
#api = Api(app)
@app.route('/classifytext',methods = ['GET'])
def classifytext():
    text = request.args.get('text')
    if(classify(text)[0][0]=='add'):        
        return json.dumps(whattoadd(text)),200
    elif(classify(text)[0][0]=='delete'):
        return json.dumps(whattodelete(text)),200
    
    

if __name__ == '__main__':
    app.run()  # run our Flask app
