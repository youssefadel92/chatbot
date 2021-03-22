# -*- coding: utf-8 -*-
"""chatbotwithouttrain.ipynb

Automatically generated by Colaboratory.

Original file is located at
    https://colab.research.google.com/drive/1yid53MuvhYtS574BfnB0N45cRNdT0p4z
"""

! pip install quantulum3

import nltk
nltk.download('punkt')
from nltk.stem.lancaster import LancasterStemmer
stemmer= LancasterStemmer()

from quantulum3 import parser

# Commented out IPython magic to ensure Python compatibility.
#  %tensorflow_version 1.x
import tensorflow as tf
import tflearn
import random
import json
import numpy as np 
import spacy
from nltk.tokenize.treebank import TreebankWordDetokenizer

from google.colab import files
files.upload()

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
    results = model.predict([bow(sentence, words)])[0]
    # filter out predictions below a threshold
    results = [[i,r] for i,r in enumerate(results) if r>ERROR_THRESHOLD]
    # sort by strength of probability
    results.sort(key=lambda x: x[1], reverse=True)
    return_list = []
    for r in results:
        return_list.append((classes[r[0]], r[1]))
    # return tuple of intent and probability
    return return_list

def response(sentence, userID='123', show_details=False):
    results = classify(sentence)
    # if we have a classification then find the matching intent tag
    if results:
        # loop as long as there are matches to process
        while results:
            for i in intents['intents']:
                # find a tag matching the first result
                if i['tag'] == results[0][0]:
                    # set context for this intent if necessary
                    if 'context' in i:
                        if show_details: print ('context:', i['context'])
                        context[userID] = i['context']

                    # check if this intent is contextual and applies to this user's conversation
                    if not 'context_filter' in i or \
                        (userID in context and 'context_filter' in i and i['context_filter'] == context[userID]):
                        if show_details: print ('tag:', i['tag'])
                        # a random response from the intent
                        return print(random.choice(i['responses']))

            results.pop(0)

def whattoadd(text):
  output=[]
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

def whattodelete(text):
  output=[]
  ignore2=['of','me','with']
  nlp=spacy.load("en_core_web_sm")
  modifiedtext=nltk.word_tokenize(text)
  modifiedtext = [w.lower() for w in modifiedtext if w not in ignore2]
  modifiedtext=TreebankWordDetokenizer().detokenize(modifiedtext)
  doc=nlp(modifiedtext)
  for chunk in doc.noun_chunks:
    if chunk.root.dep_== 'dobj':
      output.append(chunk.text)
      print(chunk.text,chunk.root.text,chunk.root.dep_,chunk.root.head.text)
  return output

with open('intents.json') as json_data:
    intents = json.load(json_data)

import pickle
data = pickle.load( open( "training_data", "rb" ) )
words = data['words']
classes = data['classes']
train_x = data['train_x']
train_y = data['train_y']

tf.reset_default_graph()
net = tflearn.input_data(shape=[None, len(train_x[0])])
net = tflearn.fully_connected(net, 10)
net = tflearn.fully_connected(net, 10)
net = tflearn.fully_connected(net, len(train_y[0]), activation='softmax')
net = tflearn.regression(net)
model = tflearn.DNN(net, tensorboard_dir='tflearn_logs')
model.load('./model.tflearn')

text="delete 2 kilo red dresses to my cart"
print(classify(text))
if(classify(text)[0][0]=='add'):
  print("Adding item")
  print(whattoadd(text))
elif(classify(text)[0][0]=='delete'):
  print("deleting item")
  print(whattodelete(text))

