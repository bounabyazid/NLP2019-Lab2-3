#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Nov 11 11:44:22 2019
Lab 2-3
@author: Yazid Bounab
"""
#https://artint.info/
#https://pade.readthedocs.io/en/latest/

#https://stevenloria.com/wordnet-tutorial/
#https://medium.com/parrot-prediction/dive-into-wordnet-with-nltk-b313c480e788
import gensim
from gensim.models import Word2Vec
from nltk import word_tokenize
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer
from nltk.corpus import wordnet as wn
from nltk.corpus import brown

from scipy.spatial.distance import cosine

from sklearn.feature_extraction.text import TfidfVectorizer

def synsets(word,pos):
    #for synset in wn.synsets(word, pos):
    #    print(synset)
    #print(wn.synsets(word, pos))
    return wn.synsets(word, pos)

def hyponyms():
    vehicle = wn.synset('vehicle.n.01')
    typesOfVehicles = list(set([w for s in vehicle.closure(lambda s:s.hyponyms()) for w in s.lemma_names()]))
    print(typesOfVehicles)
    
def hypernyms():
    return True

def RankedSynset(word,pos):
    ranked = {}
    
    for synset in wn.synsets(word,pos):
        ranked[synset] = synset.lemmas()[0].count()
    print(ranked)
    return ranked

def MehalceaSimilarity():
    T1 = 'Students feel unhappy today about the class'
    T2 = 'Many students struggled to understand some key concepts about the subject seen in the class'
   
    vectoriser=TfidfVectorizer(stop_words='english')
    tfidf = vectoriser.fit_transform([T1,T2])

    #stopword removal Stemming
    stop_words = stopwords.words('english')
    stemmer = PorterStemmer()

    words1 = [stemmer.stem(w) for w in word_tokenize(T1) if w not in stop_words]
    words2 = [stemmer.stem(w) for w in word_tokenize(T2) if w not in stop_words]
       
    return words1,words2,tfidf

def Simplified_Lesk_Algorithm_Desimbiguation(word, pos, Sentence):
    
    Senses = {}
    Examples = {}
    Definitions = {}
    
    Sysnsets = synsets(word,pos)
       
    stemmer = PorterStemmer()
    stop_words = stopwords.words('english')
    Tokens = [stemmer.stem(w) for w in word_tokenize(Sentence) if w not in stop_words]
        
    for synset in Sysnsets:
        i = 0
        Synset_Examples = {}
        for example in synset.examples():
            Example_Tokens = [stemmer.stem(w) for w in word_tokenize(example) if w not in stop_words]
            Synset_Examples[i] = (len(set(Example_Tokens) & set(Tokens)),Example_Tokens)
            i += 1
        Examples[synset.name()] = Synset_Examples
        Senses[synset.name()] = (len(set(synset.lemma_names()) & set(Tokens)),synset.lemma_names())
        Definition_Tokens = [stemmer.stem(w) for w in word_tokenize(synset.definition()) if w not in stop_words]

        Definitions[synset.name()] = (len(set(Definition_Tokens) & set(Tokens)),Definition_Tokens)
        
    return Senses,Examples,Definitions

def Best_Sense_Desimbiguation(word, pos, Sentence):
    print('Loading Google Pre-trained model ...')
    model = gensim.models.KeyedVectors.load_word2vec_format('/home/polo/Downloads/GoogleNews-vectors-negative300.bin', binary=True)
    print('Google Pre-trained model has been loaded')
    WV = model.wv
    Vec = WV[word]
        
    print('Loading Brown Corpus ...')
    b = Word2Vec(brown.sents())
    print('Brown Corpus has been loaded')

    Senses = [i[0] for i in b.most_similar(word,topn=5)]
    
    stop_words = stopwords.words('english')
    Tokens = [w for w in word_tokenize(Sentence) if w not in stop_words]

    CosW2Vec = {}
    for sense in Senses:
        CosW2Vec[sense] = {T:(1-cosine(Vec,WV[T])) for T in Tokens if not T==word}
    return CosW2Vec

#Senses1,Examples1,Definitions1 = Simplified_Lesk_Algorithm_Desimbiguation('plant','n', 'This plant needs to be watered each day.')
#Senses2,Examples2,Definitions2 = Simplified_Lesk_Algorithm_Desimbiguation('bank','n', 'The bank refused to give me a loan.')
CosW2Vec = Best_Sense_Desimbiguation('plant','n', 'This plant needs to be watered each day')
#ranked = RankedSynset('car', 'n')
#
#words1,words2,tfidf = MehalceaSimilarity()
    
    