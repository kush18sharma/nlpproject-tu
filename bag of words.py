# -*- coding: utf-8 -*-
"""
Created on Tue May  8 13:28:38 2018

@author: kushagra sharma
"""


import os
import gensim
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.ensemble import RandomForestClassifier
from KaggleWord2VecUtility import KaggleWord2VecUtility
import pandas as pd
import numpy as np

if __name__ == '__main__':
    train = pd.read_csv(os.path.join(os.path.dirname(__file__), 'data', 'labeledTrainData.tsv'), header=0, \
                    delimiter="\t", quoting=3)
    test = pd.read_csv(os.path.join(os.path.dirname(__file__), 'data', 'testData.tsv'), header=0, delimiter="\t", \
                   quoting=3 )

    print ('The first review is:')
    print (train["review"][0])

    
    input("Press Enter to continue...")


   
    
    clean_train_reviews = []

   

    print ("Cleaning and parsing the training set movie reviews...\n")
    for i in range( 0, len(train["review"])):
        clean_train_reviews.append(" ".join(KaggleWord2VecUtility.review_to_wordlist(train["review"][i], True)))


    
    print ("Creating the bag of words...\n")


    
    vectorizer = CountVectorizer(analyzer = "word",   \
                             tokenizer = None,    \
                             preprocessor = None, \
                             stop_words = None,   \
                             max_features = 5000)

    
    train_data_features = vectorizer.fit_transform(clean_train_reviews)

   
    np.asarray(train_data_features)

   
    print( "Training the random forest (this may take a while)...")


    
    forest = RandomForestClassifier(n_estimators = 100)

    
    forest = forest.fit( train_data_features, train["sentiment"] )



  
    clean_test_reviews = []

    print ("Cleaning and parsing the test set movie reviews...\n")
    for i in range(0,len(test["review"])):
        clean_test_reviews.append(" ".join(KaggleWord2VecUtility.review_to_wordlist(test["review"][i], True)))

    
    test_data_features = vectorizer.transform(clean_test_reviews)
    np.asarray(test_data_features)

    
    print ("Predicting test labels...\n")
    result = forest.predict(test_data_features)

    
    output = pd.DataFrame( data={"id":test["id"], "sentiment":result} )

    
    output.to_csv(os.path.join(os.path.dirname(__file__), 'data', 'Bag_of_Words_model.csv'), index=False, quoting=3)
    print ("Wrote results to Bag_of_Words_model.csv")

