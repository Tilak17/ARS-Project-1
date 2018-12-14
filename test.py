#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Nov  1 23:29:06 2018
test
@author: TILAK_PUROHIT
"""
import pandas as pd
import numpy as np
#from sklearn 
import pickle
import os

unique_phones = np.array(['', 'aa', 'ae', 'ah', 'aw', 'ay', 'b', 'ch', 'd', 'dh', 'dx', 'eh', 'er', 'ey', 'f', 'g', 'hh', 'ih', 'iy', 'jh', 'k', 'l', 'm', 'n', 'ng', 'ow', 'oy', 'p', 'r', 's', 'sh', 'sil', 't', 'th', 'uh', 'uw', 'v', 'w', 'y', 'z'])

def test(directory,filename, gmm_path, energy_coff):

    
    test_data=os.path.join(directory,filename)
    test_feature= pd.read_hdf(test_data)
    test_feature.head()
    X = np.array(test_feature["features"].tolist())
    y_test = np.array(test_feature["labels"].tolist())
       
    
    if(energy_coff==0):
        X_test =X[:,np.arange(X.shape[1])%13!=0]
    elif(energy_coff==1):
        X_test=X 
    N = X_test.shape
    print(N)
    
    with open(gmm_path,'rb') as G:
        GMM_models = pickle.load(G)    
    
    l= int(len(unique_phones))
    F= test_feature.shape[0]
    scores = np.zeros((l,F))
    print(scores.shape)
    for i in range(l):
        scores[i] = np.array(GMM_models[unique_phones[i]].score_samples(X_test))
    
    predicted_phoneme_index = np.argmax(scores,axis=0)
    predicted_phoneme_index=list(predicted_phoneme_index)
    predicted_phoneme=[]
    c=int(len(predicted_phoneme_index))
    for i in range(0,c):
        predicted_phoneme.append(unique_phones[predicted_phoneme_index[i]])
    predicted_phoneme=np.array(predicted_phoneme) 
    a=np.sum(predicted_phoneme==y_test)/y_test.size * 100
    print("accuracy=",a)
    
################## EXAMPLES##################################################
#mfcc

directory="/Users/TILAK_PUROHIT/Desktop/features/mfcc/"
file= "timit_test.hdf"
gmm_path="/Users/TILAK_PUROHIT/Desktop/features/mfcc/_GMM_64.pkl"
print("mfcc with components= 64 and without energy cofficient")
test(directory,file,gmm_path,0)


