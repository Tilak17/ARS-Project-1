#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Nov  1 22:47:06 2018
Train:
@author: TILAK_PUROHIT
"""
import pandas as pd
import numpy as np
from sklearn import mixture
import pickle
import os

def train(directory,filename, components, energy_coff):
    train_data=os.path.join(directory,filename)
    train_feature= pd.read_hdf(train_data)
    train_feature.head()
    X = np.array(train_feature["features"].tolist())
    y_train = np.array(train_feature["labels"].tolist())
    
    if(energy_coff==0):
        X_train =X [:,np.arange(X.shape[1])%13!=0]
    elif(energy_coff==1):
        X_train=X
    
    unique_phones = list(set(y_train))
    print("total unique phones in train dataset=",len(unique_phones))
    GMM_models = {}
    for ph in unique_phones:
        ph_gmm = mixture.GaussianMixture(n_components=components,covariance_type='diag')
        ph_gmm.fit(X_train[y_train==ph])
        GMM_models[ph]=ph_gmm
    print(len(GMM_models))
    with open(directory+"_GMM_"+components+".pkl","wb") as handle2:
        pickle.dump(GMM_models,handle2)
    handle2.close()    

################## EXAMPLES##################################################
#mfcc

directory="/Users/TILAK_PUROHIT/Desktop/features/mfcc/"
file= "timit_train.hdf"
print("mfcc with components="+ 64 +"and without energy cofficient")
train(directory,file,64,0)


#mfcc-delta
directory="/Users/TILAK_PUROHIT/Desktop/features/mfcc-delta/"
file= "timit_train.hdf"
print("mfcc with components="+ 64 +"and with energy cofficient")
train(directory,file,64,1)

#mfcc-delta-delta
directory="/Users/TILAK_PUROHIT/Desktop/features/mfcc-delta-delta/"
file= "timit_train.hdf"
print("mfcc with components="+ 64 +"and with energy cofficient")
train(directory,file,64,1)
