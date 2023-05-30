
"""
@author: Haneneby
"""
from numpy import genfromtxt
import numpy as np
from Utils.Data_utils import *
import os
import glob
import csv
import pandas as pd
from numpy import *
from Utils.Utils_models import normalize_data
from sklearn.utils import shuffle
from sklearn.model_selection import train_test_split
#from pathlib import Path
import shutil
from shutil import rmtree,copyfile,copy2
import zipfile
import logging
FORMAT = "[%(filename)s:%(lineno)s - %(funcName)20s() ] %(message)s"
logging.basicConfig(format=FORMAT)
lgr = logging.getLogger('global')
lgr.setLevel(logging.INFO)
from sklearn.preprocessing import label_binarize


def load_data(direc):

     # direc= '/local-scratch/Hanene/Data/multi-freq/Data/'
     print (direc)
    #TRAINSET

     train_dirc2='trainsetsmalloneblob'
     path1 = direc+train_dirc2+'/'+'benign/absmat' 
     immatrix3= loadimage(path1)

     path2 =   direc+train_dirc2+'/'+'malignant/absmat'  
     immatrix4= loadimage(path2)
     immatrix= 100*np.concatenate((immatrix3,immatrix4), axis=0)


     train_dirc2='trainsetsmalloneblob'
     path1 = direc+train_dirc2+'/'+'benign/label' 
     label3= loadmeasure(path1)

     path2 =   direc+train_dirc2+'/'+'malignant/label'  
     label4= loadmeasure(path2)

     label= np.concatenate((label3,label4), axis=0)


     path3 =  direc+train_dirc2+'/'+'benign/750/csv'  
     measure3=loadmeasure(path3)

     path4 =  direc+train_dirc2+'/'+'malignant/750/csv'  
     measure4=loadmeasure(path4)
     measure_750= np.concatenate((measure3,measure4), axis=0)


     path3 =  direc+train_dirc2+'/'+'benign/690/csv' 
     measure3=loadmeasure(path3)

     path4 =  direc+train_dirc2+'/'+'malignant/690/csv' 
     measure4=loadmeasure(path4)
  
     measure_690= np.concatenate((measure3,measure4), axis=0)

     path3 =  direc+train_dirc2+'/'+'benign/800/csv'  
     measure3=loadmeasure(path3)

     path4 =  direc+train_dirc2+'/'+'malignant/800/csv' 
     measure4=loadmeasure(path4)
  
     measure_800= np.concatenate((measure3,measure4), axis=0)   


     train_dirc2='trainsetsmalloneblob'
     path3 =  direc+train_dirc2+'/'+'benign/850/csv'  
     measure3=loadmeasure(path3)

     path4 =  direc+train_dirc2+'/'+'malignant/850/csv' 
     measure4=loadmeasure(path4)
     measure_850= np.concatenate((measure3,measure4), axis=0)   

     #TESTNSET
     train_dirc='testset'
     #path load GT image
     path1 = direc+train_dirc+'/'+'benign/absmat' 
     immatrix1= loadimage(path1)

     path2 =   direc+train_dirc+'/'+'malignant/absmat' 
     immatrix2= loadimage(path2)

     immatrix_test= 100*np.concatenate((immatrix1,immatrix2), axis=0)
   

     #path load  image label
     path1 = direc+train_dirc+'/'+'benign/label'
     test_label1= loadmeasure(path1)

     path2 =   direc+train_dirc+'/'+'malignant/label' 
     test_label2= loadmeasure(path2)

     label_test= np.concatenate((test_label1,test_label2), axis=0)

     #750 measure
     path1 = direc+train_dirc+'/'+'benign/750/csv'
     measure1=loadmeasure(path1)
     path1 = direc+train_dirc+'/'+'malignant/750/csv'
     measure2=loadmeasure(path1)
     testmeasure_750= np.concatenate((measure1,measure2), axis=0)   

     #690 measure
     path1 = direc+train_dirc+'/'+'benign/690/csv'
     measure1=loadmeasure(path1)
     path1 = direc+train_dirc+'/'+'malignant/690/csv'
     measure2=loadmeasure(path1)
     testmeasure_690= np.concatenate((measure1,measure2), axis=0)   

     #800 measure
     path1 = direc+train_dirc+'/'+'benign/800/csv'
     measure1=loadmeasure(path1)
     path1 = direc+train_dirc+'/'+'malignant/800/csv'
     measure2=loadmeasure(path1)
     testmeasure_800= np.concatenate((measure1,measure2), axis=0) 

     #850 measure
     path1 = direc+train_dirc+'/'+'benign/850/csv'
     measure1=loadmeasure(path1)
     path1 = direc+train_dirc+'/'+'malignant/850/csv'
     measure2=loadmeasure(path1)
     testmeasure_850= np.concatenate((measure1,measure2), axis=0) 

     label=np.where((label)==2, 1, label)
     label_test=np.where((label_test)==2, 1, label_test)

     label=np.where((label)==3, 2, label)
     label_test=np.where((label_test)==3, 2, label_test)


     X_train_690,X_train_750,X_train_800,X_train_850,y_train,Y_label = shuffle(measure_690,measure_750,measure_800,measure_850, immatrix, label, random_state=2) 

     X_test_690,X_test_750,X_test_800,X_test_850,y_test,Y_testlabel =(testmeasure_690,testmeasure_750,testmeasure_800,
     testmeasure_850,immatrix_test ,label_test) 

     return preprocess(X_train_690,X_train_750,X_train_800,X_train_850,y_train,Y_label,X_test_690,X_test_750,X_test_800,X_test_850,y_test,Y_testlabel)


def preprocess(X_train_690,X_train_750,X_train_800,X_train_850,y_train,Y_label,X_test_690,X_test_750,X_test_800,X_test_850,y_test,Y_testlabel):
     # print ('here')
     x_train_1= X_train_690#+ np.random.normal( measure_1.mean()/3, measure_1.mean()/2, 256)
     x_train_2= X_train_750#+ np.random.normal( measure_2.mean()/3, measure_2.mean()/2, 256)
     x_train_3= X_train_800#+ np.random.normal( measure_3.mean()/3, measure_3.mean()/2, 256)
     x_train_4= X_train_850#+ np.random.normal( measure_4.mean()/3, measure_4.mean()/2, 256)

     # y_train= immatrix
     x_test_2= X_test_690#+ np.random.normal( testmeasure_2.mean()/3, testmeasure_2.mean()/2, 256)
     x_test_1= X_test_750#+ np.random.normal( testmeasure_1.mean()/3, testmeasure_1.mean()/2, 256)
     x_test_3= X_test_800#+ np.random.normal( testmeasure_3.mean()/3, testmeasure_3.mean()/2, 256)
     x_test_4= X_test_850#+ np.random.normal( testmeasure_4.mean()/3, testmeasure_4.mean()/2, 256)
     # x_train= measure_750
     y_train_label= Y_label
     y_trainima= y_train
     y_testlabel=Y_testlabel
     y_testima= y_test

     # normalize data
     x_train_1= normalize_data (X_train_690) #here
     x_test_1 = normalize_data(X_test_690) #here

     x_train_2= normalize_data (X_train_750) #here
     x_test_2= normalize_data(X_test_750) #here

     x_train_3= normalize_data (X_train_800) #here
     x_test_3 = normalize_data(X_test_800) #here

     x_train_4= normalize_data (X_train_850) #here
     x_test_4= normalize_data(X_test_850) #here

     y_train = np.reshape(y_trainima, (len(y_trainima), 128, 128,1))  #
     y_test = np.reshape(y_testima, (len(y_testima), 128, 128,1))  #

     y_label= label_binarize(Y_label, classes=[0, 1, 2])#np_utils.to_categorical(y_train, 3)

     y_testlabel=label_binarize(Y_testlabel, classes=[0, 1, 2])# #np_utils.to_categorical(y_test, 3)



     return x_train_1, x_train_2, x_train_3, x_train_4, y_train, y_label, x_test_1, x_test_2, x_test_3, x_test_4, y_test, y_testlabel
