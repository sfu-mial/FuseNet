
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
import shutil
from shutil import rmtree,copyfile,copy2
import zipfile
import logging
FORMAT = "[%(filename)s:%(lineno)s - %(funcName)20s() ] %(message)s"
logging.basicConfig(format=FORMAT)
lgr = logging.getLogger('global')
lgr.setLevel(logging.INFO)
from sklearn.preprocessing import label_binarize


def load_data_t(direc):

     # direc= '/local-scratch/Hanene/Data/multi-freq/Data/'
     print (direc)
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

     label_test=np.where((label_test)==2, 1, label_test)

     label_test=np.where((label_test)==3, 2, label_test)


     X_test_690,X_test_750,X_test_800,X_test_850,y_test,Y_testlabel =(testmeasure_690,testmeasure_750,testmeasure_800,
     testmeasure_850,immatrix_test ,label_test) 

     return preprocess_t(X_test_690,X_test_750,X_test_800,X_test_850,y_test,Y_testlabel)



def preprocess_t(X_test_690,X_test_750,X_test_800,X_test_850,y_test,Y_testlabel):
 
     # y_train= immatrix
     x_test_2= X_test_690
     x_test_1= X_test_750
     x_test_3= X_test_800
     x_test_4= X_test_850
     # x_train= measure_750

     y_testlabel=Y_testlabel
     y_testima= y_test

     # normalize data
     x_test_1 = normalize_data(X_test_690) 

     x_test_2= normalize_data(X_test_750) 

     x_test_3 = normalize_data(X_test_800) 

     x_test_4= normalize_data(X_test_850) 

     y_test = np.reshape(y_testima, (len(y_testima), 128, 128,1))  #

     y_testlabel=label_binarize(Y_testlabel, classes=[0, 1, 2])



     return x_test_1, x_test_2, x_test_3, x_test_4, y_test, y_testlabel
