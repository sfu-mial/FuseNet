
from numpy import genfromtxt
import numpy as np
from Data_utils import *
import os
import glob
import csv
import pandas as pd
from numpy import *
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


def load_data():

     noise_factor = 5
     direc= '/local-scratch/Hanene/Data/multi-freq/Data/'
     print (direc)
     train_dirc='sizevsdepth'

     path1 = direc+train_dirc+'/'+'benign/absmat' 
     immatrix1= loadimage(path1)

     immatrix= immatrix1

     #path load image label
     path1 = direc+train_dirc+'/'+'benign/label' #path of folder to save images
     label1= loadmeasure(path1)

    #  path2 =   direc+train_dirc+'/'+'malignant/label'  #path of folder to save images
    #  label2= loadmeasure(path2)


     label=label1# np.concatenate((label1,label2), axis=0)


     #750 measure
     path1 = direc+train_dirc+'/'+'benign/750/csv'#measures.csv'  #path of folder to save images
     measure1=loadmeasure(path1)

    #  path2 =   direc+train_dirc+'/'+'malignant/750/csv'  #path of folder to save images
    #  measure2=loadmeasure(path2)


  
     measure_750= measure1 #np.concatenate((measure1,measure2), axis=0)


     #690 measure
     path1 = direc+train_dirc+'/'+'benign/690/csv'#measures.csv'  #path of folder to save images
     measure1=loadmeasure(path1)

    #  path2 =   direc+train_dirc+'/'+'malignant/690/csv'  #path of folder to save images
    #  measure2=loadmeasure(path2)


  
     measure_690= measure1#np.concatenate((measure1,measure2), axis=0)

     #800 measure
     path1 = direc+train_dirc+'/'+'benign/800/csv'#measures.csv'  #path of folder to save images
     measure1=loadmeasure(path1)

    #  path2 =   direc+train_dirc+'/'+'malignant/800/csv'  #path of folder to save images
    #  measure2=loadmeasure(path2)
  
     measure_800= measure1#np.concatenate((measure1,measure2), axis=0)

     #850 measure
     path1 = direc+train_dirc+'/'+'benign/850/csv'#measures.csv'  #path of folder to save images
     measure1=loadmeasure(path1)

    #  path2 =   direc+train_dirc+'/'+'malignant/850/csv'  #path of folder to save images
    #  measure2=loadmeasure(path2)

     measure_850= measure1#np.concatenate((measure1,measure2), axis=0)

    #  X_train,y_train = shuffle(measure,100*immatrix, random_state=2) #np.
     X_train_690,X_train_750,X_train_800,X_train_850,y_train,Y_label = measure_690,measure_750,measure_800,measure_850, 100*immatrix, label

    
     return X_train_690,X_train_750,X_train_800,X_train_850,y_train,Y_label


