#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Mon Dec 06 12:15:16 2017

@author: Haneneby
"""

from numpy import genfromtxt
import numpy as np

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

def readcomplex(strng):
	strng = strng.replace(" ", '')
	strng = strng.replace("i", 'j')
	c = 0
	try:
		c = complex(strng)
	except ValueError as e:
		print("Exception on input")
		print(strng)
	return c
def norm(image):
    norm_image = cv2.normalize(image, alpha=0, beta=1, norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_32F)
    return norm_image

def sortfiles(listfiles):
    return [ x for (_,x) in sorted( ( (int(stri[stri.find('-')+1:stri.find('.')]),stri) for stri in listfiles )

                                   ) ]

def myfunction( x,  noise_factor = 5):
    #print (x.shape)
    noise =  noise_factor * np.random.normal(loc=0.0, scale=10, size=x.shape) #+np.random.randint(1000,4000)
    #print (noise.shape)

    return x +noise

def copyandunzip (filename):  
    
#    dir1=  '/cs/ghassan2/students_less/hbenyedd/data/'
    dir1= '/local-scratch/Hanene/Data/'
    copy2(dir1+ filename+'.zip', '/dev/shm/hanenby')
    with zipfile.ZipFile("/dev/shm/hanenby/"+ filename +'.zip',"r") as zip_ref:
        zip_ref.extractall("/dev/shm/hanenby/")
    print (os.path.exists('/dev/shm/hanenby/'+filename))            
    return

def loadmeasure(path_to_parent):
##    measlist = sortfiles(os.listdir(path_to_parent))
##    my_data = (np.vectorize(lambda t:readcomplex(t))(genfromtxt(os.path.join(path_to_parent, csv_file), delimiter=',',dtype='str')) for csv_file in measlist)
#    my_data = np.vectorize(lambda t:readcomplex(t)) (genfromtxt(path_to_parent+'/'+'measures.csv', delimiter=',',dtype='str'))
#    #print my_data.shape
#    my_data = np.absolute(my_data)
#    measure =my_data.transpose()
    measlist = sortfiles(os.listdir(path_to_parent))
    my_data = (np.vectorize(lambda t:readcomplex(t))(genfromtxt(os.path.join(path_to_parent, csv_file), delimiter=',',dtype='str')) for csv_file in measlist)
    #my_data= np.vectorize(np.genfromtxt(path3+'/' + im2, converters={0: lambda x: x.replace('i','j')},delimiter=',', dtype=str)for im2 in measlist)
    measure = np.absolute(list(my_data))
    #measure =measure.transpose()

    #measure = np.absolute(list(my_data))
    #measure =measure.transpose()
    print (measure.shape)
    return  measure
def loadimage(path):
    imlist = removedotfile(path)
    immatrix= array([(np.around(genfromtxt(os.path.join(path, im), delimiter=',',dtype='float'),decimals=4))#.flatten()
                   for im in imlist],'f') #'%.2f' % elem for elem in myList ]
    print (immatrix.shape)
    return immatrix #[(i, immatrix[i, :, :].copy()) for i in range(immatrix.shape[0]) if immatrix[i, :, :].max() >= 0.09]
def postprocess(x,y):
    z=[]
    t=[]
    j=0
    for i in range(x.shape[0]):
        if (x[i, :, :].max() >= 0.09) or abs((x[i, :, :].max()-(x[i, :, :].min()))<0.0001):
            l = len(z)
            #print('appending')
            z.append(x[i, :, :])
            L = len(z)
            assert(l < L)
            t.append(y[i, :])
            j+=1
    zz = np.array(z)
    tt = np.array(t)
    print (x.shape)
    print (y.shape)
    print (zz.shape)
    print (tt.shape)
    return zz,tt

def nean_std_data(x):
    mvec = x.mean(0)
    stdvec = x.std(axis=0) 
    return mvec, stdvec#,mvec,stdvec
def loadrealmeas(path):
     data = pd.read_csv(path, header=None)
     values = data.values[:, :]
     rmeas = np.asmatrix(values, dtype='float64')    
     return rmeas
def removedotfile (path):
    listing = (os.listdir(path))
    i=0
    while((listing[i]=='.') or (listing[i]=='..') or (listing[i]=='.DS_Store' )):
        i=i+1
    imlist = sortfiles(listing[i:])
    return imlist



def load_data():

     noise_factor = 5
     direc= '/local-scratch/Hanene/Data/multi-freq/Data/'#/cs/ghassan2/students_less/hbenyedd/data'
     print (direc)
#TRAINSET
     #path load GT image
    #  train_dirc='localization_effect/testset_var_depth'
     train_dirc='sizevsdepth'

     path1 = direc+train_dirc+'/'+'benign/absmat' #benigns
     immatrix1= loadimage(path1)

    #  path2 =   direc+train_dirc+'/'+'malignant/absmat'  #path of folder to save images
    #  immatrix2= loadimage(path2)


     immatrix= immatrix1 #np.concatenate((immatrix1,immatrix2), axis=0)

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


