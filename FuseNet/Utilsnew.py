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
    #  #path load GT image
    #  train_dirc='trainset'
    #  path1 = direc+train_dirc+'/'+'benign/absmat' #path of folder to save images
    #  immatrix1= loadimage(path1)

    #  path2 =   direc+train_dirc+'/'+'malignant/absmat'  #path of folder to save images
    #  immatrix2= loadimage(path2)

     train_dirc2='trainsetsmalloneblob'
     path1 = direc+train_dirc2+'/'+'benign/absmat' #path of folder to save images
     immatrix3= loadimage(path1)

     path2 =   direc+train_dirc2+'/'+'malignant/absmat'  #path of folder to save images
     immatrix4= loadimage(path2)

    #  immatrix= np.concatenate((immatrix1,immatrix2), axis=0)
    #  immatrix= np.concatenate((immatrix,immatrix3), axis=0)
    #  immatrix= 100*np.concatenate((immatrix,immatrix4), axis=0)
     immatrix= 100*np.concatenate((immatrix3,immatrix4), axis=0)

    #  #path load image label
    #  path1 = direc+train_dirc+'/'+'benign/label' #path of folder to save images
    #  label1= loadmeasure(path1)

    #  path2 =   direc+train_dirc+'/'+'malignant/label'  #path of folder to save images
    #  label2= loadmeasure(path2)

     train_dirc2='trainsetsmalloneblob'
     path1 = direc+train_dirc2+'/'+'benign/label' #path of folder to save images
     label3= loadmeasure(path1)

     path2 =   direc+train_dirc2+'/'+'malignant/label'  #path of folder to save images
     label4= loadmeasure(path2)

    #  label= np.concatenate((label1,label2), axis=0)
    #  label= np.concatenate((label,label3), axis=0)
    #  label= np.concatenate((label,label4), axis=0)

     label= np.concatenate((label3,label4), axis=0)

    #  #750 measure
    #  path1 = direc+train_dirc+'/'+'benign/750/csv'#measures.csv'  #path of folder to save images
    #  measure1=loadmeasure(path1)

    #  path2 =   direc+train_dirc+'/'+'malignant/750/csv'  #path of folder to save images
    #  measure2=loadmeasure(path2)

     path3 =  direc+train_dirc2+'/'+'benign/750/csv'  #path of folder to save images
     measure3=loadmeasure(path3)

     path4 =  direc+train_dirc2+'/'+'malignant/750/csv'  #path of folder to save images
     measure4=loadmeasure(path4)
  
    #  measure= np.concatenate((measure1,measure2), axis=0)
    #  measure= np.concatenate((measure,measure3), axis=0)
    #  measure_750= np.concatenate((measure,measure4), axis=0)
     measure_750= np.concatenate((measure3,measure4), axis=0)

     #690 measure
    #  path1 = direc+train_dirc+'/'+'benign/690/csv'#measures.csv'  #path of folder to save images
    #  measure1=loadmeasure(path1)

    #  path2 =   direc+train_dirc+'/'+'malignant/690/csv'  #path of folder to save images
    #  measure2=loadmeasure(path2)

     path3 =  direc+train_dirc2+'/'+'benign/690/csv'  #path of folder to save images
     measure3=loadmeasure(path3)

     path4 =  direc+train_dirc2+'/'+'malignant/690/csv'  #path of folder to save images
     measure4=loadmeasure(path4)
  
    #  measure= np.concatenate((measure1,measure2), axis=0)
    #  measure= np.concatenate((measure,measure3), axis=0)
    #  measure_690= np.concatenate((measure,measure4), axis=0)
     measure_690= np.concatenate((measure3,measure4), axis=0)
     #800 measure
    #  path1 = direc+train_dirc+'/'+'benign/800/csv'#measures.csv'  #path of folder to save images
    #  measure1=loadmeasure(path1)

    #  path2 =   direc+train_dirc+'/'+'malignant/800/csv'  #path of folder to save images
    #  measure2=loadmeasure(path2)

     path3 =  direc+train_dirc2+'/'+'benign/800/csv'  #path of folder to save images
     measure3=loadmeasure(path3)

     path4 =  direc+train_dirc2+'/'+'malignant/800/csv'  #path of folder to save images
     measure4=loadmeasure(path4)
  
    #  measure= np.concatenate((measure1,measure2), axis=0)
    #  measure= np.concatenate((measure,measure3), axis=0)
    #  measure_800= np.concatenate((measure,measure4), axis=0)   
     measure_800= np.concatenate((measure3,measure4), axis=0)   

     #850 measure
    #  path1 = direc+train_dirc+'/'+'benign/850/csv'#measures.csv'  #path of folder to save images
    #  measure1=loadmeasure(path1)

    #  path2 =   direc+train_dirc+'/'+'malignant/850/csv'  #path of folder to save images
    #  measure2=loadmeasure(path2)

     train_dirc2='trainsetsmalloneblob'
     path3 =  direc+train_dirc2+'/'+'benign/850/csv'  #path of folder to save images
     measure3=loadmeasure(path3)

     path4 =  direc+train_dirc2+'/'+'malignant/850/csv'  #path of folder to save images
     measure4=loadmeasure(path4)
  
    #  measure= np.concatenate((measure1,measure2), axis=0)
    #  measure= np.concatenate((measure,measure3), axis=0)
    #  measure_850= np.concatenate((measure,measure4), axis=0)   
     measure_850= np.concatenate((measure3,measure4), axis=0)   

#TESTNSET
     train_dirc='testset'
     #path load GT image
     path1 = direc+train_dirc+'/'+'benign/absmat' #path of folder to save images
     immatrix1= loadimage(path1)

     path2 =   direc+train_dirc+'/'+'malignant/absmat'  #path of folder to save images
     immatrix2= loadimage(path2)

     immatrix_test= 100*np.concatenate((immatrix1,immatrix2), axis=0)
    #  immatrix_test= 100*(immatrix2)

     #path load  image label
     path1 = direc+train_dirc+'/'+'benign/label' #path of folder to save images
     test_label1= loadmeasure(path1)

     path2 =   direc+train_dirc+'/'+'malignant/label'  #path of folder to save images
     test_label2= loadmeasure(path2)

     label_test= np.concatenate((test_label1,test_label2), axis=0)
    #  label_test= test_label2

     #750 measure
     path1 = direc+train_dirc+'/'+'benign/750/csv'#measures.csv'  #path of folder to save images
     measure1=loadmeasure(path1)
     path1 = direc+train_dirc+'/'+'malignant/750/csv'#measures.csv'  #path of folder to save images
     measure2=loadmeasure(path1)
     testmeasure_750= np.concatenate((measure1,measure2), axis=0)   
    #  testmeasure_750= measure2

     #690 measure
     path1 = direc+train_dirc+'/'+'benign/690/csv'#measures.csv'  #path of folder to save images
     measure1=loadmeasure(path1)
     path1 = direc+train_dirc+'/'+'malignant/690/csv'#measures.csv'  #path of folder to save images
     measure2=loadmeasure(path1)
     testmeasure_690= np.concatenate((measure1,measure2), axis=0)   
    #  testmeasure_690= measure2

     #800 measure
     path1 = direc+train_dirc+'/'+'benign/800/csv'#measures.csv'  #path of folder to save images
     measure1=loadmeasure(path1)
     path1 = direc+train_dirc+'/'+'malignant/800/csv'#measures.csv'  #path of folder to save images
     measure2=loadmeasure(path1)
     testmeasure_800= np.concatenate((measure1,measure2), axis=0) 
    #  testmeasure_800= measure2
     #850 measure
     path1 = direc+train_dirc+'/'+'benign/850/csv'#measures.csv'  #path of folder to save images
     measure1=loadmeasure(path1)
     path1 = direc+train_dirc+'/'+'malignant/850/csv'#measures.csv'  #path of folder to save images
     measure2=loadmeasure(path1)
     testmeasure_850= np.concatenate((measure1,measure2), axis=0) 
    #  testmeasure_850= measure2

     label=np.where((label)==2, 1, label)
     label_test=np.where((label_test)==2, 1, label_test)

     label=np.where((label)==3, 2, label)
     label_test=np.where((label_test)==3, 2, label_test)


    #  X_train,y_train = shuffle(measure,100*immatrix, random_state=2) #np.
     X_train_690,X_train_750,X_train_800,X_train_850,y_train,Y_label = shuffle(measure_690,measure_750,measure_800,measure_850, immatrix, label, random_state=2) 

    #  X_test_690,X_test_750,X_test_800,X_test_850,y_test,Y_testlabel = shuffle(testmeasure_690,testmeasure_750,testmeasure_800,
    #  testmeasure_850,immatrix_test ,label_test, random_state=2) 
     X_test_690,X_test_750,X_test_800,X_test_850,y_test,Y_testlabel =(testmeasure_690,testmeasure_750,testmeasure_800,
     testmeasure_850,immatrix_test ,label_test) 

    # #  #revierse 
    #  #np.fliplr(immatrix)
    #  reverse_y_train=y_train[:,:,::-1] 

    #  y_train= np.concatenate((y_train,reverse_y_train), axis=0)


    #  reverse_X_train_690=X_train_690[...,::-1] 
    #  X_train_690= np.concatenate((X_train_690,reverse_X_train_690), axis=0)

    #  reverse_X_train_750=X_train_750[...,::-1] 
    #  X_train_750= np.concatenate((X_train_750,reverse_X_train_750), axis=0)

    #  reverse_X_train_800=X_train_800[...,::-1] 
    #  X_train_800= np.concatenate((X_train_800,reverse_X_train_800), axis=0)

    #  reverse_X_train_850=X_train_850[...,::-1] 
    #  X_train_850= np.concatenate((X_train_850,reverse_X_train_850), axis=0)

    #  Y_label= np.concatenate((Y_label,Y_label), axis=0)

     return X_train_690,X_train_750,X_train_800,X_train_850,y_train,Y_label,X_test_690,X_test_750,X_test_800,X_test_850,y_test,Y_testlabel
    #  return measure_690,measure_750,measure_800,measure_850, immatrix, label, testmeasure_690,testmeasure_750,testmeasure_800,testmeasure_850,immatrix_test ,label_test
    #  X_train, y_train, X_test,y_test

