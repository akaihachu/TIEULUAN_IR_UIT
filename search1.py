import streamlit as st
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import  LabelEncoder
import xgboost as xgb
from PIL import Image 

import cv2
import numpy as np



import h5py

import matplotlib.pyplot as plt
import matplotlib.image as mpimg

from vgg16.vgg16 import VGGNet
from searcher import Searcher
# image = Image.open('egypt.jpg')
# st.image(image, caption='Egypt Picture')

uploaded_file = st.file_uploader("Choose an Image File", accept_multiple_files=False)
if uploaded_file is not None:
    file_bytes = np.asarray(bytearray(uploaded_file.read()), dtype=np.uint8)
    opencv_image = cv2.imdecode(file_bytes, 1)
    st.image(opencv_image, channels="BGR")
    


    st.write(f"Hello1")
    h5f = h5py.File("vgg16/index.h5",'r')
    feats = h5f['dataset_1'][:]
	#print(feats)
    imgNames = h5f['dataset_2'][:]
#print(imgNames)
    h5f.close()   
    query = cv2.imread("../query_images/tajmahal.jpg")

    model = VGGNet()

    queryVec = model.extract_feat("../query_images/tajmahal.jpg")
	# dot product between two vectors can be used as aggregate for similarity as the projection of vector u on vector v (u^T.v) is considered as similar 
	# when the angle between them is 0 degrees. Therefore, more is the resultant of their product implies more is the similarity b/n them
    scores = np.dot(queryVec, feats.T)
	#print(scores)
    rank_ID = np.argsort(scores)[::-1]
	#print(rank_ID)
    rank_score = scores[rank_ID]
    st.write(f"Hello2")

# myWindow = cv2.resize(query,(960,960))
# cv2.imshow("query",myWindow)

    maxres = 10
    imlist = [imgNames[index] for i,index in enumerate(rank_ID[0:maxres])]
    st.write("top %d images in order are: " %maxres)

# 	# show top #maxres retrieved result one by one
    for i,im in enumerate(imlist):
        st.write(i)
        image = Image.open("../" +  "database" +"/"+str(im, 'utf-8'))
        st.image(image, caption='Egypt Picture')
    
st.write(f"Hello1  end")
    
    



        

        
#run
# python3 search.py --query ../query_images/results_pyramids.jpg --class color 
