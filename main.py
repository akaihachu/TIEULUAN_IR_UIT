import streamlit as st
import pandas as pd
# from sklearn.model_selection import train_test_split
# from sklearn.preprocessing import  LabelEncoder
import xgboost as xgb
from PIL import Image 
import cv2
import numpy as np


import h5py
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
from vgg16.vgg16 import VGGNet
from searcher import Searcher



from os import listdir
from math import ceil


uploaded_file = st.file_uploader("Choose an Image File", accept_multiple_files=False)



directory = r'images\bike'
files = listdir(directory)
def initialize():    
    df = pd.DataFrame({'file':files,
                    'incorrect':[False]*len(files),
                    'label':['']*len(files)})
    df.set_index('file', inplace=True)
    return df
    
def update (image, col): 
    df.at[image,col] = st.session_state[f'{col}_{image}']
    if st.session_state[f'incorrect_{image}'] == False:
       st.session_state[f'label_{image}'] = ''
       df.at[image,'label'] = ''    
    
    

if uploaded_file is not None:
    file_bytes = np.asarray(bytearray(uploaded_file.read()), dtype=np.uint8)
    opencv_image = cv2.imdecode(file_bytes, 1)
    st.image(opencv_image, channels="BGR")
    
    
    if 'df' not in st.session_state:
        df = initialize()
        st.session_state.df = df
    else:
        df = st.session_state.df 

    controls = st.columns(3)
    with controls[0]:
        batch_size1 = st.select_slider("Batch size:",range(4,20,4))
        batch_size=10
    with controls[1]:
        row_size = st.select_slider("Row size:", range(1,6), value = 5)
    num_batches = len(files)
    
    with controls[2]:
        page = st.selectbox("Page", range(1,num_batches+1))

    batch = files[(page-1)*batch_size : page*batch_size]
    
    grid = st.columns(row_size)
    col = 0
    for image in batch:
        with grid[col]:
            st.image(f'{directory}\{image}', caption='bike')
        col = (col + 1) % row_size
    
    
    
    h5f = h5py.File("vgg16/index.h5",'r')
    feats = h5f['dataset_1'][:]
    
	#print(feats)
    imgNames = h5f['dataset_2'][:]
    
#print(imgNames)
    h5f.close()   
    query = cv2.imread("egypt.jpg")

    model = VGGNet()

#     queryVec = model.extract_feat("egypt.jpg")
    queryVec = model.extract_feat_for_image(opencv_image)


	# dot product between two vectors can be used as aggregate for similarity as the projection of vector u on vector v (u^T.v) is considered as similar 
	# when the angle between them is 0 degrees. Therefore, more is the resultant of their product implies more is the similarity b/n them
    scores = np.dot(queryVec, feats.T)
	#print(scores)
    rank_ID = np.argsort(scores)[::-1]
	#print(rank_ID)
    rank_score = scores[rank_ID]
    st.write(f"Hello2")


    maxres = 10
    imlist = [imgNames[index] for i,index in enumerate(rank_ID[0:maxres])]
    st.write("top %d images in order are: " %maxres)

# 	# show top #maxres retrieved result one by one
    for i,im in enumerate(imlist):
        st.write(i)
        image = Image.open("database" +"/"+str(im, 'utf-8'))
        st.image(image, caption='Egypt Picture')    
st.write(f"Hello1  end")        
    
        
#run
# python3 search.py --query ../query_images/results_pyramids.jpg --class color 
