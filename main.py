# import streamlit as st
# import pandas as pd
# from sklearn.model_selection import train_test_split
# from sklearn.preprocessing import  LabelEncoder
# import xgboost as xgb
# import numpy as np
# import cv2
# from PIL import Image

# st.header("Test bài tập lớn Tìm kiếm thông tin thị giác -Đại học UIT ")
# name=st.text_input("Enter your Name: ", key="name")
# data = pd.read_csv("https://raw.githubusercontent.com/gurokeretcha/WishWeightPredictionApplication/master/Fish.csv")
# #load label encoder
# encoder = LabelEncoder()
# encoder.classes_ = np.load('classes.npy',allow_pickle=True)

# # load model
# best_xgboost_model = xgb.XGBRegressor()
# best_xgboost_model.load_model("best_model.json")

# if st.checkbox('Show Training Dataframe'):
#     data

# st.subheader("Please select relevant features of your fish!")
# left_column, right_column = st.columns(2)
# with left_column:
#     inp_species = st.radio(
#         'Name of the fish:',
#         np.unique(data['Species']))


# input_Length1 = st.slider('Vertical length(cm)', 0.0, max(data["Length1"]), 1.0)
# input_Length2 = st.slider('Diagonal length(cm)', 0.0, max(data["Length2"]), 1.0)
# input_Length3 = st.slider('Cross length(cm)', 0.0, max(data["Length3"]), 1.0)
# input_Height = st.slider('Height(cm)', 0.0, max(data["Height"]), 1.0)
# input_Width = st.slider('Diagonal width(cm)', 0.0, max(data["Width"]), 1.0)


# if st.button('Make Prediction'):
#     input_species = encoder.transform(np.expand_dims(inp_species, -1))
#     inputs = np.expand_dims(
#         [int(input_species), input_Length1, input_Length2, input_Length3, input_Height, input_Width], 0)
#     prediction = best_xgboost_model.predict(inputs)
#     print("final pred", np.squeeze(prediction, -1))
#     st.write(f"Your fish weight is: {np.squeeze(prediction, -1):.2f}g")

#     st.write(f"Thank you {st.session_state.name}! I hope you liked it.")
#     st.write(f"If you want to see more advanced applications you can follow me on [medium](https://medium.com/@gkeretchashvili)")
#     st.write(f"Hello{name}")


# image = Image.open('egypt.jpg')
# st.image(image, caption='Egypt Picture')

# uploaded_file = st.file_uploader("Choose an Image File", accept_multiple_files=False)
# if uploaded_file is not None:
#     file_bytes = np.asarray(bytearray(uploaded_file.read()), dtype=np.uint8)
#     opencv_image = cv2.imdecode(file_bytes, 1)
#     st.image(opencv_image, channels="BGR")



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

image = Image.open('query_images/egypt.jpg')
st.image(image, caption='Egypt Picture')

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
    query = cv2.imread("egypt.jpg")

    model = VGGNet()

    queryVec = model.extract_feat("egypt.jpg")
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
        image = Image.open("database" +"/"+str(im, 'utf-8'))
        st.image(image, caption='Egypt Picture')    
st.write(f"Hello1  end")
    
    



      
        
#run
# python3 search.py --query ../query_images/results_pyramids.jpg --class color 

