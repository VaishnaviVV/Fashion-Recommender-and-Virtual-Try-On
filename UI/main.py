import streamlit as st
import os
from PIL import Image
import numpy as np
import pickle
import tensorflow
from tensorflow.keras.preprocessing import image
from tensorflow.keras.layers import GlobalMaxPooling2D
from tensorflow.keras.applications.resnet50 import ResNet50,preprocess_input
from sklearn.neighbors import NearestNeighbors
from numpy.linalg import norm
import glob
import shutil
import os



feature_list = np.array(pickle.load(open('embeddings.pkl','rb')))
filenames = pickle.load(open('filenames.pkl','rb'))

model = ResNet50(weights='imagenet',include_top=False,input_shape=(224,224,3))
model.trainable = False

model = tensorflow.keras.Sequential([
    model,
    GlobalMaxPooling2D()
])

st.title('Fashion Recommendation System')


def save_uploaded_file(uploaded_file):
    try:
        with open(os.path.join('uploads',uploaded_file.name),'wb') as f:
            f.write(uploaded_file.getbuffer())
        return 1
    except:
        return 0

# def save_input_file(uploadedfile):
#     print(uploadedfile)
#     with open(os.path.join("input_images",uploadedfile),"wb") as f:
#         f.write(uploadedfile.getbuffer())
#     return st.success("Saved File:{} to input_images".format(uploadedfile))

dst_dir = "/Users/narendraomprakash/Desktop/Narendra/Semester-VI-WINTER2021/TARP/Datasets/input_images/"
# /Users/narendraomprakash/Desktop/Narendra/Semester-VI-WINTER2021/TARP/Datasets/img/Abstract_Asymmetrical_Hem_Top/img_00000045.jpg
def save_input_file(uploadedfile):
    newname="_".join(uploadedfile.split("/")[-2:])
    filename=uploadedfile.split("/")[-1]
    for jpgfile in glob.iglob(uploadedfile):
        shutil.copy(jpgfile, dst_dir)
    os.rename(dst_dir+filename,dst_dir+newname)

def feature_extraction(img_path,model):
    img = image.load_img(img_path, target_size=(224, 224))
    img_array = image.img_to_array(img)
    expanded_img_array = np.expand_dims(img_array, axis=0)
    preprocessed_img = preprocess_input(expanded_img_array)
    result = model.predict(preprocessed_img).flatten()
    normalized_result = result / norm(result)

    return normalized_result

def recommend(features,feature_list):
    neighbors = NearestNeighbors(n_neighbors=6, algorithm='brute', metric='euclidean')
    neighbors.fit(feature_list)

    distances, indices = neighbors.kneighbors([features])

    return indices

# steps
# file upload -> save
uploaded_file = st.file_uploader("Choose an image")
if uploaded_file is not None:
    if save_uploaded_file(uploaded_file):
        # display the file
        display_image = Image.open(uploaded_file)
        st.image(display_image)
        # feature extract
        features = feature_extraction(os.path.join("uploads",uploaded_file.name),model)
        #st.text(features)
        # recommendention
        indices = recommend(features,feature_list)
        # show
        col1,col2,col3,col4,col5 = st.columns(5)


        with col1:
            st.image(filenames[indices[0][0]])
            save_input_file(filenames[indices[0][0]])
        with col2:
            st.image(filenames[indices[0][1]])
            save_input_file(filenames[indices[0][1]])
        with col3:
            st.image(filenames[indices[0][2]])
            save_input_file(filenames[indices[0][2]])
        with col4:
            st.image(filenames[indices[0][3]])
            save_input_file(filenames[indices[0][3]])
        with col5:
            st.image(filenames[indices[0][4]])
            save_input_file(filenames[indices[0][4]])


    else:
        st.header("Some error occured in file upload")

if st.button('Segment'):
    os.system('python GarmentTransfer.py')


directory="/Users/narendraomprakash/Desktop/Narendra/Semester-VI-WINTER2021/TARP/Datasets/result_images/"
segmented=os.listdir(directory)
try:
    segmented.remove('.DS_Store')
except:
    pass
print(segmented)

if len(segmented)!=0:
    col1,col2,col3,col4,col5 = st.columns(5)

    with col1:
        st.image(directory+segmented[0])
    with col2:
        st.image(directory+segmented[1])
    with col3:
        st.image(directory+segmented[2])
    with col4:
        st.image(directory+segmented[3])
    with col5:
        st.image(directory+segmented[4])

directory="/Users/narendraomprakash/Desktop/Narendra/Semester-VI-WINTER2021/TARP/Datasets/result_images/"
segmented=os.listdir(directory)
try:
    segmented.remove('.DS_Store')
except:
    pass
count=1
print(count)
col1,col2,col3,col4,col5 = st.columns(5)
l=[col1,col2,col3,col4,col5]
for files in segmented:
    with l[count-1]:
        if st.button('Try dress '+str(count)):
            os.system('python /Users/narendraomprakash/Desktop/Narendra/Semester-VI-WINTER2021/TARP/Virtual-Try-On-master/virtualtryon.py '+directory+files)
    count+=1