from tkinter import DISABLED
from turtle import onclick
import streamlit as st
# import pandas as pd
from PIL import Image
from time import sleep
import joblib
import os

os.environ['KMP_DUPLICATE_LIB_OK']='True'

# !pip install --upgrade diffusers transformers scipy ftfy
# !pip install flax==0.5.0 --no-deps
# !pip install ipywidgets msgpack rich 
# import torch
# import os


# from preprocess import imgToTensor
# from resnet import resnet18
# import os

# classes = ('plane', 'car', 'bird', 'cat',
#         'deer', 'dog', 'frog', 'horse', 'ship', 'truck')

# MODEL_DIR = '/opt/models/'
# for filename in os.listdir(MODEL_DIR):
#     if filename[-4:] == '.pth':
#         filepath = os.path.join(MODEL_DIR,filename)
# MODEL_PATH = filepath

pipe = joblib.load('pipeline-gpu.pkl')

st.title("Stable Diffusion Image Generator")

st.text_input('Text prompt', key='prompt')
# image = Image.open('bird.jpeg')




# st.text_input("Your name", key="name")

# You can access the value at any point with:
# st.session_state.name

def generate_image():
    prompt = st.session_state.prompt
    image = pipe(prompt, guidance_scale=9.9, height=512, width=512,
                       num_inference_steps=2, seed='random', scheduler='LMSDiscreteScheduler')["sample"][0]
    # if prompt=='bird':
    #     image = Image.open('bird.jpeg')
    # else: 
    #     image = Image.open('cat.jpeg')
        
    st.image(image, caption='Generated image.', use_column_width=True)
    # sleep(5)

# st.button('Generate', on_click=generate_image)
# generateButton = st.button("Generate", DISABLED=True)
if st.button("Generate"):
    # generateButton.disabled=True
    generate_image()
    # generateButton.enabled=False



# uploaded_file = st.file_uploader("Choose an image...", type=['png','jpeg'])
# if uploaded_file is not None:
#     image = Image.open(uploaded_file)
#     st.image(image, caption='Uploaded Image.', use_column_width=True)
#     st.write("")
#     st.write("Classifying...")

#     device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
#     model = resnet18(3, 10)
#     model.load_state_dict(torch.load(MODEL_PATH, map_location=device))

#     tensor = imgToTensor(image)
    
#     output = model(tensor)
#     _, predicted = torch.max(output.data, 1)
#     prediction = classes[predicted]

#     st.write(prediction)