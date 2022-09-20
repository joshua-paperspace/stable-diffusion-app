# from tkinter import DISABLED
# from turtle import onclick
import streamlit as st
# import pandas as pd
from PIL import Image
from time import sleep
import joblib
import os

import torch
from torch import autocast

os.environ['KMP_DUPLICATE_LIB_OK']='True'

st.set_page_config(
    page_title="Stable Diffusion Image Generator",
    page_icon="🎈",
    layout='wide'
)

def _max_width_():
    max_width_str = f"max-width: 4000px;"
    st.markdown(
        f"""
    <style>
    .reportview-container .main .block-container{{
        {max_width_str}
    }}
    </style>    
    """,
        unsafe_allow_html=True,
    )


_max_width_()

ca, cb, cc = st.columns([1, 7, 1])

# with c30:
    # st.image("logo.png", width=400)
with cb:
    st.title("Stable Diffusion Image Generator")
    st.header("")



    with st.expander("ℹ️ - About this app", expanded=True):

        st.write(
            """     
        The *Stable Diffusion Image Generator* app is an easy-to-use interface built in Streamlit that allows you to enter in a prompt and view AI generated images!
            """
        )

        st.markdown("")

    st.markdown("")
    st.markdown("## Generation Form")

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

    # @st.cache
    pipe = joblib.load('/opt/models/pipeline-gpu.pkl')
    device = "cuda"
    pipe = pipe.to(device)

    # st.title("Stable Diffusion Image Generator")


    def generate_image():
        prompt = st.session_state.prompt
        with autocast("cuda"):
            image = pipe(prompt, guidance_scale=7.5, height=512, width=512,
                           num_inference_steps=50, seed='random', scheduler='LMSDiscreteScheduler')["sample"][0]
        # if prompt=='bird':
        #     image_name = 'bird.jpeg'
        # else: 
        #     image_name = 'cat.jpeg'

        image_name = 'generated_image.jpeg"'
        # image = Image.open(image_name)

        from io import BytesIO
        buf = BytesIO()
        image.save(buf, format="JPEG")
        byte_im = buf.getvalue()


        # st.image(image, caption='Generated image.', use_column_width=True)
        return image, byte_im, image_name

        # btn.data=byte_im
        # btn.file_name=image_name

    # ce, c1, ce, c2, c3 = st.columns([0.07, 1, 0.07, 5, 0.07])
    # cs, c1, c2, c3, cLast = st.columns([2, 1.5, 1.5, 1.5, 2])
c1, c2, c3, c4 = st.columns([1, 3.5, 3.5, 1])
with c2:
    with st.form(key="my_form"):
        
            size = st.radio(
                "Choose image size",
                ["Small", "Medium", "Large"],
            )

            num_inference_steps = st.slider(
                "# of generation steps",
                min_value=10,
                max_value=100,
                value=50,
                help="You can choose the number of inference steps the model will take during image generation. Between 10 and 100, default number is 50.",
            )

            submit_button = st.form_submit_button(label="Generate")
                # generateButton.disabled=True
with c3:
    st.text_area(label='Text prompt', value='Enter in your text prompt', key='prompt', height=260)

if not submit_button:
    st.stop()

c1, c2, c3, c4, c5 = st.columns([1, 5, .3, 1, 1])
image, byte_im, image_name = generate_image()
with c2:
    st.image(image, caption='Generated image.', use_column_width=True)
with c4:
    btn = st.download_button(
        label="Download Image",
        data=byte_im,
        file_name=image_name,
        mime="image/jpeg",
        )

# st.write(size)

            # generateButton.enabled=False

        # if ModelType == "Default (DistilBERT)":
            # kw_model = KeyBERT(model=roberta)

# image = Image.open('bird.jpeg')



# import torch
# from torch import autocast
# from diffusers import StableDiffusionPipeline

# model_id = "CompVis/stable-diffusion-v1-4"
# device = "cuda"
# pipe = StableDiffusionPipeline.from_pretrained(
#     model_id, torch_dtype=torch.float16, revision="fp16", use_auth_token=True)
# pipe = pipe.to(device)

# import joblib
# joblib.dump(pipe, 'pipeline.pkl', compress = 1)

# sample_num = 1
# lst = []
# prompt = 'a corgi astronaut on mars'
# for i in range(sample_num):
#     with autocast("cuda"):
#         a = lst.append(pipe(prompt, guidance_scale=7.5, height=512, width=512,
#                        num_inference_steps=50, seed='random', scheduler='LMSDiscreteScheduler')["sample"][0])
#         lst.append(a)
#         display(a)
#         a.save(f'outputs/gen-image-{i}.png')





# st.text_input("Your name", key="name")

# You can access the value at any point with:
# st.session_state.name



    # sleep(5)


# byte_im =''
# image_name=''


# btn = st.download_button(
#     label="Download Image",
#     data=byte_im,
#     file_name=image_name,
#     mime="image/jpeg",
#     )

# st.button('Generate', on_click=generate_image)
# generateButton = st.button("Generate", DISABLED=True)





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

