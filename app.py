# from tkinter import DISABLED
# from turtle import onclick
import streamlit as st
from PIL import Image
from time import sleep
import joblib
import os
from io import BytesIO
import torch
from torch import autocast

os.environ['KMP_DUPLICATE_LIB_OK']='True'

st.set_page_config(
    page_title="Stable Diffusion Image Generator",
    page_icon="🎈",
    layout='wide'
)

# @st.cache(hash_funcs={torch.nn.parameter.Parameter: lambda _: None})
def get_pipeline():
    pipe = joblib.load('/opt/models/pipeline-gpu.pkl')
    device = "cuda"
    pipe = pipe.to(device)
    return pipe


def generate_image():
    prompt = st.session_state.prompt

    size_mapping = {'Small': 256, 'Medium': 512, 'Large': 1024}
    size = st.session_state.size
    hw = size_mapping[size]

    num_inference_steps = st.session_state.num_inference_steps

    with autocast("cuda"):
        image = pipe(prompt, guidance_scale=7.5, height=hw, width=hw,
                        num_inference_steps=num_inference_steps, seed='random', scheduler='LMSDiscreteScheduler')["sample"][0]
    # if prompt=='bird':
    #     image_name = 'bird.jpeg'
    # else: 
    #     image_name = 'cat.jpeg'

    image_name = 'generated_image.jpeg'
    # image = Image.open(image_name)

    buf = BytesIO()
    image.save(buf, format="JPEG")
    byte_im = buf.getvalue()
    return image, byte_im, image_name, prompt

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
pipe = get_pipeline()

ca, cb, cc = st.columns([1, 7, 1])
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


c1, c2, c3, c4 = st.columns([1, 3.5, 3.5, 1])
with c2:
    with st.form(key="my_form"):
        
            size = st.radio(
                "Choose image size",
                ["Small", "Medium", "Large"],
                key='size')

            num_inference_steps = st.slider(
                "# of generation steps",
                min_value=10,
                max_value=100,
                value=50,
                help="You can choose the number of inference steps the model will take during image generation. Between 10 and 100, default number is 50.",
                key='num_inference_steps')

            submit_button = st.form_submit_button(label="Generate")

with c3:
    st.text_area(label='Text prompt', value='Enter in your text prompt', key='prompt', height=260)

if not submit_button:
    st.stop()

c1, c2, c3, c4, c5 = st.columns([1, 5, .4, 1.6, 1])
image, byte_im, image_name, prompt = generate_image()

with c2:
    st.image(image, caption=prompt, use_column_width=True)
with c4:
    btn = st.download_button(
        label="📥 Download Image",
        data=byte_im,
        file_name=image_name,
        mime="image/jpeg",
        )

