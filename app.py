import tensorflow as tf
import streamlit as st
import numpy as np
import time
import tempfile
from streamlit_drawable_canvas import st_canvas

import pandas as pd
from PIL import Image
import streamlit as st
from streamlit_drawable_canvas import st_canvas

class_names = ['banana', 'baseball', 'bat', 'bee', 'bowtie', 'broccoli', 'cactus', 'candle', 'carrot', 'circle', 'cloud', 'computer', 'crown', 'diamond', 'duck', 'ear', 'eye', 'guitar', 'house', 'key', 'keyboard', 'leaf', 'light bulb', 'line', 'pants', 'piano', 'shorts', 'smiley face', 'square', 'star', 'stop sign', 'sun', 't-shirt', 'tent', 'tooth', 'triangle']

v4 = tf.keras.models.load_model("model-v4.h5")

st.title("Doodle Classifier")

def complete_pred(img_path,model):
    img = tf.keras.preprocessing.image.load_img(img_path, target_size=(256, 256), color_mode="grayscale")
    img_array = tf.keras.preprocessing.image.img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0)
    predictions = model.predict(img_array)
    ans = class_names[np.argmax(predictions)]
    return ans

    


sidebar = st.sidebar
sidebar.write("Navigator")

options = sidebar.selectbox(
    "What are your favorite colors",
    ["〽️ | Sketch your image", "〰️ | Capture your image", "♻️ | Upload your image"]
)
   
if options == "♻️ | Upload your image":
   st.header("Image Uploader:")
   uploader = st.file_uploader(label="Upload the image here",type=["jpg","png"])       

   if uploader is not None:           
       time.sleep(3)
       st.toast("Image fetched!",icon="ℹ️")
       col1,col2 = st.columns(2)
    
        
       print("log: File uploaded!")
       with tempfile.NamedTemporaryFile(delete=False) as temp_file:
            temp_file.write(uploader.read())
            image_filename = temp_file.name
       ans = complete_pred(img_path=image_filename,model=v4)
       progress_text = "Processing....."
       my_bar = st.progress(0, text=progress_text)

    
       
    #    st.header(ans)
       st.image(caption=ans,image=uploader)
       st.toast(f'Your doodle is predicted to be: {ans}', icon="ℹ️")


elif options == "〰️ | Capture your image":
    pic = st.camera_input("Show the doodle!")
    if pic:
        #  img = Image.fromarray(np.uint8(pic))
        #  with tempfile.NamedTemporaryFile(delete=False, suffix=".png") as temp_file:
        #     img.save(temp_file, format="PNG")
        #     image_filename = temp_file.name
         
         ans = complete_pred(img_path=pic,model=v4)
         progress_text = "Processing...."
         my_bar = st.progress(0, text=progress_text)
         
         for percent_complete in range(100):
                time.sleep(0.05)
                my_bar.progress(percent_complete + 1, text=progress_text)
         st.toast(f'Your doodle is predicted to be: {ans}', icon="ℹ️")



elif options == "〽️ | Sketch your image":
     # Specify canvas parameters in application
    drawing_mode = st.sidebar.selectbox(
        "Drawing tool:", ("freedraw", "line", "rect", "circle", "transform")
    )

    stroke_width = st.sidebar.slider("Stroke width: ", 1, 25, 3)
    realtime_update = st.sidebar.checkbox("Update in realtime", True)

    

# Create a canvas component
    canvas_result = st_canvas(
        fill_color="rgba(255, 165, 0, 0.3)",  # Fixed fill color with some opacity
        stroke_width=stroke_width,
        stroke_color="#000000",
        background_color="#EEEEEE",
        update_streamlit=realtime_update,
        height=500,
        drawing_mode=drawing_mode,
        key="canvas",
    )
    btn4 = st.button("Classify")
    if btn4:
        img = Image.fromarray(np.uint8(canvas_result.image_data))
        with tempfile.NamedTemporaryFile(delete=False, suffix=".png") as temp_file:
            img.save(temp_file, format="PNG")
            image_filename = temp_file.name
        ans = complete_pred(img_path=image_filename,model=v4)
        time.sleep(2)
        st.toast(f'Your doodle is predicted to be: {ans}', icon="ℹ️")



st.divider()
# adi = st.link("©️Vishal Adithya.A","https://github.com/Bissaru")
# ab = st.link_button("©️Akshay Balaji","https://github.com/Shay2030")
st.write("""This app's framework and frontend is built by Vishal Adithya.A & Akshay Balaji while the backend,ML and deep learning stuff is made by Vishal Adithya.A.""")
st.write("All contents,models,frameworks,images,etc present here are ©️")
st.write("For more info contact : https://github.com/Bissaru or https://github.com/Shay2030")

