# Create streamlit app
import streamlit as st
import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import transforms
from PIL import Image

st.title('Alzheimer\'s Disease Detection')

st.write('This is a simple web app to predict Alzheimer\'s Disease using MRI images')

# Use .pth model saved from training
model = torch.load('/.streamlit/alzheimer_efficientnet_model.pth')
model.eval()

# Preprocess image
def preprocess(image):
    transform = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])
    image = transform(image)
    image = image.unsqueeze(0)
    return image

# Predict
def predict(image):
    image = preprocess(image)
    output = model(image)
    _, predicted = torch.max(output, 1)
    return predicted

st.write('Please upload an MRI image')

# Load image
uploaded_file = st.file_uploader("Choose an MRI image...", type="jpg")
if uploaded_file is not None:
    image = Image.open(uploaded_file)
    st.image(image, caption='Uploaded MRI.', use_column_width=True)
    st.write("")
    st.write("Classifying...")
    label = predict(preprocess(image))
    if label == 0:
        st.write('The model predicts that this MRI image is from a mild Alzheimer\'s Disease patient')
    elif label == 1:
        st.write('The model predicts that this MRI image is from a moderate Alzheimer\'s Disease patient')
    elif label == 2:
        st.write('The model predicts that this MRI image is from a non-demented patient')
    elif label == 3:
        st.write('The model predicts that this MRI image is from a very mild Alzheimer\'s Disease patient')