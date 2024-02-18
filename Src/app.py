import streamlit as st
from PIL import Image
from torchvision import transforms
import os
import torch
from torchvision import datasets, transforms, models
from torch.utils.data import DataLoader, random_split
from torch import nn, optim

st.title('Alzheimer\'s Disease Detection')
st.write('This is a simple web app to predict Alzheimer\'s Disease using MRI images.')

# Load the model
MODEL_PATH = 'alzheimer_efficientnet_model.pth'

try:
    model = torch.load(MODEL_PATH, map_location=torch.device('cpu'))
    model.eval()
    model_loaded = True
except FileNotFoundError:
    st.error(f"Model file not found at {MODEL_PATH}. Please check the path.")
    model_loaded = False

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
    with torch.no_grad():
        output = model(image)
        _, predicted = torch.max(output, 1)
    return predicted.item()

st.write('Please upload an MRI image.')

# Load image
uploaded_file = st.file_uploader("Choose an MRI image...", type=["jpg", "jpeg"])
if uploaded_file is not None and model_loaded:
    image = Image.open(uploaded_file).convert('RGB')  # Ensure image is RGB
    st.image(image, caption='Uploaded MRI.', use_column_width=True)
    st.write("Classifying...")
    preprocessed_image = preprocess(image)
    label = predict(preprocessed_image)
    labels = ['mild Alzheimer\'s Disease', 'moderate Alzheimer\'s Disease', 'non-demented', 'very mild Alzheimer\'s Disease']
    if label in range(len(labels)):
        st.write(f'The model predicts that this MRI image is from a {labels[label]} patient.')
    else:
        st.error('An error occurred during classification.')
