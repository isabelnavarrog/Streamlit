import streamlit as st
import torch
import torchvision
import torchvision.transforms as transforms
import matplotlib.pyplot as plt
import numpy as np
import skimage
from PIL import Image
from cnn import CNN
from cnn import load_data
from cnn import load_model_weights

# ========================================================================
# Functions
# ========================================================================
train_dir = './dataset/training'
valid_dir = './dataset/validation'
train_loader, valid_loader, num_classes = load_data(train_dir, 
                                                    valid_dir, 
                                                    batch_size=32, 
                                                    img_size=224) # ResNet50 requires 224x224 images


model_weights = load_model_weights('resnet152-29epoch-10unfreeze')
my_trained_model = CNN(torchvision.models.resnet152(weights='DEFAULT'), num_classes)
my_trained_model.load_state_dict(model_weights)

transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Lambda(lambda x: x.repeat(3, 1, 1) if x.shape[0] == 1 else x)
])

def predict_image(img):
    tensor = transform(img)
    my_trained_model.eval()
    with torch.no_grad():
        output = my_trained_model(tensor.unsqueeze(0))
        _, predicted = torch.max(output, 1)
        classnames = train_loader.dataset.classes
    return predicted, classnames


# ========================================================================
# Application Config
# ========================================================================
# Centra el contenido
st.title("Laboratorio de Deep Learning")

st.markdown(
    """
    <style>
    .center {
        display: flex;
        justify-content: center;
    }
    </style>
    """,
    unsafe_allow_html=True,
)

# Crea una fila con tres columnas
space, col1, space2 = st.columns([1,1,1])

# En la segunda columna, muestra el logo
with col1:
    st.image("img/icai.png", width=100)

# Muestra un encabezado centrado
st.markdown("<h1 style='text-align: center', style= 'color:green;'>Redes Neuronales Convolucionales para Clasificación de Imágenes</h1>", unsafe_allow_html=True)
# Sube una imagen
uploaded_image = st.file_uploader("Elige una imagen", type=['png', 'jpg', 'jpeg'])
if uploaded_image is not None:
    imagen = Image.open(uploaded_image).convert('RGB')
    clase_predicha, classnames = predict_image(imagen)
    clase = classnames[clase_predicha]
    # Muestra la imagen
    st.image(imagen, use_column_width=True)
    # Muestra el resultado centrado
    st.markdown(f"<h3 class='center' style='color: black;'> La imagen es {clase}</h3>", unsafe_allow_html=True)
    st.balloons()