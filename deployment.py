import streamlit as st
import torch
import cv2
import numpy as np
import pickle
from torch.autograd import Variable
from PIL import Image
from streamlit_image_select import image_select
import requests

class AlexNet(torch.nn.Module):
    def __init__(self):
        super(AlexNet,self).__init__()
        self.conv = torch.nn.Sequential(
            torch.nn.Conv2d(in_channels=3,
                            out_channels=96,
                            kernel_size=11,
                            stride=4),
            torch.nn.BatchNorm2d(96),
            torch.nn.ReLU(),        
            torch.nn.MaxPool2d(3,2),
            torch.nn.Conv2d(96,256,5,padding=2),
            torch.nn.BatchNorm2d(256),
            torch.nn.ReLU(),
            torch.nn.MaxPool2d(3,2),
            torch.nn.Conv2d(256,384,3,padding=1),
            torch.nn.ReLU(),
            torch.nn.Conv2d(384,384,3,padding=1),
            torch.nn.ReLU(),
            torch.nn.Conv2d(384,256,3,padding=1),
            torch.nn.ReLU(),
            torch.nn.MaxPool2d(3,2),
        )
        self.fc = torch.nn.Sequential(
            torch.nn.Linear(256*5*5,4096),
            torch.nn.ReLU(),     
            torch.nn.Linear(4096, 4096),
            torch.nn.ReLU(),
            torch.nn.Linear(4096, 2),
            #torch.nn.Softmax(dim=1)         
        )
    def forward(self, x):
        x = self.conv(x)
        print(x.size())
        x = x.contiguous().view(-1,256*6*6) 
        x = self.fc(x) 
        return x
    
# Load the trained model
model = torch.load("model.pkl",map_location=torch.device("cuda" if torch.cuda.is_available() else "cpu"))

# Define the labels
labels = ['AI-generated', 'Real']

def eval(model, img):
    model.eval()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")    
    # Preprocess the image
    img = torch.tensor(img)
    img = img.float()
    img = img.unsqueeze(0)  # Add an extra dimension to represent the batch size of 1
    img = np.transpose(img, (0, 3, 1, 2))  # Transpose the dimensions to match the model's input
    
    # Move the input to the device
    img = img.to(device)
    
    # Evaluate the model
    out = model(img)
    # Get the predicted label and the probabilities
    prob = torch.nn.functional.softmax(out, dim=1)
    prob = prob.detach().cpu().numpy().squeeze()  # Convert tensor to numpy array    
    # Get the predicted label
    _, pred = torch.max(out, 1)
    return pred.item(),prob


# Define the Streamlit app
def main():
    st.title("Real vs AI-generated Face Image Classifier")
    st.write("This app classifies face images as Real or AI-generated")
    image_urls = [
        "https://raw.githubusercontent.com/josepharielct/real-fake-face-detection/main/image%20for%20demo/ai%20generate/001DDU0NI4.jpg",
        "https://raw.githubusercontent.com/josepharielct/real-fake-face-detection/main/image%20for%20demo/real/00006.jpg",
        "https://raw.githubusercontent.com/josepharielct/real-fake-face-detection/main/image%20for%20demo/real/licensed-image.jpg",
        "https://raw.githubusercontent.com/josepharielct/real-fake-face-detection/main/image%20for%20demo/ai%20generate/s_6F9FA8C635E5B49A4AA9474CB76D1146BC1F6408CE5EEC320713FFF4758F98B5_1569007594270_003015.jpg",
    ]

    # Download the images and store them in the root folder
    for i, url in enumerate(image_urls):
        response = requests.get(url)
        with open(f"{i+1}.jpg", "wb") as f:
            f.write(response.content)

    # Store the saved image names as a list
    saved_images = [f"{i+1}.jpg" for i in range(len(image_urls))]

    # Allow the user to upload an image
    uploaded_file = st.file_uploader("Choose a face image", type=['jpg', 'jpeg', 'png'])
    selected_file = image_select(
    label="FOR DEMO: Select a face",
    images=[
        saved_images[0],
        saved_images[1],
        saved_images[2],
        saved_images[3]
    ]
)
    # If an image is uploaded, show it and classify it
    if uploaded_file is not None and selected_file is not None:
        image = Image.open(uploaded_file)
        image.save("img.jpg")
        image = cv2.imread('img.jpg')
        image_up = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
        image = cv2.resize(image, (227, 227), interpolation=cv2.INTER_AREA)
        image_up = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
    elif uploaded_file is not None:
        image = Image.open(uploaded_file)
        image.save("img.jpg")
        image = cv2.imread('img.jpg')
        image_up = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
        image = cv2.resize(image, (227, 227), interpolation=cv2.INTER_AREA)
        image_up = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
    elif selected_file is not None:
        image = Image.open(selected_file)
        image.save("img.jpg")
        image = cv2.imread('img.jpg')
        image_up = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
        image = cv2.resize(image, (227, 227), interpolation=cv2.INTER_AREA)
        image_up = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
    # Predict
    pred_label, prob = eval(model,image)
    pred_prob = prob[pred_label] * 100
    # Show the uploaded image and the predicted label and probability
    st.write(f"The image is {labels[pred_label]} with {pred_prob:.2f}% probability")
    st.image(image_up, caption='Image', use_column_width=True)
        

# Run the Streamlit app
if __name__ == '__main__':
    main()
