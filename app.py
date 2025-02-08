from flask import Flask, render_template, request, send_from_directory, url_for, redirect

from flask_uploads import UploadSet, IMAGES, configure_uploads
from flask_wtf import FlaskForm
from flask_wtf.file import FileField, FileRequired, FileAllowed
from wtforms import SubmitField
import os

import torch
import torch.nn as nn
import torchvision.transforms as transforms
from PIL import Image
from architecture import CNN_NeuralNet

#credit to Red Eyed Coder Club on Youtube for code
#NOTES: add session cookies to prevent reuploading of same image over and over again
#add a login to keep track of images and plants
#add camera input to take photo

app = Flask(__name__)
app.config['SECRET_KEY'] = "hello"
app.config['UPLOADED_PHOTOS_DEST'] = "uploads"

photos = UploadSet("photos", IMAGES)
configure_uploads(app, photos)

class UploadForm(FlaskForm):
    photo = FileField(
        validators=[
            FileAllowed(photos, "Only images are allowed"),
            FileRequired("File field should not be empty")
        ]
    )
    submit = SubmitField("Upload")

num_classes = 38
num_channels = 3
model = CNN_NeuralNet(num_channels, num_classes)  # Initialize and load model
model.load_state_dict(torch.load("plant_disease_model.pth", map_location=torch.device('cpu')))
model.eval()

transform = transforms.Compose([
    transforms.Resize((256, 256)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])

@app.route('/uploads<filename>')
def get_file(filename):
    return send_from_directory(app.config['UPLOADED_PHOTOS_DEST'], filename)

@app.route('/camera')
def camera():
    return render_template("camera.html")

@app.route('/home')
def home():
    return redirect(url_for("index"))

@app.route('/', methods=["GET", "POST"])
def index():
    form = UploadForm()
    file_url = None
    prediction = None
    prediction_name = None
    prediction_status = None

    if form.validate_on_submit():
        filename = photos.save(form.photo.data)
        file_url = url_for("get_file", filename=filename) #bigggggGGGGG

        image_path = os.path.join(app.config['UPLOADED_PHOTOS_DEST'], filename)
        image = Image.open(image_path).convert('RGB')
        image = transform(image).unsqueeze(0) 

        with torch.no_grad():
            output = model(image)
            _, predicted = torch.max(output, 1)
            prediction = predicted.item() 
            class_names = ['Apple___Apple_scab', 'Apple___Black_rot', 'Apple___Cedar_apple_rust', 'Apple___healthy', 
                'Blueberry___healthy', 'Cherry_(including_sour)___healthy', 'Cherry_(including_sour)___Powdery_mildew', 
                'Corn_(maize)___Cercospora_leaf_spot Gray_leaf_spot', 'Corn_(maize)___Common_rust_', 'Corn_(maize)___healthy', 
                'Corn_(maize)___Northern_Leaf_Blight', 'Grape___Black_rot', 'Grape___Esca_(Black_Measles)', 'Grape___healthy', 
                'Grape___Leaf_blight_(Isariopsis_Leaf_Spot)', 'Orange___Haunglongbing_(Citrus_greening)', 'Peach___Bacterial_spot', 
                'Peach___healthy', 'Pepper,_bell___Bacterial_spot', 'Pepper,_bell___healthy', 'Potato___Early_blight', 'Potato___healthy', 
                'Potato___Late_blight', 'Raspberry___healthy', 'Soybean___healthy', 'Squash___Powdery_mildew', 'Strawberry___healthy', 
                'Strawberry___Leaf_scorch', 'Tomato___Bacterial_spot', 'Tomato___Early_blight', 'Tomato___healthy', 'Tomato___Late_blight', 
                'Tomato___Leaf_Mold', 'Tomato___Septoria_leaf_spot', 'Tomato___Spider_mites Two-spotted_spider_mite', 'Tomato___Target_Spot', 
                'Tomato___Tomato_mosaic_virus', 'Tomato___Tomato_Yellow_Leaf_Curl_Virus']
            prediction_name = class_names[prediction]

            tokens = prediction_name.split("_")
            tokens = list(filter(None, tokens))
            prediction_name = " ".join(tokens[:-1])
            prediction_status = tokens[-1:][0]
            

    return render_template("index.html", form=form, file_url=file_url, prediction={"name":prediction_name, "status":prediction_status})

