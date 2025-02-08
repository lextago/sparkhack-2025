from flask import Flask, render_template, request, send_from_directory, url_for, redirect

from flask_uploads import UploadSet, IMAGES, configure_uploads
from flask_wtf import FlaskForm
from flask_wtf.file import FileField, FileRequired, FileAllowed
from wtforms import SubmitField
import os
import base64
from io import BytesIO

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

app.config['UPLOAD_FOLDER'] = 'static/uploads/'

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
    transforms.ToTensor()
])

@app.route('/uploads<filename>')
def get_file(filename):
    return send_from_directory(app.config['UPLOADED_PHOTOS_DEST'], filename)

@app.route('/camera', methods=["GET", "POST"])
def camera():
    # Get the JSON data from the request
    data = request.get_json()

    # Get the base64 image string from the request
    image_data = data.get('image')

    if image_data:
        # Remove the prefix "data:image/png;base64," if it exists
        image_data = image_data.split(',')[1]  # Strip the prefix

        # Decode the base64 string into bytes
        image_bytes = base64.b64decode(image_data)

        # Convert the bytes to an image using PIL
        image = Image.open(BytesIO(image_bytes))

        # Save the image to the uploads folder
        image_path = os.path.join(app.config['UPLOAD_FOLDER'], 'captured_image.png')
        image.save(image_path)

        return render_template('camera.html', message="Image uploaded successfully!")
    else:
        # Return a template with an error message
        return render_template('camera.html', message="Error: No image provided")

@app.route('/home')
def home():
    return redirect(url_for("index"))

@app.route('/about')
def about():
    return render_template("about.html")

@app.route('/', methods=["GET", "POST"])
def index():
    form = UploadForm()
    file_url = None
    prediction_name = None
    prediction_status = None

    class_names = [ #BIGGGGGGGGG
    'Apple___Apple_scab', 'Apple___Black_rot', 'Apple___Cedar_apple_rust', 'Apple___healthy', 
    'Blueberry___healthy', 'Cherry_(including_sour)___healthy', 'Cherry_(including_sour)___Powdery_mildew', 
    'Corn_(maize)___Cercospora_leaf_spot Gray_leaf_spot', 'Corn_(maize)___Common_rust_', 'Corn_(maize)___healthy', 
    'Corn_(maize)___Northern_Leaf_Blight', 'Grape___Black_rot', 'Grape___Esca_(Black_Measles)', 'Grape___healthy', 
    'Grape___Leaf_blight_(Isariopsis_Leaf_Spot)', 'Orange___Haunglongbing_(Citrus_greening)', 'Peach___Bacterial_spot', 
    'Peach___healthy', 'Pepper,_bell___Bacterial_spot', 'Pepper,_bell___healthy', 'Potato___Early_blight', 'Potato___healthy', 
    'Potato___Late_blight', 'Raspberry___healthy', 'Soybean___healthy', 'Squash___Powdery_mildew', 'Strawberry___healthy', 
    'Strawberry___Leaf_scorch', 'Tomato___Bacterial_spot', 'Tomato___Early_blight', 'Tomato___healthy', 'Tomato___Late_blight', 
    'Tomato___Leaf_Mold', 'Tomato___Septoria_leaf_spot', 'Tomato___Spider_mites Two-spotted_spider_mite', 'Tomato___Target_Spot', 
    'Tomato___Tomato_mosaic_virus', 'Tomato___Tomato_Yellow_Leaf_Curl_Virus'
    ]


    if form.validate_on_submit():
        filename = photos.save(form.photo.data)
        file_url = url_for("get_file", filename=filename)

        # Preprocess the image
        image_path = os.path.join(app.config['UPLOADED_PHOTOS_DEST'], filename)
        image = Image.open(image_path).convert('RGB')
        image = transform(image)  # Apply preprocessing transforms

        # Get prediction using the new function
        prediction_name = predict_image(image, model, class_names)

        # Process class name
        tokens = prediction_name.split("_")
        tokens = list(filter(None, tokens))
        prediction_name = " ".join(tokens[:-1])
        prediction_status = tokens[-1:][0]

    return render_template("index.html", form=form, file_url=file_url, prediction={"name": prediction_name, "status": prediction_status})

def get_default_device():
    """Pick GPU if available, else CPU"""
    if torch.cuda.is_available:
        return torch.device("cuda")
    else:
        return torch.device("cpu")
    
device = torch.device("cpu")

# Move tensor(s) to the selected device (CPU/GPU)
def to_device(data, device):
    """Move tensor(s) to chosen device"""
    if isinstance(data, (list, tuple)):
        return [to_device(x, device) for x in data]
    return data.to(device, non_blocking=True)

def predict_image(img, model, class_names):
    """Predicts the class of a given image using the trained model."""
    model.eval()  # Ensure model is in evaluation mode
    xb = img.unsqueeze(0)  

    with torch.no_grad():  
        output = model(xb)
        probabilities = torch.nn.functional.softmax(output, dim=1) 
        confidence, predicted = torch.max(probabilities, 1)  # Get highest probability class (plant, disease type)
        prediction_index = predicted.item()

    # Debug prints
    print(f"Predicted index: {prediction_index}")
    print(f"Number of classes: {len(class_names)}")
    print(f"Class names: {class_names}")

    if prediction_index >= len(class_names):
        raise ValueError(f"Predicted index {prediction_index} is out of range for class_names with length {len(class_names)}")

    print(f"Predicted index: {prediction_index}, Class: {class_names[prediction_index]}, Confidence: {confidence.item():.4f}")

    return class_names[prediction_index]



