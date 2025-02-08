from flask import Flask, render_template, request, send_from_directory, url_for, redirect

from flask_uploads import UploadSet, IMAGES, configure_uploads
from flask_wtf import FlaskForm
from flask_wtf.file import FileField, FileRequired, FileAllowed
from wtforms import SubmitField

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

@app.route('/uploads<filename>')
def get_file(filename):
    return send_from_directory(app.config['UPLOADED_PHOTOS_DEST'], filename)

@app.route('/home')
def home():
    return redirect(url_for("index"))

@app.route('/', methods=["GET", "POST"])
def index():
    form = UploadForm()
    if form.validate_on_submit():
        filename = photos.save(form.photo.data)
        file_url = url_for("get_file", filename=filename) #biggggg
    else:
        file_url = None
    return render_template("index.html", form=form, file_url=file_url)