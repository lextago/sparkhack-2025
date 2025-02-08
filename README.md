## Prerequisites

Use the package manager [pip](https://pip.pypa.io/en/stable/) to install the prereqs

```bash
python -m pip install -r requirements.txt
```

Create the virtual environment

```bash
python -m venv venv
```

Navigate to the repository folder

Set Flask Environment Variables:

Windows (CMD):
```bash
set FLASK_APP=app.py
set FLASK_ENV=development
```

macOS/Linux:
```
export FLASK_APP=app.py
export FLASK_ENV=development
```

Then activate it:

```bash
venv\Scripts\Activate
```

And finally run 
```bash
flask run --port=8000
```

## Usage

Take any of the photos in the /test folder, upload, and submit once 
you're on the landing page of the site.

## Warnings

You may encounter an error as such:

```
ImportError: cannot import name 'secure_filename' from 'werkzeug'
```

Please follow the instructions here to resolve that as it is related
to Werkzeug itself [here]. (https://stackoverflow.com/questions/61628503/flask-uploads-importerror-cannot-import-name-secure-filename)

## Credits

Model architecture and notebook code by [vipoooool] (https://www.kaggle.com/datasets/vipoooool/new-plant-diseases-dataset/data)