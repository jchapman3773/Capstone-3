import os
from uuid import uuid4
from flask import Flask, request, redirect, render_template
from werkzeug.utils import secure_filename
import boto3

# UPLOAD_FOLDER = 'uploads/'
s3 = boto3.resource('s3')
bucket = s3.Bucket('bananaforscale')

ALLOWED_EXTENSIONS = set(['txt', 'pdf', 'png', 'jpg', 'jpeg', 'gif'])

application = Flask(__name__)

def allowed_file(filename):
    return '.' in filename and \
           filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

@application.route('/', methods=['GET','POST'])
def hello():
    return render_template('home.html')

@application.route('/upload', methods=['GET', 'POST'])
def upload_file():
    if request.method == 'POST':
        # check if the post request has the file part
        if 'file' not in request.files:
            return 'Error: No file part, please try again'
        file = request.files['file']
        if file.filename == '':
            return 'Error: No file name, please try again'
        if file and allowed_file(file.filename):
            feet = request.form['Feet']
            inches = request.form['Inches']
            if feet == '' or inches == '':
                return 'Error: Blank entry in form. Please fill out entire form.'
            if feet.isprintable() and inches.isprintable():
                height = (float(feet)*12)+float(inches)
                filename = secure_filename(file.filename)
                file_ext = '.'+ filename.rsplit('.', 1)[1].lower()
                new_filename = uuid4().hex + "_" + str(height) + file_ext
                obj = bucket.Object('heights.csv')
                prev = obj.get()['Body'].read().decode('utf-8')
                new = prev + '\n' + new_filename + ',' + str(height)
                obj.put(Body=new,Key='heights.csv')
                bucket.upload_fileobj(file,f'uploads/{new_filename}')
            else:
                return 'Error: Non-Printable characters. Please try again.'
            return redirect('/success')
        else:
            return redirect(request.url)
    return render_template('upload.html')

@application.route('/success', methods=['GET'])
def success():
    return '''
    <!DOCTYPE html>
        <body>
            <h1 align="center">File Upload Success!</h1>
        </body>
    '''

if __name__ == '__main__':
    application.run(debug=True)
