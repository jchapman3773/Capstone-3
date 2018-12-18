import pandas as pd
from io import BytesIO
from uuid import uuid4
from flask import Flask, request, redirect, render_template
from werkzeug.utils import secure_filename
from boto3 import resource, client
from keras.models import load_model
from keras.applications.xception import preprocess_input
from keras import backend as K
from keras.preprocessing import image
from numpy import expand_dims
from sqlalchemy import create_engine
from fix_image_orientation import fix_orientation

s3 = resource('s3')
bucket = s3.Bucket('bananaforscale')

engine = create_engine('postgresql://banana:forscale@bananaforscale.ckaldwfguyw5.us-east-2.rds.amazonaws.com:5432/bananaforscale')

ALLOWED_EXTENSIONS = set(['txt', 'pdf', 'png', 'jpg', 'jpeg', 'gif'])

application = Flask(__name__)

@application.route('/testdb',methods=['GET','POST'])
def testdb():
    df_new = pd.read_sql_table('heights',con=engine)
    return str(df_new.iloc[:,:])

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
                # upload to database
                df = pd.DataFrame.from_dict({'image':[new_filename],
                                            'height_inch':[height]})
                df.to_sql('heights',con=engine,if_exists='append')
                # upload height to bucket csv
                obj = bucket.Object('heights.csv')
                prev = obj.get()['Body'].read().decode('utf-8')
                new = prev + '\n' + new_filename + ',' + str(height)
                obj.put(Body=new,Key='heights.csv')
                # upload image to bucket
                file = fix_orientation(file)
                try:
                    if file.format.lower() not in ALLOWED_EXTENSIONS:
                        format = 'JPEG'
                    else:
                        format = file.format
                except:
                    format = 'JPEG'
                buffer = BytesIO()
                file.save(buffer,format)
                buffer.seek(0) # rewind pointer
                bucket.put_object(Key=f'uploads/{new_filename}',Body=buffer,
                                    ContentType=f'image/{format}')
            else:
                return 'Error: Non-Printable characters. Please try again.'
            return redirect('/success')
        else:
            return redirect(request.url)
    return render_template('upload.html')

@application.route('/class_prediction', methods=['GET', 'POST'])
def class_prediction():
    if request.method == 'POST':
        if 'file' not in request.files:
            return 'Error: No file part, please try again'
        file = request.files['file']
        if file.filename == '':
            return 'Error: No file name, please try again'
        if file and allowed_file(file.filename):
            model = load_model('models/transfer_CNN.h5')
            img = image.load_img(file, target_size=(800, 800))
            x = image.img_to_array(img)
            x = preprocess_input(x)
            x = expand_dims(x, axis=0)
            pred = model.predict(x)
            key = {0:'Banana', 1:'Both', 2:'Neither', 3:'Person'}
            K.clear_session()
            return render_template('prediction_answer.html',variable=[pred,key])
        else:
            return redirect(request.url)
    return render_template('prediction.html')

@application.route('/success', methods=['GET'])
def success():
    return '''
    <!DOCTYPE html>
        <body>
            <h1 align="center">File Upload Success!</h1>
        </body>
    '''

@application.route('/form', methods=['GET','POST'])
def form():
    if request.method == 'POST':
        comments = request.form['comments']
        client = client('s3')
        client.put_object(Body=comments,Bucket='bananaforscale',Key=f'comments/{uuid4().hex}.txt')
        return redirect('/success')
    return '''
    <!DOCTYPE html>
        <body>
            <form method=POST enctype=multipart/form-data>
                <div class="form-group">
                    <label for="comments">Questions, Comments, Errors, etc:</label>
                    <br>
                    <input type="comments" class="form-control" placeholder="Type Here" name="comments">
                </div>
                <button type="submit" class="btn btn-primary" value=Upload>Submit</button>
            </form>
        </body>
    '''

if __name__ == '__main__':
    application.run(debug=True)
