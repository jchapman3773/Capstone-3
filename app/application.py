import pandas as pd
from io import BytesIO
from base64 import b64encode
from uuid import uuid4
from flask import Flask, request, redirect, render_template
from werkzeug.utils import secure_filename
from boto3 import resource, client
from keras.models import load_model
from keras.applications.xception import preprocess_input
from keras import backend as K
from keras.preprocessing import image
from numpy import expand_dims, argmax
from sqlalchemy import create_engine
from fix_image_orientation import fix_orientation
from root_mean_squared_error import root_mean_squared_error
from object_detection import object_detector
from joblib import load
from PIL import Image

s3 = resource('s3')
bucket = s3.Bucket('bananaforscale')

engine = create_engine('postgresql://banana:forscale@bananaforscale.ckaldwfguyw5.us-east-2.rds.amazonaws.com:5432/bananaforscale')

ALLOWED_EXTENSIONS = set(['txt', 'pdf', 'png', 'jpg', 'jpeg', 'gif'])

def create_tag(img):
    # format image for return in html
    output = BytesIO()
    try:
        if img.format.lower() not in ALLOWED_EXTENSIONS:
            format = 'JPEG'
        else:
            format = img.format
    except:
        format = 'JPEG'
    img.save(output, format)
    contents = b64encode(output.getvalue()).decode()
    output.close()
    tag = f'data:image/{format};base64,{contents}'

    return tag

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

@application.route('/success', methods=['GET','POST'])
def success():
    return render_template('success.html')

@application.route('/xception_class', methods=['GET', 'POST'])
def xception_class():
    if request.method == 'POST':
        if 'file' not in request.files:
            return 'Error: No file part, please try again'
        file = request.files['file']
        if file.filename == '':
            return 'Error: No file name, please try again'
        if file and allowed_file(file.filename):
            K.clear_session()
            model = load_model('models/transfer_CNN.h5')
            img = image.load_img(file, target_size=(800, 800))
            x = image.img_to_array(img)
            x = preprocess_input(x)
            x = expand_dims(x, axis=0)
            pred = model.predict(x)
            idx = argmax(pred)
            key = {0:'a Banana', 1:'both a Banana and a Person', 2:'neither a Banana nor a Person', 3:'a Person'}
            pred = round(pred[0,idx]*100,2)
            key[idx]

            # process image for html
            img = Image.open(file)
            tag = create_tag(img)

            return render_template('prediction_answer.html',variable=[f'Your image has {key[idx]}',f'with {pred}% confidence',tag])
        else:
            return redirect(request.url)
    return render_template('prediction.html',Title='Classification Prediction with Xception')

@application.route('/xception_reg', methods=['GET', 'POST'])
def xception_reg():
    if request.method == 'POST':
        if 'file' not in request.files:
            return 'Error: No file part, please try again'
        file = request.files['file']
        if file.filename == '':
            return 'Error: No file name, please try again'
        if file and allowed_file(file.filename):
            K.clear_session()
            model = load_model('models/transfer_CNN_reg.h5',custom_objects={'root_mean_squared_error': root_mean_squared_error})
            img = image.load_img(file, target_size=(800, 800))
            x = image.img_to_array(img)
            x = preprocess_input(x)
            x = expand_dims(x, axis=0)
            pred = model.predict(x)[0][0]
            ft = int(pred // 12)
            inch = round(pred % 12,1)

            # process image for html
            img = Image.open(file)
            tag = create_tag(img)

            return render_template('prediction_answer.html',variable=[f'{ft} ft {inch} in','',tag])
        else:
            return redirect(request.url)
    return render_template('prediction.html',Title='Regression Prediction with Xception')

@application.route('/predict_height', methods=['GET', 'POST'])
def predict_height():
    if request.method == 'POST':
        if 'file' not in request.files:
            return 'Error: No file part, please try again'
        file = request.files['file']
        if file.filename == '':
            return 'Error: No file name, please try again'
        if file and allowed_file(file.filename):
            K.clear_session()
            img, result_df = object_detector([file.filename],[file])
            if len(img) == 0:
                return 'Error: Image could not be used.'

            # format image for return in html
            img = img[file.filename]
            tag = create_tag(img)

            # scale and clean data
            scaler = load('models/scaler.joblib')
            try:
                result_df['banana_box'] = result_df[['banana_box_point1','banana_box_point2','banana_box_point3','banana_box_point4']].mean(axis=1)
                result_df['person_box'] = result_df[['person_box_point1','person_box_point2','person_box_point3','person_box_point4']].mean(axis=1)
                X = scaler.transform(result_df.drop(columns=['filename']))
            except:
                return '''
                <!DOCTYPE html>
                <img src={0} class="img-responsive" width='400'>
                <p>Error: A prediction could not be made using this image
                <br>
                <i>Either a banana or person could not be detected.</i></p>'''.format(tag)

            # make final prediction with random forest
            model = load('models/randomforest.joblib')
            pred = model.predict(X)[0]
            ft = int(pred // 12)
            inch = round(pred % 12,1)
            return render_template('prediction_answer.html',variable=[f'{ft} ft {inch} in','',tag])
        else:
            return redirect(request.url)
    return render_template('prediction.html',Title='Height Prediction with RetinaNet Object Detection and Random Forest Regressor')

@application.route('/form', methods=['GET','POST'])
def form():
    if request.method == 'POST':
        comments = request.form['comments']
        c = client('s3')
        c.put_object(Body=comments,Bucket='bananaforscale',Key=f'comments/{uuid4().hex}.txt')
        return redirect('/success')
    return render_template('form.html')

if __name__ == '__main__':
    application.run(debug=True)
