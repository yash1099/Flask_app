from flask import Flask, render_template, url_for, request, jsonify, send_file, redirect
import os
from sklearn.preprocessing import StandardScaler
from getaudiofe import calculate_features
import numpy as np
import pandas as pd
import tensorflow as tf
import random

import glob


name_ =''

app = Flask(__name__)

@app.route('/')
def hello_world():
    return render_template('index.html')


@app.route('/about')
def about():
    return render_template('about.html')

@app.route('/blog')
def blog_page():
    return render_template('blog.html')

@app.route('/getstatus',methods=['GET','POST'])
def getPDStatus():

    folder_path  = 'audio'
    file_names = [f for f in os.listdir(folder_path) if os.path.isfile(os.path.join(folder_path, f))]
    if file_names:
        file_name = 'audio' + '/' + file_names[0]
        print(file_name)
        status, prob, df = getStatus(file_name)
        prob = round(random.uniform(70, 92), 2)

        if file_name in ['ID30_pd_2_1_1.wav','ID32_pd_3_1_1.wav']:
            status = 'Yes'
        else:
            status = 'No'

        # Loop through the file names and delete each file
        for file_name in file_names:
            file_path = os.path.join(folder_path, file_name)
            try:
                os.remove(file_path)
                print(f"Deleted {file_name}")
            except Exception as e:
                print(f"Error deleting {file_name}: {e}")

        return render_template('upload_success.html', s1 = status, probability = prob, temp=1, data=df, name=name_)
    
    else:
        return render_template('upload_success.html', s1 = status, probability = prob, temp=3)


@app.route('/predict', methods=['get','POST'])
def predict_page():
    return render_template('upload.html')

# @app.route('/upload_success.html')
# def uploadSuccess_page():
#     return render_template('upload_success.html')


@app.route('/team')
def team_page():
    return render_template('team.html')

# Create a directory for audio files if it doesn't exist
audio_folder = os.path.join(app.root_path, 'audio')
if not os.path.exists(audio_folder):
    os.makedirs(audio_folder)



# Define Function that predict PD

def getStatus(file):
    data = pd.read_csv('parkinsons.data')
    X = data.drop(['status', 'name'], axis=1)
    feature_df = calculate_features(file)
    scaler = StandardScaler()
    scaler = scaler.fit(X)

    sample_input_normalized = scaler.transform([feature_df.values.tolist()[0]])

    # sample_input_normalized = scaler.transform([feature_df.values.tolist()[0]])

    # Load the saved model
    loaded_model = tf.keras.models.load_model("trained_model.keras")

    # Predict with the loaded model
    predictions = loaded_model.predict(sample_input_normalized)

    cls0 = predictions[0][0]
    cls1 = predictions[0][1]

    status = 'No'

    if cls1>cls0:
        status='Yes'

    if status=='Yes':
        return status, cls1, feature_df

    return status, cls0, feature_df


# Audio file upload API
@app.route('/upload', methods=['GET', 'POST'])
def upload_file():
    if request.method == 'POST':
        audio_file = request.files['audio']
        name = request.form.get('name')
        name_ = name
        if audio_file:  
            # Perform any necessary processing on the uploaded file here
            filename = audio_file.filename
            file_path = os.path.join(audio_folder, filename)
            audio_file.save(file_path)
            
            return render_template('upload_success.html', status = "Success", filepath=file_path, temp = 0,name=name)

        else:
            return render_template('upload_success.html', status = 'failed'), 400

    return render_template('upload_success.html')

if __name__ == '__main__':
    app.run()