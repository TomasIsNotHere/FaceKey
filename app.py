from flask import Flask,render_template, request, jsonify
from PIL import Image
import numpy as np
import io
import base64
from matplotlib import pyplot as plt 
import tensorflow as tf 
from tensorflow.keras.models import load_model

app = Flask(__name__)

gpus = tf.config.experimental.list_physical_devices('GPU')
for gpu in gpus:
    tf.config.experimental.set_memory_growth(gpu, True)
    
tf.config.list_physical_devices('GPU')

faceTracker = load_model('bbmodel/done.h5')
spoofModel = load_model('smodel/done.h5')


def to_str(var):
    return str(list(np.reshape(np.asarray(var), (1, np.size(var)))[0]))[1:-1]

@app.route('/')
def index():
    return render_template('main.html')

@app.route('/facekey')
def facekey():
    return render_template('index.html')

@app.route('/endingpage')
def endingpage():
    return render_template('endingpage.html')

@app.route('/detections', methods=['POST'])
def detections():

    data = request.json
    image_data_url = data.get('image')
    image_data = image_data_url.split(',')[1]  # Remove the data URL header
    image = Image.open(io.BytesIO(base64.b64decode(image_data)))  # Convert base64 to image

    # Convert image to numpy array
    image_np = np.array(image)
    resized = tf.image.resize(image_np, (120,120))
    resized = np.expand_dims(resized/255,0)
    
    yhat = faceTracker.predict(resized)
    sample_coords = yhat[1][0]

    if yhat[0] > 0.5:
        minVal = tuple(np.multiply(sample_coords[:2], [640,480]).astype(int))
        maxVal = tuple(np.multiply(sample_coords[2:], [640,480]).astype(int))
        
        x, y, w, h = minVal[0], minVal[1], maxVal[0]-minVal[0]+1,  maxVal[1]-minVal[1]+1
        
        roi = image_np[y:y+h, x:x+w]
        predRoi = roi.shape

        if predRoi[0] < 150:
            status = 'get closer,'+ str(predRoi)+','+to_str(x)+','+to_str(y)+','+to_str(w)+','+to_str(h)
        else:
            spoofImg = tf.image.resize(roi, (256,256))
            spoofPred = spoofModel.predict(np.expand_dims(spoofImg/255, 0))
            
            if spoofPred > 0.98:
                status = 'real,' + str(predRoi)+','+to_str(x)+','+to_str(y)+','+to_str(w)+','+to_str(h)
            else: 
                status = 'fake,' + str(predRoi)+','+to_str(x)+','+to_str(y)+','+to_str(w)+','+to_str(h)

        response = {'status': status}
    else: 
        x,y,w,h = 0,0,0,0
        status = 'noface,'+x+','+y+','+w+','+h
        response = {'status': status}
    #,'x': x ,'y': y ,'w': w ,'h': h 
    return jsonify(response)
if __name__ == '__main__':
    app.run(host="0.0.0.0",debug=True)
