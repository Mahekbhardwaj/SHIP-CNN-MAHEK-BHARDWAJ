
import numpy as np
import os
import tensorflow 
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
from flask import Flask , request, render_template
from werkzeug.utils import secure_filename
from gevent.pywsgi import WSGIServer

app = Flask(__name__)
#path = "C:/Users/mini/Downloads/CNN Flask/CNN Flask/ships.h5"
#model=tensorflow.lite.TFLiteConverter.from_keras_model("ships.h5") 
model = load_model("ships.h5")
                 
@app.route('/')
def index():
    return render_template('index.html')

@app.route('/predict',methods = ['GET','POST'])
def upload():
    if request.method == 'POST':
        f = request.files['image']
        print("current path")
        basepath = os.path.dirname(__file__)
        print("current path", basepath)
        filepath = os.path.join(basepath,'uploads',f.filename)
        print("upload folder is ", filepath)
        f.save(filepath)
        
        img = image.load_img(filepath,target_size = (128,128)) 
        x = image.img_to_array(img)
        print(x)
        x = np.expand_dims(x,axis =0)
        print(x)
        #model.compile() 
        preds = model.predict_classes(x)
        print("prediction",preds)
        index = ['aircraft carrier','bulker ships','cruise ships','drilling rigs','fire fighter ships','fishing vessels','inland dry cargo','motor vessels','restaurant ships','submarines']
        text = "The classified ships is : " + str(index[preds[0]])
    return text
if __name__ == '__main__':
    app.run(debug =False) 
