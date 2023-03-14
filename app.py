from flask import Flask, request,render_template
import joblib
from joblib import Parallel, delayed
import base64
from io import BytesIO
import tensorflow as tf 
from tensorflow import keras
import numpy as np
import io
from PIL import Image
import cv2
import pickle
import numpy as np
from keras import backend as K
from tensorflow.keras import layers
app = Flask(__name__,static_url_path="/static")
model = keras.models.load_model('models\model_1_data_aug.h5')
model.summary()
xgb=joblib.load('xgb_new_clf.pkl')

import cv2

# img=cv2.imread('training_test/val\Potato___healthy\image (4).JPG')
# img=cv2.resize(img,(256,256))
# img=np.reshape(img,[1,256,256,3])



# outputs=[]
# for layer in model.layers:
#     img = layer(img)
#     outputs.append(img)

# prediction = xgb.predict(outputs[3])
# new_pred=prediction.argmax(axis=1)
# accuracy=xgb.predict_proba(outputs[3])
# print(np.max(accuracy)*100)

# class_names=['potato_early_blight','potato_late_blight','potato_healthy']
# print(outputs[3].shape)
# y_pred=xgb.predict(outputs[3])
# y_new=np.argmax(y_pred,axis=1)
# val=int(y_new.item())

# print(class_names[val])
class_names=['Apple___Apple_scab', 'Apple___Black_rot', 'Apple___Cedar_apple_rust', 'Apple___healthy', 'Cherry___Powdery_mildew', 'Cherry___healthy', 'Orange___Haunglongbing_(Citrus_greening)', 'Peach___Bacterial_spot', 'Peach___healthy', 'Pepper_bell___Bacterial_spot', 'Pepper_bell___healthy', 'Potato___Early_blight', 'Potato___Late_blight', 'Potato___healthy', 'Squash___Powdery_mildew', 'Strawberry___Leaf_scorch', 'Strawberry___healthy', 'Tomato___Bacterial_spot', 'Tomato___Early_blight', 'Tomato___Late_blight', 'Tomato___Leaf_Mold', 'Tomato___Septoria_leaf_spot', 'Tomato___Spider_mites Two-spotted_spider_mite', 'Tomato___Target_Spot', 'Tomato___Tomato_Yellow_Leaf_Curl_Virus', 'Tomato___healthy', 'Tomato_mosaic_virus', 'banana_healthy', 'banana_segatoka', 'banana_xamthomonas', 'orange_Canker', 'orange_Greening', 'orange_Scab', 'orange_healthy', 'rice_Bacterial_leaf _blight', 'rice_Brown_Spot', 'rice_Healthy', 'rice_Hispa', 'rice_Leaf_Blast', 'rice_Leaf_smut', 'soyabean_Caterpillar', 'soyabean_Diabrotica_speciosa', 'soybean_healthy', 'sugarcane_ Bacterial_Blight', 'sugarcane_ Infected', 'sugarcane_Healthy_corn', 'sugercane_red_ hot', 'tea_leaf_blight', 'tea_red_leaf_spot', 'tea_red_scab', 'wheat_Healthy', 'wheat_septoria', 'wheat_stripe_rust']


@app.route('/')
def index():
    return render_template('image_classification.html')

@app.route('/', methods=['POST'])
def predict():

   
  
    file = request.files['image']

    image=request.files.get('image').read()
    image = np.frombuffer(image, np.uint8)
    image = cv2.imdecode(image, cv2.IMREAD_COLOR)
    image=cv2.resize(image,(256,256))
    image_new=np.reshape(image,[1,256,256,3])
    outputs = []
    for layer in model.layers:
        image_new = layer(image_new)
        outputs.append(image_new)
    
    prediction = xgb.predict_proba(outputs[3])
    # new_pred=prediction.argmax(axis=1)
    # accuracy=xgb.predict_proba(outputs[3])
    # accuracy=np.max(prediction)*100
    # new_pred=model.predict(image_new)
    confidence=np.max(prediction)
    new_val=prediction.argmax(axis=1)
    val=int(new_val.item())
    result1=class_names[val];
    plant_name=result1.split('_')[0]
    image_str = base64.b64encode(cv2.imencode('.jpg', image)[1]).decode()

    return render_template('result.html',disease=result1,accuracy=np.max(confidence)*100,image=image_str,result1=plant_name)

if __name__ == '__main__':
    app.run(host='0.0.0.0',port=8080,debug=True)


