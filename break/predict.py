import os
from keras.models import model_from_json
import keras.backend as K
import traceback
import random
from PIL import Image
import time
import numpy as np

filename = "./captcha_gen_and_break/gen/4_test/0.png"
img = Image.open(filename)
#print np.asarray(img).shape
data = img.resize([160 ,60])
data = np.multiply(data, 1 / 255.0)
data = np.asarray(data)
mf=open('./cnn.json')
model = model_from_json(mf.read())
mf.close()
model.load_weights('./cnn.h5')
preds = model.predict(np.reshape(data, (1, 60, 160, 3)))
K.clear_session()
pred = preds
num = str(pred[0].argmax()) + \
    str(pred[1].argmax()) + \
    str(pred[2].argmax()) + \
    str(pred[3].argmax())
print(num)
