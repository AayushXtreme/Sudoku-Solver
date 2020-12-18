## fix the digits background
import cv2
import numpy as np
import tensorflow as tf
from skimage.segmentation import clear_border


# load model architecture
with open('models/model.json', 'r') as file:
        json_file =  file.read()

model = tf.keras.models.model_from_json(json_file)
# model.summary()
model.load_weights('models/model.h5')


# preprocess image
def preprocess(im):
    im = cv2.resize(im, (28, 28), interpolation=cv2.INTER_AREA)
    im = cv2.cvtColor(im, cv2.COLOR_BGR2GRAY)
    thresh = cv2.threshold(im, 0, 255, cv2.THRESH_BINARY_INV | cv2.THRESH_OTSU)[1]
    cell = clear_border(thresh)
    digit = cell.astype("float") / 255.0
    digit = tf.keras.preprocessing.image.img_to_array(digit)
    digit = np.expand_dims(digit, axis=0)
    return digit, cell

def ocr(im, debug=False):
    digit = 0
    cell, thresh = preprocess(im)
    if cell.sum() < 10:
        digit = 0
    else:
        pred = model.predict(cell).argmax(axis=1)[0]
        digit = pred

    # display 
    if debug:
        print(digit)
        cv2.imshow('img' ,im)
        cv2.waitKey(0)
        cv2.destroyAllWindows()
    return digit




