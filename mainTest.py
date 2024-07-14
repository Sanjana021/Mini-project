import cv2
import numpy as np
from keras.models import load_model
from PIL import Image
model=load_model('brainTumorDetectionCategorical.h5')

image = cv2.imread('C:\\Users\\Prapthi\\Desktop\\BTC\\prediction\\pred0.jpg')

img = Image.fromarray(image)
img=img.resize((64,64))
img=np.array(img)
print(img)
img = np.expand_dims(img,axis=0)

predictions = model.predict(img)


predicted_class = predictions.argmax(axis=-1)

print("Predicted class:", predicted_class)