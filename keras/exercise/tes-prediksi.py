import numpy as np
from keras.preprocessing import image
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import klasifikasi as classifier
# %matplotlib inline # magic command in notebook to skip calling plt.show()
 
# Change it to your filename
fn = '/Users/fadhilhanri/Documents/Challenge/python/test-image/sample_room.png'
 
# Predicting images
path = fn
img = image.load_img(path, target_size=(150,150))
imgplot = plt.imshow(img)
x = image.img_to_array(img)
x = np.expand_dims(x, axis=0)
 
images = np.vstack([x])
classes = classifier.model.predict(images, batch_size=10)
  
print(fn)
if classes==0:
  print('clean')
  plt.show()
else:
  print('messy')
  plt.show()