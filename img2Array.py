from PIL import Image
import numpy as np
import os

BASE_DIR = 'dataset'
train_folder = BASE_DIR+'/Test/digits/digit_'
data = np.empty((0,1025),dtype=int)
print(data.dtype)
for i in range(10):
	im_dir=train_folder+str(i)+'/'
	images = sorted(os.listdir(im_dir))
	print("Dir: {} and label : {}".format(im_dir,i))
	for img in images:
		image = Image.open(im_dir+img)
		# convert image to numpy array
		im_data = np.append(np.asarray(image).flatten(),i)
		data = np.vstack((data,im_data)) #add flatened array to data
	print(data.shape)
print("Final shape: ",data.shape)
np.savetxt("Test.csv", data, delimiter=",", fmt="%d")
