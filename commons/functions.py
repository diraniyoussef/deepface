import os
import numpy as np
import pandas as pd
import cv2
import base64
from pathlib import Path

from deepface.detectors import FaceDetector

import tensorflow as tf
tf_version = tf.__version__
tf_major_version = int(tf_version.split(".")[0])
tf_minor_version = int(tf_version.split(".")[1])

if tf_major_version == 1:
	import keras
	from keras.preprocessing.image import load_img, save_img, img_to_array
	from keras.applications.imagenet_utils import preprocess_input
	from keras.preprocessing import image
elif tf_major_version == 2:
	from tensorflow import keras
	from tensorflow.keras.preprocessing.image import load_img, save_img, img_to_array
	from tensorflow.keras.applications.imagenet_utils import preprocess_input
	from tensorflow.keras.preprocessing import image

from git import Repo
import time
#--------------------------------------------------

def initialize_input(img1_path, img2_path = None):

	if type(img1_path) == list:
		bulkProcess = True
		img_list = img1_path.copy()
	else:
		bulkProcess = False

		if (
			(type(img2_path) == str and img2_path != None) #exact image path, base64 image
			or (isinstance(img2_path, np.ndarray) and img2_path.any()) #numpy array
		):
			img_list = [[img1_path, img2_path]]
		else: #analyze function passes just img1_path
			img_list = [img1_path]

	return img_list, bulkProcess

def initializeFolder():

	home = str(Path.home())

	if not os.path.exists(home+"/.deepface"):
		os.mkdir(home+"/.deepface")
		print("Directory ",home,"/.deepface created")

	if not os.path.exists(home+"/.deepface/weights"):
		os.mkdir(home+"/.deepface/weights")
		print("Directory ",home,"/.deepface/weights created")

def loadBase64Img(uri):
   encoded_data = uri.split(',')[1]
   nparr = np.fromstring(base64.b64decode(encoded_data), np.uint8)
   img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
   return img

def load_image(img):

	exact_image = False
	if type(img).__module__ == np.__name__:
		exact_image = True

	base64_img = False
	if len(img) > 11 and img[0:11] == "data:image/":
		base64_img = True

	#---------------------------

	if base64_img == True:
		img = loadBase64Img(img)

	elif exact_image != True: #image path passed as input
		if os.path.isfile(img) != True:
			raise ValueError("Confirm that ",img," exists")

		img = cv2.imread(img)

	return img

def detect_face(img, detector_backend = 'opencv', grayscale = False, enforce_detection = True, align = True):

	img_region = [0, 0, img.shape[0], img.shape[1]]

	#----------------------------------------------
	#people would like to skip detection and alignment if they already have pre-processed images
	if detector_backend == 'skip':
		#Added by Youssef
		#print("detector_backend is set to skip")
		#End adding by Youssef		
		return img, img_region

	#----------------------------------------------

	#detector stored in a global variable in FaceDetector object.
	#this call should be completed very fast because it will return found in memory
	#it will not build face detector model in each call (consider for loops)
	face_detector = FaceDetector.build_model(detector_backend)

	try:
		detected_face, img_region = FaceDetector.detect_face(face_detector, detector_backend, img, align)
		#Added by Youssef
		#print("detected_face, img_region ", detected_face, img_region)
		#End adding by Youssef
	except: #if detected face shape is (0, 0) and alignment cannot be performed, this block will be run
		#Added by Youssef
		#print("detected_face is None")
		#End adding by Youssef		
		detected_face = None

	if (isinstance(detected_face, np.ndarray)):
		#Added by Youssef
		#print("isinstance(detected_face, np.ndarray) is true")
		#print("detected_face, img_region ", detected_face, img_region)
		#End adding by Youssef
		return detected_face, img_region
	else:
		if detected_face == None:
			if enforce_detection != True:
				#Added by Youssef
				#print("detected_face is none and enforce_detection is not true")
				#End adding by Youssef
				return img, img_region
			else:
				#Added by Youssef
				#print("detected_face is none and enforce_detection is true")
				#End adding by Youssef
				raise ValueError("Face could not be detected. Please confirm that the picture is a face photo or consider to set enforce_detection param to False.")

def normalize_input(img, normalization = 'base'):

	#issue 131 declares that some normalization techniques improves the accuracy

	if normalization == 'base':
		return img
	else:
		#@trevorgribble and @davedgd contributed this feature

		img *= 255 #restore input in scale of [0, 255] because it was normalized in scale of  [0, 1] in preprocess_face

		if normalization == 'raw':
			pass #return just restored pixels

		elif normalization == 'Facenet':
			mean, std = img.mean(), img.std()
			img = (img - mean) / std

		elif(normalization=="Facenet2018"):
			# simply / 127.5 - 1 (similar to facenet 2018 model preprocessing step as @iamrishab posted)
			img /= 127.5
			img -= 1

		elif normalization == 'VGGFace':
			# mean subtraction based on VGGFace1 training data
			img[..., 0] -= 93.5940
			img[..., 1] -= 104.7624
			img[..., 2] -= 129.1863

		elif(normalization == 'VGGFace2'):
			# mean subtraction based on VGGFace2 training data
			img[..., 0] -= 91.4953
			img[..., 1] -= 103.8827
			img[..., 2] -= 131.0912

		elif(normalization == 'ArcFace'):
			#Reference study: The faces are cropped and resized to 112×112,
			#and each pixel (ranged between [0, 255]) in RGB images is normalised
			#by subtracting 127.5 then divided by 128.
			img -= 127.5
			img /= 128

	#-----------------------------

	return img

def preprocess_face(img, target_size=(224, 224), grayscale = False, enforce_detection = True, detector_backend = 'opencv', return_region = False, align = True):

	#img might be path, base64 or numpy array. Convert it to numpy whatever it is.
	img = load_image(img)
	base_img = img.copy()

	img, region = detect_face(img = img, detector_backend = detector_backend, grayscale = grayscale, enforce_detection = enforce_detection, align = align)

	if img.shape[0] == 0 or img.shape[1] == 0:
		if enforce_detection == True:
			raise ValueError("Detected face shape is ", img.shape,". Consider to set enforce_detection argument to False.")
		else: #restore base image
			img = base_img.copy()

	#--------------------------

	#post-processing
	if grayscale == True:
		img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

	#---------------------------------------------------
	#resize image to expected shape

	# img = cv2.resize(img, target_size) #resize causes transformation on base image, adding black pixels to resize will not deform the base image

	factor_0 = target_size[0] / img.shape[0]
	factor_1 = target_size[1] / img.shape[1]
	factor = min(factor_0, factor_1)

	dsize = (int(img.shape[1] * factor), int(img.shape[0] * factor))
	img = cv2.resize(img, dsize)

	# Then pad the other side to the target size by adding black pixels
	diff_0 = target_size[0] - img.shape[0]
	diff_1 = target_size[1] - img.shape[1]
	if grayscale == False:
		# Put the base image in the middle of the padded image
		img = np.pad(img, ((diff_0 // 2, diff_0 - diff_0 // 2), (diff_1 // 2, diff_1 - diff_1 // 2), (0, 0)), 'constant')
	else:
		img = np.pad(img, ((diff_0 // 2, diff_0 - diff_0 // 2), (diff_1 // 2, diff_1 - diff_1 // 2)), 'constant')

	#double check: if target image is not still the same size with target.
	if img.shape[0:2] != target_size:
		img = cv2.resize(img, target_size)

	#---------------------------------------------------

	#normalizing the image pixels

	img_pixels = image.img_to_array(img) #what this line doing? must?
	img_pixels = np.expand_dims(img_pixels, axis = 0)
	img_pixels /= 255 #normalize input in [0, 1]

	#---------------------------------------------------

	if return_region == True:
		return img_pixels, region
	else:
		return img_pixels

def find_input_shape(model):

	#face recognition models have different size of inputs

	input_shape = model.layers[0].input_shape

	#my environment returns (None, 224, 224, 3) but some people mentioned that they got [(None, 224, 224, 3)]. I think this is because of version issue.
	if type(input_shape) == list:
		input_shape = input_shape[0][1:3]
	else:
		input_shape = input_shape[1:3]

	#----------------------
	#issue 289: it seems that tf 2.5 expects you to resize images with (x, y)
	#whereas its older versions expect (y, x)

	if tf_major_version == 2 and tf_minor_version >= 5:
		x = input_shape[0]; y = input_shape[1]
		input_shape = (y, x)

	#----------------------

	if type(input_shape) == list: #issue 197: some people got array here instead of tuple
		input_shape = tuple(input_shape)

	return input_shape

def check_change(db_path="."): #this whole function can be called in its own thread
	#User might add an image to his database, might rename, move, delete, etc...
	try:
		repo = Repo(db_path) #this throws an error if it's not git-initialized already
	except Exception:
		repo = Repo.init(db_path, bare=False) #initializing
	
	img_type = (".jpeg",".jpg",".png")
	#check for untracked files
	to_be_added_images_list = []
	for f in repo.untracked_files: 
		if(f.endswith(img_type)):
			to_be_added_images_list.append(f) #images aren't tracked yet
	#check for unstaged files (which are already tracked)
	to_be_removed_images_list = []
	for x in repo.index.diff(None):
		if(x.b_path.endswith(img_type)):#added in case someone messes with the repo, like adds something which isn't an image to the git staging area i.e. tracking it.
			if(x.change_type == 'M'): #not sure how an image file could be modified, anyway
				to_be_added_images_list.append(x.b_path)
			elif(x.change_type == 'D'):
				to_be_removed_images_list.append(x.b_path)
	#Although user might rename an image and it might be worthy to trace such thing, but it is not a direct process in git so postponed. A renamed file is deleted and made new when it comes to git and us.
	
	#Goal Section
	#this function check_change is to trace the files to update the pkl vector representation of the images quickly without having to read the whole database for a tiny change made by the user like adding an image or deleting another.
	
	#End of Goal Section
		
	#Now staging images to make them ready for committing and to clear the staging area for future changes
	commit = False
	if(len(to_be_added_images_list) > 0):
		#Attempting to add the files to the staging (committing) area in order to finally commit
		commit = True
		try:
			repo.index.add(to_be_added_images_list) #images are now tracked and are in the committing area.
		except Exception as err:
			print("Git error while adding untracked file(s)", err)	
	if(len(to_be_removed_images_list) > 0):
		commit = True
		#removing images
		try:
        repo.index.remove(to_be_removed_images_list) #it removes the file after being added or even committed, then it returns it back to being untracked if it ever existed again, I guess
		except Exception as err:
			print("Git error while adding untracked file(s)", err)
	#committing 
	if(commit):
		repo.index.commit("commit at " + time.ctime().replace(" ", "_"))