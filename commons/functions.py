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

from deepface.basemodels import Boosting
from git import Repo
import time

from tqdm import tqdm
import pickle
from deepface.commons import distance as dst
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

def detect_face(img, detector_backend = 'opencv', grayscale = False, hard_detection_failure = True, align = True):

	img_region = [0, 0, img.shape[0], img.shape[1]]

	#----------------------------------------------
	#people would like to skip detection and alignment if they already have pre-processed images
	if detector_backend == 'skip':
		return img, img_region

	#----------------------------------------------

	#detector stored in a global variable in FaceDetector object.
	#this call should be completed very fast because it will return found in memory
	#it will not build face detector model in each call (consider for loops)
	face_detector = FaceDetector.build_model(detector_backend)

	try:
		detected_face, img_region = FaceDetector.detect_face(face_detector, detector_backend, img, align)
	except: #if detected face shape is (0, 0) and alignment cannot be performed, this block will be run
		detected_face = None

	if (isinstance(detected_face, np.ndarray)):
		return detected_face, img_region
	else:
		if detected_face == None:
			if hard_detection_failure != True:
				return img, img_region
			else:
				raise ValueError("Face could not be detected. Please confirm that the picture is a face photo or consider to set hard_detection_failure param to False.")

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

def preprocess_face(img, target_size=(224, 224), grayscale = False, hard_detection_failure = True, detector_backend = 'opencv', return_region = False, align = True):

	#img might be path, base64 or numpy array. Convert it to numpy whatever it is.
	img = load_image(img)
	base_img = img.copy()

	img, region = detect_face(img = img, detector_backend = detector_backend, grayscale = grayscale, hard_detection_failure = hard_detection_failure, align = align)

	if img.shape[0] == 0 or img.shape[1] == 0:
		if hard_detection_failure == True:
			raise ValueError("Detected face shape is ", img.shape,". Consider to set hard_detection_failure argument to False.")
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

def get_models(build_model, model_name ='VGG-Face', model = None):
	if model == None:

		if model_name == 'Ensemble':
			print("Ensemble learning enabled")
			models = Boosting.loadModel()

		else: #model is not ensemble
			model = build_model(model_name)
			models = {}
			models[model_name] = model

	else: #model != None
		print("Already built model is passed")

		if model_name == 'Ensemble':
			Boosting.validate_model(model)
			models = model.copy()
		else:
			models = {}
			models[model_name] = model
	
	return models

def get_model_and_metric_names(model_name = 'VGG-Face', distance_metric = 'cosine'):

	if model_name == 'Ensemble':
		model_names = ['VGG-Face', 'Facenet', 'OpenFace', 'DeepFace']
		metric_names = ['cosine', 'euclidean', 'euclidean_l2']
	elif model_name != 'Ensemble':
		model_names = []; metric_names = []
		model_names.append(model_name)
		metric_names.append(distance_metric)
	
	return (model_names, metric_names)

def check_change(db_path="."): #this whole function can be called in its own thread
	#User might add an image to his database, might rename, move, delete, etc...
	try:
		repo = Repo(db_path) #this throws an error if it's not git-initialized already
	except Exception:
		repo = Repo.init(db_path, bare=False) #initializing
	
	img_type = (".jpeg", ".jpg", ".png", ".bmp")
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


def create_representation_file(file_name, model_names, models, db_path = ".", model_name = 'VGG-Face', hard_detection_failure = True, detector_backend = 'opencv', align = True, normalization = 'base'):
	employees = []

	for r, d, f in os.walk(db_path): # r=root, d=directories, f = files
		for file in f:
			if ('.jpg' in file.lower()) or ('.png' in file.lower()):
				exact_path = r + "/" + file
				employees.append(exact_path)

	if len(employees) == 0:
		raise ValueError("There is no image in ", db_path," folder! Validate .jpg or .png files exist in this path.")

	#------------------------
	#find representations for db images

	representations = []

	pbar = tqdm(range(0,len(employees)), desc='Finding representations', disable = prog_bar)

	#for employee in employees:
	for index in pbar:
		employee = employees[index]

		instance = []
		instance.append(employee)

		for j in model_names:
			custom_model = models[j]

			representation = represent(img_path = employee
				, model_name = model_name, model = custom_model
				, hard_detection_failure = hard_detection_failure, detector_backend = detector_backend
				, align = align
				, normalization = normalization)

			instance.append(representation)

		#-------------------------------

		representations.append(instance)

	f = open(db_path+'/'+file_name, "wb")
	pickle.dump(representations, f)
	f.close()

	print("Representations stored in ",db_path,"/",file_name," file. Please delete this file when you add new identities in your database.")
	return representations

def get_df(representations, model_names, model_name = 'VGG-Face'):
	if model_name != 'Ensemble':
		df = pd.DataFrame(representations, columns = ["identity", "%s_representation" % (model_name)])
	else: #ensemble learning

		columns = ['identity']
		[columns.append('%s_representation' % i) for i in model_names]

		df = pd.DataFrame(representations, columns = columns)
	return df

def get_find_result(df, represent, img_paths, model_names, models, metric_names, model_name  ='VGG-Face', hard_detection_failure = True, detector_backend = 'opencv', align = True, normalization = 'base', prog_bar = True):
	df_base = df.copy() #df will be filtered in each img. we will restore it for the next item.

	resp_obj = []

	global_pbar = tqdm(range(0, len(img_paths)), desc='Analyzing', disable = prog_bar)
	for j in global_pbar:
		img_path = img_paths[j]

		#find representation for passed image

		for j in model_names:
			custom_model = models[j]

			target_representation = represent(img_path = img_path
				, model_name = model_name, model = custom_model
				, hard_detection_failure = hard_detection_failure, detector_backend = detector_backend
				, align = align
				, normalization = normalization)
			#print("target_representation list is : ", target_representation) #this is a vector of length 2622 in case of VGG-Face model_name
			for k in metric_names:
				distances = []
				for index, instance in df.iterrows():
					source_representation = instance["%s_representation" % (j)]

					if k == 'cosine':
						distance = dst.findCosineDistance(source_representation, target_representation)
					elif k == 'euclidean':
						distance = dst.findEuclideanDistance(source_representation, target_representation)
					elif k == 'euclidean_l2':
						distance = dst.findEuclideanDistance(dst.l2_normalize(source_representation), dst.l2_normalize(target_representation))
					else :
						raise Exception("Not a valid distance_metric")

					distances.append(distance)

				#---------------------------

				if model_name == 'Ensemble' and j == 'OpenFace' and k == 'euclidean':
					continue
				else:
					df["%s_%s" % (j, k)] = distances

					if model_name != 'Ensemble':
						threshold = dst.findThreshold(j, k)
						df = df.drop(columns = ["%s_representation" % (j)])
						df = df[df["%s_%s" % (j, k)] <= threshold]

						df = df.sort_values(by = ["%s_%s" % (j, k)], ascending=True).reset_index(drop=True)

						resp_obj.append(df)
						df = df_base.copy() #restore df for the next iteration

		#----------------------------------

		if model_name == 'Ensemble':

			feature_names = []
			for j in model_names:
				for k in metric_names:
					if model_name == 'Ensemble' and j == 'OpenFace' and k == 'euclidean':
						continue
					else:
						feature = '%s_%s' % (j, k)
						feature_names.append(feature)

			#print(df.head())

			x = df[feature_names].values

			#--------------------------------------

			boosted_tree = Boosting.build_gbm()

			y = boosted_tree.predict(x)

			verified_labels = []; scores = []
			for i in y:
				verified = np.argmax(i) == 1
				score = i[np.argmax(i)]

				verified_labels.append(verified)
				scores.append(score)

			df['verified'] = verified_labels
			df['score'] = scores

			df = df[df.verified == True]
			#df = df[df.score > 0.99] #confidence score
			df = df.sort_values(by = ["score"], ascending=False).reset_index(drop=True)
			df = df[['identity', 'verified', 'score']]

			resp_obj.append(df)
			df = df_base.copy() #restore df for the next iteration

		#----------------------------------
	return resp_obj
