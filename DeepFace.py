import warnings
warnings.filterwarnings("ignore")
 
import os
#os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

import time
from os import path
import numpy as np
import pandas as pd
from tqdm import tqdm
import pickle

from deepface.basemodels import VGGFace, OpenFace, Facenet, Facenet512, FbDeepFace, DeepID, DlibWrapper, ArcFace, Boosting
from deepface.extendedmodels import Age, Gender, Race, Emotion
from deepface.commons import functions, functions1, realtime, distance as dst

import tensorflow as tf
tf_version = int(tf.__version__.split(".")[0])
if tf_version == 2:
	import logging
	tf.get_logger().setLevel(logging.ERROR)

from deepface.detectors import FaceDetector
import cv2
import re
import threading

def build_model(model_name):

	"""
	This function builds a deepface model
	Parameters:
		model_name (string): face recognition or facial attribute model
			VGG-Face, Facenet, OpenFace, DeepFace, DeepID for face recognition
			Age, Gender, Emotion, Race for facial attributes

	Returns:
		built deepface model
	"""

	global model_obj #singleton design pattern

	models = {
		'VGG-Face': VGGFace.loadModel,
		'OpenFace': OpenFace.loadModel,
		'Facenet': Facenet.loadModel,
		'Facenet512': Facenet512.loadModel,
		'DeepFace': FbDeepFace.loadModel,
		'DeepID': DeepID.loadModel,
		'Dlib': DlibWrapper.loadModel,
		'ArcFace': ArcFace.loadModel,
		'Emotion': Emotion.loadModel,
		'Age': Age.loadModel,
		'Gender': Gender.loadModel,
		'Race': Race.loadModel
	}

	if not "model_obj" in globals():
		model_obj = {}

	if not model_name in model_obj.keys():
		model = models.get(model_name)
		if model:
			model = model() #Youssef - here is the building
			model_obj[model_name] = model
			#print(model_name," built")
		else:
			raise ValueError('Invalid model_name passed - {}'.format(model_name))

	return model_obj[model_name]

def verify(img1_path, img2_path = '', model_name = 'VGG-Face', distance_metric = 'cosine', model = None, hard_detection_failure = True, detector_backend = 'opencv', align = True, prog_bar = True, normalization = 'base'):

	"""
	This function verifies an image pair is same person or different persons.

	Parameters:
		img1_path, img2_path: exact image path, numpy array or based64 encoded images could be passed. If you are going to call verify function for a list of image pairs, then you should pass an array instead of calling the function in for loops.

		e.g. img1_path = [
			['img1.jpg', 'img2.jpg'],
			['img2.jpg', 'img3.jpg']
		]

		model_name (string): VGG-Face, Facenet, OpenFace, DeepFace, DeepID, Dlib, ArcFace or Ensemble

		distance_metric (string): cosine, euclidean, euclidean_l2

		model: Built deepface model. A face recognition model is built every call of verify function. You can pass pre-built face recognition model optionally if you will call verify function several times.

			model = DeepFace.build_model('VGG-Face')

		hard_detection_failure (boolean): If any face could not be detected in an image, then verify function will return exception. Set this to False not to have this exception. This might be convenient for low resolution images.

		detector_backend (string): set face detector backend as retinaface, mtcnn, opencv, ssd or dlib

		prog_bar (boolean): enable/disable a progress bar

	Returns:
		Verify function returns a dictionary. If img1_path is a list of image pairs, then the function will return list of dictionary.

		{
			"verified": True
			, "distance": 0.2563
			, "max_threshold_to_verify": 0.40
			, "model": "VGG-Face"
			, "similarity_metric": "cosine"
		}

	"""

	tic = time.time()

	img_list, bulkProcess = functions.initialize_input(img1_path, img2_path)

	resp_objects = []

	#--------------------------------

	if model_name == 'Ensemble':
		model_names = ["VGG-Face", "Facenet", "OpenFace", "DeepFace"]
		metrics = ["cosine", "euclidean", "euclidean_l2"]
	else:
		model_names = []; metrics = []
		model_names.append(model_name)
		metrics.append(distance_metric)

	#--------------------------------

	if model == None:
		if model_name == 'Ensemble':
			models = Boosting.loadModel()
		else:
			model = build_model(model_name)
			models = {}
			models[model_name] = model
	else:
		if model_name == 'Ensemble':
			Boosting.validate_model(model)
			models = model.copy()
		else:
			models = {}
			models[model_name] = model

	#------------------------------

	disable_option = (False if len(img_list) > 1 else True) or not prog_bar

	pbar = tqdm(range(0,len(img_list)), desc='Verification', disable = disable_option)

	for index in pbar:

		instance = img_list[index]

		if type(instance) == list and len(instance) >= 2:
			img1_path = instance[0]; img2_path = instance[1]

			ensemble_features = []

			for i in  model_names:
				custom_model = models[i]

				#img_path, model_name = 'VGG-Face', model = None, hard_detection_failure = True, detector_backend = 'mtcnn'
				img1_representation = represent(img_path = img1_path
						, model_name = model_name, model = custom_model
						, hard_detection_failure = hard_detection_failure, detector_backend = detector_backend
						, align = align
						, normalization = normalization
						)

				img2_representation = represent(img_path = img2_path
						, model_name = model_name, model = custom_model
						, hard_detection_failure = hard_detection_failure, detector_backend = detector_backend
						, align = align
						, normalization = normalization
						)

				#----------------------
				#find distances between embeddings

				for j in metrics:

					if j == 'cosine':
						distance = dst.findCosineDistance(img1_representation, img2_representation)
					elif j == 'euclidean':
						distance = dst.findEuclideanDistance(img1_representation, img2_representation)
					elif j == 'euclidean_l2':
						distance = dst.findEuclideanDistance(dst.l2_normalize(img1_representation), dst.l2_normalize(img2_representation))
					else:
						raise ValueError("Invalid distance_metric passed - ", distance_metric)

					distance = np.float64(distance) #causes trobule for euclideans in api calls if this is not set (issue #175)
					#----------------------
					#decision

					if model_name != 'Ensemble':

						threshold = dst.findThreshold(i, j)

						if distance <= threshold:
							identified = True
						else:
							identified = False

						resp_obj = {
							"verified": identified
							, "distance": distance
							, "max_threshold_to_verify": threshold
							, "model": model_name
							, "similarity_metric": distance_metric

						}

						if bulkProcess == True:
							resp_objects.append(resp_obj)
						else:
							return resp_obj

					else: #Ensemble

						#this returns same with OpenFace - euclidean_l2
						if i == 'OpenFace' and j == 'euclidean':
							continue
						else:
							ensemble_features.append(distance)

			#----------------------

			if model_name == 'Ensemble':

				boosted_tree = Boosting.build_gbm()

				prediction = boosted_tree.predict(np.expand_dims(np.array(ensemble_features), axis=0))[0]

				verified = np.argmax(prediction) == 1
				score = prediction[np.argmax(prediction)]

				resp_obj = {
					"verified": verified
					, "score": score
					, "distance": ensemble_features
					, "model": ["VGG-Face", "Facenet", "OpenFace", "DeepFace"]
					, "similarity_metric": ["cosine", "euclidean", "euclidean_l2"]
				}

				if bulkProcess == True:
					resp_objects.append(resp_obj)
				else:
					return resp_obj

			#----------------------

		else:
			raise ValueError("Invalid arguments passed to verify function: ", instance)

	#-------------------------

	toc = time.time()

	if bulkProcess == True:

		resp_obj = {}

		for i in range(0, len(resp_objects)):
			resp_item = resp_objects[i]
			resp_obj["pair_%d" % (i+1)] = resp_item

		return resp_obj

def analyze(img_path, actions = ['emotion', 'age', 'gender', 'race'] , models = {}, hard_detection_failure = True, detector_backend = 'opencv', prog_bar = True):

	"""
	This function analyzes facial attributes including age, gender, emotion and race

	Parameters:
		img_path: exact image path, numpy array or base64 encoded image could be passed. If you are going to analyze lots of images, then set this to list. e.g. img_path = ['img1.jpg', 'img2.jpg']

		actions (list): The default is ['age', 'gender', 'emotion', 'race']. You can drop some of those attributes.

		models: facial attribute analysis models are built in every call of analyze function. You can pass pre-built models to speed the function up.

			models = {}
			models['age'] = DeepFace.build_model('Age')
			models['gender'] = DeepFace.build_model('Gender')
			models['emotion'] = DeepFace.build_model('Emotion')
			models['race'] = DeepFace.build_model('race')

		hard_detection_failure (boolean): The function throws exception if a face could not be detected. Set this to True if you don't want to get exception. This might be convenient for low resolution images.

		detector_backend (string): set face detector backend as retinaface, mtcnn, opencv, ssd or dlib.

		prog_bar (boolean): enable/disable a progress bar
	Returns:
		The function returns a dictionary. If img_path is a list, then it will return list of dictionary.

		{
			"region": {'x': 230, 'y': 120, 'w': 36, 'h': 45},
			"age": 28.66,
			"gender": "woman",
			"dominant_emotion": "neutral",
			"emotion": {
				'sad': 37.65260875225067,
				'angry': 0.15512987738475204,
				'surprise': 0.0022171278033056296,
				'fear': 1.2489334680140018,
				'happy': 4.609785228967667,
				'disgust': 9.698561953541684e-07,
				'neutral': 56.33133053779602
			}
			"dominant_race": "white",
			"race": {
				'indian': 0.5480832420289516,
				'asian': 0.7830780930817127,
				'latino hispanic': 2.0677512511610985,
				'black': 0.06337375962175429,
				'middle eastern': 3.088453598320484,
				'white': 93.44925880432129
			}
		}

	"""

	img_paths, bulkProcess = functions.initialize_input(img_path)

	#---------------------------------

	built_models = list(models.keys())

	#---------------------------------

	#pre-trained models passed but it doesn't exist in actions
	if len(built_models) > 0:
		if 'emotion' in built_models and 'emotion' not in actions:
			actions.append('emotion')

		if 'age' in built_models and 'age' not in actions:
			actions.append('age')

		if 'gender' in built_models and 'gender' not in actions:
			actions.append('gender')

		if 'race' in built_models and 'race' not in actions:
			actions.append('race')

	#---------------------------------

	if 'emotion' in actions and 'emotion' not in built_models:
		models['emotion'] = build_model('Emotion')

	if 'age' in actions and 'age' not in built_models:
		models['age'] = build_model('Age')

	if 'gender' in actions and 'gender' not in built_models:
		models['gender'] = build_model('Gender')

	if 'race' in actions and 'race' not in built_models:
		models['race'] = build_model('Race')

	#---------------------------------

	resp_objects = []

	disable_option = (False if len(img_paths) > 1 else True) or not prog_bar

	global_pbar = tqdm(range(0,len(img_paths)), desc='Analyzing', disable = disable_option)

	for j in global_pbar:
		img_path = img_paths[j]

		resp_obj = {}

		disable_option = (False if len(actions) > 1 else True) or not prog_bar

		pbar = tqdm(range(0, len(actions)), desc='Finding actions', disable = disable_option)

		img_224 = None # Set to prevent re-detection

		region = [] # x, y, w, h of the detected face region
		region_labels = ['x', 'y', 'w', 'h']

		is_region_set = False

		#facial attribute analysis
		for index in pbar:
			action = actions[index]
			pbar.set_description("Action: %s" % (action))

			if action == 'emotion':
				emotion_labels = ['angry', 'disgust', 'fear', 'happy', 'sad', 'surprise', 'neutral']
				img, region = functions.preprocess_face(img = img_path, target_size = (48, 48), grayscale = True, hard_detection_failure = hard_detection_failure, detector_backend = detector_backend, return_region = True) #target_size = (224, 224)

				emotion_predictions = models['emotion'].predict(img)[0,:]

				sum_of_predictions = emotion_predictions.sum()

				resp_obj["emotion"] = {}

				for i in range(0, len(emotion_labels)):
					emotion_label = emotion_labels[i]
					emotion_prediction = 100 * emotion_predictions[i] / sum_of_predictions
					resp_obj["emotion"][emotion_label] = emotion_prediction

				resp_obj["dominant_emotion"] = emotion_labels[np.argmax(emotion_predictions)]

			elif action == 'age':
				if img_224 is None:
					img_224, region = functions.preprocess_face(img = img_path, target_size = (224, 224), grayscale = False, hard_detection_failure = hard_detection_failure, detector_backend = detector_backend, return_region = True)

				age_predictions = models['age'].predict(img_224)[0,:]
				apparent_age = Age.findApparentAge(age_predictions)

				resp_obj["age"] = int(apparent_age) #int cast is for the exception - object of type 'float32' is not JSON serializable

			elif action == 'gender':
				if img_224 is None:
					img_224, region = functions.preprocess_face(img = img_path, target_size = (224, 224), grayscale = False, hard_detection_failure = hard_detection_failure, detector_backend = detector_backend, return_region = True)

				gender_prediction = models['gender'].predict(img_224)[0,:]

				if np.argmax(gender_prediction) == 0:
					gender = "Woman"
				elif np.argmax(gender_prediction) == 1:
					gender = "Man"

				resp_obj["gender"] = gender

			elif action == 'race':
				if img_224 is None:
					img_224, region = functions.preprocess_face(img = img_path, target_size = (224, 224), grayscale = False, hard_detection_failure = hard_detection_failure, detector_backend = detector_backend, return_region = True) #just emotion model expects grayscale images
				race_predictions = models['race'].predict(img_224)[0,:]
				race_labels = ['asian', 'indian', 'black', 'white', 'middle eastern', 'latino hispanic']

				sum_of_predictions = race_predictions.sum()

				resp_obj["race"] = {}
				for i in range(0, len(race_labels)):
					race_label = race_labels[i]
					race_prediction = 100 * race_predictions[i] / sum_of_predictions
					resp_obj["race"][race_label] = race_prediction

				resp_obj["dominant_race"] = race_labels[np.argmax(race_predictions)]

			#-----------------------------

			if is_region_set != True:
				resp_obj["region"] = {}
				is_region_set = True
				for i, parameter in enumerate(region_labels):
					resp_obj["region"][parameter] = int(region[i]) #int cast is for the exception - object of type 'float32' is not JSON serializable

		#---------------------------------

		if bulkProcess == True:
			resp_objects.append(resp_obj)
		else:
			return resp_obj

	if bulkProcess == True:

		resp_obj = {}

		for i in range(0, len(resp_objects)):
			resp_item = resp_objects[i]
			resp_obj["instance_%d" % (i+1)] = resp_item

		return resp_obj

def get_model(model_name): # needed for the pool
	return build_model(model_name)

def enhanced_stream(db_path = '.', auto_add = False, actions = [], model_name ='VGG-Face', skip_no_face_images = True, detector_backend = 'opencv', align = False, normalization = 'base', distance_metric = 'cosine', source = "", process_only = True, number_of_processes = 1):
	"""
	This function is similar to enhanced_find function but it acts when detecting a face in a video instead of an image.
	These are the additional features :
	1) The git functionality runs at the start and upon user request, according to a GUI button or maybe a listener to a text file change event, as well that is while the code is running; this is the case where the video is running and user added some images to someone in the database or changed the name of someone or some image in the database. 
	2) If skip_no_face_images is set to False code will add a probably bad image representation of the full image when no face is detected. But when set to True continue with no detected face, user will be informed and code will skip this particular image and will make the representations of other useful images if existing.
	3) When process_only is True we create or override a pkl file. [old: It's like a csv file generated showing in every timestamp all the persons who were there. The names of the colomns of the csv file are the names of the persons. A 50 minutes stream of 30 fps will be recorded in a .csv file which can be opened in Excel without problems. If we needed more stream time we could simply make another csv file when the first one reaches e.g. 1 million records.] 
	This pkl file is useful to replay smoothly (play_with_annotations function) with some limited info like name and main emotion, as well to make statistics. E.g. we can trace a particular person when he appeared and when he disappeared. It's useful for recognition as well for emotion analysis, e.g. we can track when a particular person was happy with which persons, or e.g. what is the average of each emotion for a particular person or the average of emotions for all persons. Having e.g. 6 persons detected, there can be an analysis to each and every one of them [old: we would have a table where the colomns are the name of those people]
	This pkl file is a list of dictionaries, so it can be made a json file and can be accessed using specific database tools like Mongodb or couch or whatever.
	N.B: Some statistics cannot be exaggerated with their result, since we only get (for now at least) from 1 camera and it won't show a target person all the time, besides, this emotion recognition lacks continuity, e.g. I suspect that it will detect happiness while the person is actually feeling bad, but still it holds some valuable info, like e.g. how long can a person maintain an "apparent" feeling.
	4) If actions = [] then this function only applies face recognition to a stream. We check each and every frame for recognition only. If actions = ['emotion'] then preferrably be it an offline stream since it would take a lot of time. This will take each frame as being worthy of analyzing; no freezing, no time_threshold, no frame_threshold. Anyway we have the option, probably in another function, of being recorded so that at replay time one can check the emotions continuously. 
	5) Naming convention to refer to a certain person when detected in the stream can be like : "ahmad yassine_1.jpg" or "ahmad yassine 1.jpg". Statistics is usually made about each person on its own, but this relies on the user following a good convention, i.e. naming "ahmed yassine1.jpg" and "ahmad yassine2.jpg" they will be considered different persons. 
	6) New detected persons not found in database representations will be added as images to the database under a random name and as an embedding which will be recognized during the stream and which will be added to the representation pkl as well at the end of stream. User has to rename them and probably write the old and new names then he must click on the update button. This is made by the auto-add --TODO
	7) source can be an YouTube video. --TODO
	8) Multiprocessing so that realtime may be feasible with process_only set to False or whether set to True. User may enter how many processes he wishes. Multiprocessing can be effective in getting representations of images in the database, as well as in frames processing in case process_only is True, and in processing 1 frame at a time in case process_only is False, thus in realtime.--TODO Allowing pressing 1 to abort long running processes an terminate the program. --TODO Even later we can use distribute processes on multiple computers. --TODO
	9) make an expiration license for the software so that it can be distributable to the user. --TODO
	

	An example : DeepFace.enhanced_stream(db_path = '/home/youssef/database2', skip_no_face_images = True, source = '/home/youssef/videos/hi.mp4')
	model_name : "VGG-Face", "Facenet", "OpenFace", "DeepFace"
	distance_metric : "cosine", "euclidean", "euclidean_l2"
	detector_backend : "retinaface", "mtcnn", "opencv", "ssd" or "dlib"
	normalization : "base", "raw", "Facenet", "Facenet2018", "VGGFace", "VGGFace2", "ArcFace" (it's there in functions.normalize_input)
 
	#TODO
	launch it from the terminal or the cmd prompt --DONE

	work on calling play_with_annotations right after calling enhanced_stream will automatically specify the generated pkl file.--POSTPONED

	Know why some images aren't recognized (although they are fine, image of Anis) in the database of images to get representations from ?

	Test how small an image can be in order to be recognized. Small enough. --DONE
	
	When a face is near the edge, it doesn't probably detect it, how to fix it ?

	test fidelity of the recognition in case a person is totally turning his head right or left

	test later some file formats in the database like .bmp, .png, .tiff... Will they work and be reliable ?

	test if 2 or more persons appear in the video, will it perform right ?

	add the timing to each frame of frames_info in the pkl file

	use auto_add to get a panoramic set of profile images of a particular person if that helps in recognition.
	
	Adding a pause, backward, and forward functionality to the user while showing frame number. Also allowing the user to navigate in time. And showing the elapsed time.

	The name of the frames_info_....pkl file has to contain the model and distance metric used.

	Do the ensemble model if it provides better accurracy.

	I guess Arabic names don't work ?

	It would be interesting that in the process of finding representations of the images in the database to select a rectangle around the image. This way the user can make sure that the correct face is truly selected. On another hand, user should not be bothered with this; it's the R&D business.
	
	Since it's probably a good idea to expand the database of images, if the app recognized a face in frames e.g. 126 and 128, and only detected (not recognized) a face in frame 127 in the same box placement, then we may add the image and assign it the name of the recognized person.

	One good feature is to allow the user while replaying to tag a face and add it to the database, and perhaps change the tag of a face. But auto-add somehow plays the same role when supervised by the supervisor (user).

	Make a script so that when double clicked it prepends all the images inside with a name given by the user.
	"""


	functions1.print_license()
	print("Version 1.0\n\n\n")

	#inform about the time
	start_time = time.ctime()
	print(start_time)

	#check passed db folder exists
	db_path = functions1.validate_win_path(db_path)

	if os.path.isdir(db_path) == False:
		print("Provided database is not a valid directory/folder.\nStopping execution.")
		return None	
	#------------------------

	#get recognizable source
	source = functions1.validate_win_path(source)
	#------------------------

	print("Detector backend is ", detector_backend, ". Building face detector model...")
	face_detector = FaceDetector.build_model(detector_backend)

	#------------------------

	print("Finding distance threshold...")
	#tuned thresholds for model and metric pair
	threshold = dst.findThreshold(model_name, distance_metric)
	print("threshold with distance_metric", distance_metric, "is", threshold) #TODO debugging

	print("Building", model_name, "model...")
	
	model = build_model(model_name)
	print(model_name," is built")

	input_shape = functions.find_input_shape(model)
	input_shape_x = input_shape[0]; input_shape_y = input_shape[1]

	#------------------------

	text_color = (255,255,255)

	employees = []
	img_type = (".jpg", ".jpeg", ".bmp", ".png")

	#find embeddings for employee list

	tic = time.time()

	#-----------------------

	file_name = "representations_%s.pkl" % (model_name)
	file_name = file_name.replace("-", "_").lower()
	pkl_path = db_path+"/"+file_name

	embeddings = []

	if path.exists(pkl_path):

		print("Found existing embedding file", pkl_path)
		try:
			with open(db_path+'/'+file_name, 'rb') as f:
				embeddings = pickle.load(f)
				print("There are ", len(embeddings)," embeddings found in ", file_name)
		except Exception as err:
			print(err)
		
		#check git and update embeddings
		embeddings = functions1.check_git_and_update_embeddings(embeddings, model_name, quick_represent, pkl_path, db_path = db_path, target_size = (input_shape_y, input_shape_x), hard_detection_failure = skip_no_face_images, detector_backend = detector_backend, normalization = normalization, img_type = img_type, number_of_processes = number_of_processes)

		#Just about leaving the program we shall write changes to pickle file, since user or program may make changes to embeddings.

	else: #representation .pkl file to be later created from scratch
		print(".pkl file wasn't found, so scrolling whole database for all images.")
		employees = functions1.get_employees(db_path, img_type, path_type = "relative")
		
		if len(employees) == 0:
			#embeddings = [] #redundant
			images_undetected_faces_list = []
			auto_add = True
			print("No images were found in the database") 
			if auto_add:
				print("images will be automatically added from video source")
			else:
				print("make sure that auto_add is set to True. Aborting...")
				return
		else:
			try:
				embeddings, images_undetected_faces_list = functions1.get_embeddings(employees, model_name, quick_represent, db_path = db_path, target_size = (input_shape_y, input_shape_x), hard_detection_failure = skip_no_face_images, detector_backend = detector_backend, normalization = normalization, number_of_processes = number_of_processes)
			except Exception as err:
				print("caught exception when trying to find embeddings to images in db_path where no pkl file exists\n", err)
				return 
		#checking for changes in git is always needed then we commit
		to_be_added_images_list, to_be_removed_images_list = functions1.check_change(db_path = db_path, img_type = img_type)
		functions1.commit_changes(to_be_removed_images_list, to_be_added_images_list, images_undetected_faces_list, db_path = db_path)

	toc = time.time()
	print("Done processing images representations in " + str(toc - tic) + " seconds")

	functions1.save_pkl(content = embeddings, exact_path = pkl_path)

	if source == "":
		return None

	df = pd.DataFrame(embeddings, columns = ['employee', 'embedding'])
	df['distance_metric'] = distance_metric
	print(df)
	
	emotion_model = None
	if("emotion" in actions):
		print("Building emotion model...")
		emotion_model = build_model('Emotion')
		print("Emotion model loaded")

	cap = cv2.VideoCapture(source) #webcam or path to video file

	frames_info_name = ""

	if(not process_only):	
		#here we think realtime		
		
		global frame_index	#in multithreading accessing this variable by another thread is reliable I guess https://stackoverflow.com/questions/53780267/an-equivalent-to-java-volatile-in-python
		frame_index = 0

		print("Attempting to play the stream with annotations...\nPress q to abort")

		ret = True
		while(not (cv2.waitKey(1) & 0xFF == ord('q')) and ret == True): 
			ret, img = cap.read() 

			if img is None:
				break
			
			frame_index += 1
			print(frame_index)

			#whether detection alone or with emotion, we need to execute all that in a separate thread in order not to interrupt reading the next frame(s) and showing them to user i.e. preserving user experience.

			threading.Thread(target=functions1.process_frame, args = (frame_index, img, face_detector, df, threshold, model), kwargs={ "detector_backend": detector_backend, "align": align, "target_size": (input_shape_y, input_shape_x), "process_only": process_only, "auto_add": auto_add, "db_path": db_path, "emotion_model": emotion_model, "normalization": normalization, "img_type": img_type}).start() #https://www.geeksforgeeks.org/multithreading-python-set-1/ and https://www.geeksforgeeks.org/multithreading-in-python-set-2-synchronization/
			cv2.imshow('img',img)
		
	else: #process_only
		tic = time.time()
		frames_info = functions1.process_frames(cap, face_detector, df, threshold, model, detector_backend = detector_backend, align = align, target_size = (input_shape_y, input_shape_x), auto_add = auto_add, db_path = db_path, emotion_model = emotion_model, normalization = normalization, img_type = img_type)
		toc = time.time()
		print("Processing frames done with " + str(toc - tic) + " seconds")
		
		#save the frames info in a pkl file
		if frames_info is not None:
			print("Saving frames info of the stream to a pkl file...")
			frames_info_name = source
			frames_info_name = frames_info_name.split("/")[-1].replace(".mp4", "")
			#frames_info_name = frames_info_name.split("/")[-1].replace(".webm", "") #cv2 is not really well with webm ?
			frames_info_name = "frames_info_%s.pkl" % (frames_info_name)
			frames_info_name = frames_info_name.replace("-", "_").lower()
			#frames_info_name = "/".join(source.split("/")[:-1])+"/"+frames_info_name
			frames_info_name = db_path+"/"+frames_info_name
			functions1.save_pkl(content = frames_info, exact_path = frames_info_name)
			print("Done saving")
		else:
			print("no face was detected in this stream")
	
			
	#kill open cv things
	cap.release()
	cv2.destroyAllWindows()

	print("Done with enhanced_stream")

	return frames_info_name

def play_with_annotations(source, frames_info_path, window_name = "img", speed = "normal", fps = 30, im_size = ()):
	"""
	frames_info_path is a path to a pkl file holding all annotations information inferred from a video source.
	source is the path to the mp4 file.
	speed can be either "fast", "slow", or "normal".
	fps is usually left to 30. If we wanted to slow the video even more we can lower the fps to say 20 or less, and vice versa to speed it up a lot.
	e.g. play_with_annotations("/home/youssef/database2/hi.mp4", "/home/youssef/database2/frames_info_hi.pkl")
	im_size is e.g. (960, 540)

	
	TODO
	Needs to play the sound as well.
	show time of the video
	allow a pause, rewind and forward buttons
	"""
	functions1.print_license()
	print("Version 1.0\n\n\n")
	
	try:
		with open(frames_info_path, 'rb') as f:
			frames_info = pickle.load(f)
	except Exception as err:
		if(err.args[0] == 2): print("Error while attempting to open file of path \"{}\". Make sure it exists.".format(frames_info_path))
		return

	if(speed == "fast"):
		wait_key_time = int(np.round(1/fps * 1000 / 2))
	elif(speed == "slow"):
		wait_key_time = int(np.round(1/fps * 1000 * 2))
	else:
		wait_key_time = int(np.round(1/fps * 1000))

	print("Playing the stream with annotations...\nPress q to abort")
	frame_info_index = 0
	frame_index = 0

	cap = cv2.VideoCapture(source) #webcam or path to video file
	
	#if(frames_info is not None):
	ret = True
	while(not (cv2.waitKey(wait_key_time) & 0xFF == ord('q')) and ret == True):
		ret, img = cap.read() 

		if img is None:
			break
		
		# Extracting frames info...
		if(frame_info_index < len(frames_info) and frames_info[frame_info_index]["frame_index"] == frame_index):
			#the frames_info item of index frame_info_index has some face(s) inside
			for face_info in frames_info[frame_info_index]["detected_faces"]:
				functions1.face_inform(face_info, img)

			frame_info_index += 1
		
		frame_index += 1
		#print(frame_index, frame_info_index, frames_info[frame_info_index]["frame_index"])
		if(im_size != ()):
			img = cv2.resize(img, im_size) 
		cv2.imshow(window_name,img)

	cap.release()
	cv2.destroyAllWindows()
	print("Done with play_with_annotations")

def enhanced_find(img_path, db_path, auto_add = False, model_name ='VGG-Face', distance_metric = 'cosine', model = None, hard_detection_failure = True, detector_backend = 'opencv', align = True, prog_bar = True, normalization = 'base'):
	"""
	This function detects a face in an input image, and searches for that person in a database.
	More than 1 person is also handled. 
	In the search, it relies on a .pkl file and creates one in case it wasn't created already.
	The following is what differs from the old find function :
		1) It supports a git api, so as when this function is called, it checks the git status to know whether to remove, rename, or add something to the pkl file because the user had made a similar change to the database. 
		2) It supports automatic addition of new persons detected while running according the value of auto_add. Having e.g. 6 persons detected, if some of those weren't recognized, a new profile will be created in the database in a folder called "new", and it will contain a folder for each new person with some random name. 
	"""

	return None

def find(img_path, db_path, model_name ='VGG-Face', distance_metric = 'cosine', model = None, hard_detection_failure = True, detector_backend = 'opencv', align = True, prog_bar = True, normalization = 'base'):

	"""
	This function applies verification several times and find an identity in a database

	Parameters:
		img_path: exact image path, numpy array or based64 encoded image. If you are going to find several identities, then you should pass img_path as array instead of calling find function in a for loop. e.g. img_path = ["img1.jpg", "img2.jpg"]

		db_path (string): You should store some .jpg files in a folder and pass the exact folder path to this.

		model_name (string): VGG-Face, Facenet, OpenFace, DeepFace, DeepID, Dlib or Ensemble

		distance_metric (string): cosine, euclidean, euclidean_l2

		model: built deepface model. A face recognition models are built in every call of find function. You can pass pre-built models to speed the function up.

			model = DeepFace.build_model('VGG-Face')

		hard_detection_failure (boolean): The function throws exception if a face could not be detected. Set this to True if you don't want to get exception. This might be convenient for low resolution images.

		detector_backend (string): set face detector backend as retinaface, mtcnn, opencv, ssd or dlib

		prog_bar (boolean): enable/disable a progress bar. If set to True it disables.

	Returns:
		This function returns pandas data frame. If a list of images is passed to img_path, then it will return list of pandas data frame.
	"""

	tic = time.time()

	img_paths, bulkProcess = functions.initialize_input(img_path) #bulkprocess means that img_paths is more than 1 image

	#-------------------------------

	if os.path.isdir(db_path) == True:

		models = functions1.get_models(build_model, model_name, model)

		#---------------------------------------

		model_names, metric_names = functions1.get_model_and_metric_names(model_name, distance_metric)

		#---------------------------------------

		file_name = "representations_%s.pkl" % (model_name)
		file_name = file_name.replace("-", "_").lower()

		if path.exists(db_path+"/"+file_name):

			print("WARNING: Representations for images in ",db_path," folder were previously stored in ", file_name, ". If you added new instances after this file creation, then please delete this file and call find function again. It will create it again.")

			f = open(db_path+'/'+file_name, 'rb')
			representations = pickle.load(f)

			print("There are ", len(representations)," representations found in ",file_name)

		else: #create representation.pkl from scratch
			representations = functions1.create_representation_file(file_name, model_names, models, represent, db_path = db_path, model_name = model_name, hard_detection_failure = hard_detection_failure, detector_backend = detector_backend, align = align, normalization = normalization, prog_bar = prog_bar, img_type = (".jpg", ".png"))
		#----------------------------
		#now, we got representations for facial database

		df = functions1.get_df(representations, model_names, model_name = model_name)

		resp_obj = functions1.get_find_result(df, represent, img_paths, model_names, models, metric_names, model_name = model_name, hard_detection_failure = hard_detection_failure, detector_backend = detector_backend, align = align, normalization = normalization, prog_bar = prog_bar)

		toc = time.time()

		print("find function lasts ",toc-tic," seconds")

		if len(resp_obj) == 1:
			return resp_obj[0]

		return resp_obj

	else:
		raise ValueError("Passed db_path does not exist!")

	return None

def represent(img_path, model_name = 'VGG-Face', model = None, hard_detection_failure = True, detector_backend = 'opencv', align = True, normalization = 'base'):

	"""
	This function represents facial images as vectors.

	Parameters:
		img_path: exact image path, numpy array or based64 encoded images could be passed.

		model_name (string): VGG-Face, Facenet, OpenFace, DeepFace, DeepID, Dlib, ArcFace.

		model: Built deepface model. A face recognition model is built every call of verify function. You can pass pre-built face recognition model optionally if you will call verify function several times. Consider to pass model if you are going to call represent function in a for loop.

			model = DeepFace.build_model('VGG-Face')

		hard_detection_failure (boolean): If any face could not be detected in an image, then verify function will return exception. Set this to False not to have this exception. This might be convenient for low resolution images.

		detector_backend (string): set face detector backend as retinaface, mtcnn, opencv, ssd or dlib

		normalization (string): normalize the input image before feeding to model

	Returns:
		Represent function returns a multidimensional vector. The number of dimensions is changing based on the reference model. E.g. FaceNet returns 128 dimensional vector; VGG-Face returns 2622 dimensional vector.
	"""

	if model is None:
		model = build_model(model_name)

	#---------------------------------

	#decide input shape
	input_shape_x, input_shape_y = functions.find_input_shape(model)

	#detect and align
	img = functions.preprocess_face(img = img_path
		, target_size=(input_shape_y, input_shape_x)
		, hard_detection_failure = hard_detection_failure
		, detector_backend = detector_backend
		, align = align)

	#---------------------------------
	#custom normalization

	img = functions.normalize_input(img = img, normalization = normalization)

	#---------------------------------

	#represent
	embedding = model.predict(img)[0].tolist()
	#in realtime.analysis function it's 
	#img_representation = model.predict(img)[0,:]

	return embedding

def quick_represent(img_path, model, target_size=(224, 224), hard_detection_failure = True, detector_backend = 'opencv', normalization = 'base'):

	"""
	This function is the same as "represent" but assuming model is built and input_shape is passed as argument. Comments within are important though.
	"""

	#detect and align
	#preprocess_face returns single face. this is expected for source images in db.
	img = functions.preprocess_face(img = img_path
		, target_size = (target_size[0], target_size[1])
		, hard_detection_failure = hard_detection_failure
		, detector_backend = detector_backend
		) #align is omitted so set to default in analysis function in realtime.py

	#custom normalization. It wasn't originally part of analysis function in realtime.py
	img = functions.normalize_input(img = img, normalization = normalization)

	#represent
	embedding = model.predict(img)[0,:]

	return embedding

def stream(db_path = '', model_name ='VGG-Face', detector_backend = 'opencv', distance_metric = 'cosine', enable_face_analysis = True, source = 0, time_threshold = 5, frame_threshold = 5):

	"""
	This function applies real time face recognition and facial attribute analysis

	Parameters:
		db_path (string): facial database path. You should store some .jpg files in this folder.

		model_name (string): VGG-Face, Facenet, OpenFace, DeepFace, DeepID, Dlib or Ensemble

		detector_backend (string): opencv, ssd, mtcnn, dlib, retinaface

		distance_metric (string): cosine, euclidean, euclidean_l2

		enable_facial_analysis (boolean): Set this to False to just run face recognition

		source: Set this to 0 for access web cam. Otherwise, pass exact video path.

		time_threshold (int): how many second analyzed image will be displayed

		frame_threshold (int): how many frames required to focus on face

	"""

	if time_threshold < 1:
		raise ValueError("time_threshold must be greater than the value 1 but you passed "+str(time_threshold))

	if frame_threshold < 1:
		raise ValueError("frame_threshold must be greater than the value 1 but you passed "+str(frame_threshold))

	realtime.analysis(db_path, model_name, detector_backend, distance_metric, enable_face_analysis
						, source = source, time_threshold = time_threshold, frame_threshold = frame_threshold)


def detectFace(img_path, detector_backend = 'opencv', hard_detection_failure = True, align = True):

	"""
	This function applies pre-processing stages of a face recognition pipeline including detection and alignment

	Parameters:
		img_path: exact image path, numpy array or base64 encoded image

		detector_backend (string): face detection backends are retinaface, mtcnn, opencv, ssd or dlib

	Returns:
		deteced and aligned face in numpy format
	"""

	img = functions.preprocess_face(img = img_path, detector_backend = detector_backend
		, hard_detection_failure = hard_detection_failure, align = align)[0] #preprocess_face returns (1, 224, 224, 3)
	return img[:, :, ::-1] #bgr to rgb

#---------------------------
#main

functions.initializeFolder()
