import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
import re

import imageio

import numpy as np
import pandas as pd
import cv2

from git import Repo
import time
from multiprocessing import Pool

from tqdm import tqdm
import pickle

from deepface import DeepFace
from deepface.basemodels import Boosting
from deepface.commons import distance as dst, functions
from deepface.detectors import FaceDetector

from functools import partial

def get_models(build_model, model_name ='VGG-Face', model = None):
	if model == None:

		if model_name == 'Ensemble':
			print("Ensemble learning enabled")
			models = Boosting.loadModel()

		else: #model is not ensemble
			model = build_model(model_name)
			models = {}
			models[model_name] = model

	else: #model is not None
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

def validate_win_path(path):
	if os.name == 'nt': #running on windows os
		path = path.replace("\\","/")
		#path = repr(path) #it probably makes trouble with e.g. \a \b \1 etc...
		#path = path.replace("\\","/")
		#path = path.strip("'")
	return path


def check_change(db_path=".", img_type = (".jpeg", ".jpg", ".png", ".bmp")): #this whole function can be called in its own thread
	"""
	The goal of this function check_change is to trace the files without having to read the whole database for any tiny change made by the user like adding an image or deleting another.
	"""
	#User might add an image to his database, might rename, move, delete, etc...
	try:
		repo = Repo(db_path) #this throws an error if it's not git-initialized already
	except Exception:
		repo = Repo.init(db_path, bare=False) #initializing
	
	#check for untracked files
	to_be_added_images_list = []
	for f in repo.untracked_files: 
		if(f.lower().endswith(img_type)):
			f = validate_win_path(f)
			to_be_added_images_list.append(f) #images aren't tracked yet
	#check for unstaged files (which are already tracked)
	to_be_removed_images_list = []
	for x in repo.index.diff(None):
		if(x.b_path.endswith(img_type)):#added in case someone messes with the repo, like adds something which isn't an image to the git staging area i.e. tracking it.
			f = validate_win_path(x.b_path)
			if(x.change_type == 'M'): #not sure how an image file could be modified, anyway
				to_be_added_images_list.append(f)
				to_be_removed_images_list.append(f) #this is in case 2 committed files interchanged their names together, so in a best effort manner they must both be removed, then they both must added
			elif(x.change_type == 'D'):
				to_be_removed_images_list.append(f)
	#Although user might rename an image and it might be worthy to trace such thing, but it is not a direct process in git so postponed. A renamed file is deleted and made new when it comes to git and us.
	
	return to_be_added_images_list, to_be_removed_images_list


def commit_changes(to_be_removed_images_list, to_be_added_images_list, images_undetected_faces_list = [], db_path="."):
	#User might add an image to his database, might rename, move, delete, etc...

	#first remove from added_images_list the images in which no face was detected; we want to keep them untracked
	for im_path_no_face in images_undetected_faces_list: #im_path a relative path
		for i in range(len(to_be_added_images_list)):
			if to_be_added_images_list[i] == im_path_no_face:
				to_be_added_images_list.pop(i)
				break
	
	try:
		repo = Repo(db_path) #this throws an error if it's not git-initialized already
	except Exception:
		repo = Repo.init(db_path, bare=False) #initializing
		
	#Now staging images to make them ready for committing and to clear the staging area for future changes
	commit = False
	#below, removing is made on purpose before adding, for the sake of modified files which I made as belonging to both. Because a modified file must be committed as being added, not removed.
	if(len(to_be_removed_images_list) > 0):
		commit = True
		#removing images
		try:
			repo.index.remove(to_be_removed_images_list) #it removes the file after being added or even committed, then it returns it back to being untracked if it ever existed again, I guess
		except Exception as err:
			print("Git error while adding untracked file(s)", err)
	if(len(to_be_added_images_list) > 0):
		#Attempting to add the files to the staging (committing) area in order to finally commit
		commit = True
		try:
			repo.index.add(to_be_added_images_list) #images are now tracked and are in the committing area.
		except Exception as err:
			print("Git error while adding untracked file(s)", err)	

	#committing 
	if(commit):
		repo.index.commit("commit at " + time.ctime().replace(" ", "_"))
	
	pass


def update_embeddings(embeddings, removed_images_list, added_images_list, model_name, represent, db_path = ".", target_size = (224, 224), hard_detection_failure = False, detector_backend = 'opencv', normalization = 'base', number_of_processes = 1):
	#if(len(removed_images_list)!=0):
	for removed_image in removed_images_list: # processing removed images are made on purpose before added images, for the sake of modified files.
		for i in range(len(embeddings)):
			if(embeddings[i][0] == removed_image):
				embeddings.pop(i)
				break # even if user has named 2 images in separate folders the same name, it still works since we deal with relative paths
	
	images_undetected_faces_list = []

	if(len(added_images_list)!=0):#not needed but anyway.	
		added_embeddings, images_undetected_faces_list = get_embeddings(added_images_list, model_name, represent, db_path = db_path, target_size = (target_size[0], target_size[1]), hard_detection_failure = hard_detection_failure, detector_backend = detector_backend, normalization = normalization, number_of_processes = number_of_processes)
		embeddings.extend(added_embeddings)
		#embeddings.sort() #list sort is smart; it sorts first according to first column (which we care only about) then according to second column, but in practice it's not working and saying "The truth value of an array with more than one element is ambiguous. Use a.any() or a.all()" but it's not a big deal anyway I guess.
	return embeddings, images_undetected_faces_list


def check_git_and_update_embeddings(embeddings, model_name, represent, pkl_path, db_path = ".", target_size = (224, 224), hard_detection_failure = False, detector_backend = 'opencv', normalization = 'base', img_type = (".jpg", ".jpeg", ".bmp", ".png"), number_of_processes = 1):
	"""
	This is the case where the representations pkl file still exists. But implicitly, the .git folder also should exist.
	Weird result happens when the .git folder is deleted (added images will be all the images again, and the embeddings will be doubled each which is reflected on the df data frame as well as the newly to be saved pkl representations file.)
	"""
	#check git for possible user-made changes then commit
	added_images_list, removed_images_list = check_change(db_path = db_path, img_type = img_type)
	print("Supposed updates in database after last save to", pkl_path, "are :")
	print("\tAdded images list :", added_images_list)
	print("\tRemoved images list :", removed_images_list)

	images_undetected_faces_list = []
	try:
		embeddings, images_undetected_faces_list = update_embeddings(embeddings, removed_images_list, added_images_list, model_name, represent, db_path = db_path, target_size = (target_size[0], target_size[1]), hard_detection_failure = hard_detection_failure, detector_backend = detector_backend, normalization = normalization, number_of_processes = number_of_processes)
	except Exception as err:
		print("check_git_and_update_embeddings", err)
	
	#now committing
	commit_changes(removed_images_list, added_images_list, images_undetected_faces_list, db_path = db_path)

	return embeddings

def get_employees(db_path = ".", img_type = (".jpg", ".jpeg", ".bmp", ".png"), path_type = "exact"):
	"""
    if path_type is "relative" the function returns list of relative paths of images i.e. path starting from the db_path excluding it. (any other keyword than "exact" will be considered as "relative")
    elif path_type is "exact" the function returns list of exact paths i.e. path starting from the db_path including.
	"""
	employees = []
	for r, d, f in os.walk(db_path): # r=root, d=directories, f = files
		r = validate_win_path(r)
		if(r.split("/")[-1] == ".git"):
			print("bypassing git file") #debugging TODO on windows it's fine. On Ubuntu ?
			continue
		for file in f:
			for t in img_type:
				if (file.lower().endswith(t)):
					path = r + "/" + file # exact path
                    #exact_path = os.path.join(r, file)
					if(path_type != "exact"): # relative path
						path = path[len(db_path) + 1:] # +1 for the '/' after db_path and before relative path
					employees.append(path)
					break
	print("Total number of images is {}".format(len(employees)))
	return employees
 
def get_embeddings_process(employees, employees_index_list, model_name, represent, index, db_path = ".", target_size = (224, 224), hard_detection_failure = False, detector_backend = 'opencv', normalization = 'base'):
	"""
	For technical reasons which have to do with pool in multiprocessing I had to put this function in the top level; it cannot be nested in get_embeddings function.
	And index is the last parameter. This has to do with tqdm with pool.
	And model_name is passed instead of model because pool couldn't simply recognize DeepFace.model, so I had to read get the model every time

	This function will return a dictionary e.g. either
	{"embedding": [employee, representation]}
	or
	{"undetected_faces_image":index}
	"""
	process_id = os.getpid()
	print("\nWorker process id : {}".format(process_id))
	
	embeddings = []
	images_undetected_faces_list = []
	
	if index + 1 < len(employees_index_list): #this is a valid process to work with, i.e. there are some employee(s), i.e. employees_index_list[index + 1] won't raise an exception

		pbar = tqdm(range(employees_index_list[index + 1] - employees_index_list[index]), position= 0)

		for i in pbar:
			employee = employees[i + employees_index_list[index]] #it's a copy byval, not references
			pbar.set_description("Process {} : Finding embedding for {}".format(process_id, employee.split("/")[-1])) #according to usage employee can be a full exact path or just a path after (without) the db_path. Both cases, .split("/")[-1] works fine. employee may even not contain '/' and it works fine.
			
			try: #this try-except is useful in case hard_detection_failure was set to True and not face was detected
				img_representation = represent(db_path +'/'+ employee, model = DeepFace.get_model(model_name), target_size = (target_size[0], target_size[1]), hard_detection_failure = hard_detection_failure, detector_backend = detector_backend, normalization = normalization)

				embeddings.append([employee, img_representation])

			except Exception as err:
				#print(err) #usual message is as follows : Face could not be detected. Please confirm that the picture is a face photo or consider to set hard_detection_failure param to False.
				#print("Could not detect a face in this image : {}".format(employee))
				images_undetected_faces_list.append(employee)

	return [embeddings, images_undetected_faces_list] #return value is made a list because of pool

def get_share(employees_len, number_of_processes):
	"""
	divides the employees_len onto number_of_processes in an equal manner as much as possible. But returns another list by incrementally cumulatively adding the equal share list.
	e.g. employees_len = 18 and number_of_processes = 4, so :
	process1 will be assigned 5 emplyees
	process2 will be assigned 5 emplyees, 
	process3 will be assigned 4 emplyees, 
	process4 will be assigned 4 emplyees,
	so the return value will be [5,5,4,4] . Now they are equally divided.
	The incremented values would be [5,10,14,18]
	and the return value would be [0,5,10,14,18]
	"""
	q = int(employees_len/number_of_processes)
	share_list = [q]*number_of_processes
	to_add_1 = employees_len - sum(share_list)
	share_list_temp = [share_list[i] + 1 for i in range(to_add_1)]
	share_list_temp.extend([share_list[i] for i in range(len(share_list) - to_add_1)])
	while share_list_temp.count(0): share_list_temp.remove(0) # in case value was e.g. [1,1,0,0] so it becomes [1,1]
	n = 0; share_list_ = [0]
	for n_i in share_list_temp:
		n = n + n_i
		share_list_.append(n)
	return share_list_

def minimize_number_of_processes(employees_len, number_of_processes):
	"""
	Assuming that each process should at least hold 150 employees, if the number_of_processes was high then it must be lowered as such. We're not interested now in increasing the number of processes if it deserves to be higher, because this might be intended by the user, e.g. his hardware.
	"""
	while employees_len / number_of_processes < 150 and number_of_processes > 1:
		number_of_processes -= 1

	return number_of_processes

def get_embeddings(employees, model_name, represent, db_path = ".", target_size = (224, 224), hard_detection_failure = False, detector_backend = 'opencv', normalization = 'base', number_of_processes = 1):
	"""
	This is almost like the represent function in DeepFace module, but more fitting to the use case.
	The return value is enough to tell the story.

	This task will be subdivided into as many processes desired by user
	"""
	#if(len(employees) == 0):
	#	return [], []

	if(int(number_of_processes) != number_of_processes or number_of_processes <= 0):
		print("number_of_processes entered by the user is not suitable. Assuming 1 as number of processes...")
		number_of_processes = 1

	embeddings = []
	images_undetected_faces_list = []

	print("Main process of id {}".format(os.getpid()))

	employees.sort()
	employees_len = len(employees)

	number_of_processes = minimize_number_of_processes(employees_len, number_of_processes)

	if number_of_processes == 1 :
		first_employee_index = 0
		result_l = get_embeddings_process(employees, [first_employee_index, employees_len], model_name, represent, first_employee_index, db_path = db_path, target_size = target_size, hard_detection_failure = hard_detection_failure, detector_backend = detector_backend, normalization = normalization)
		result_l = [result_l]
	else:
		share_list = get_share(employees_len, number_of_processes)
		with Pool(number_of_processes) as pool: #if it were not that Pool() without specifying number_of_processes as argument is not working fine, I would had been interested in using it as so
			func = partial(get_embeddings_process, employees, share_list, model_name, represent, db_path = db_path, target_size = target_size, hard_detection_failure = hard_detection_failure, detector_backend = detector_backend, normalization = normalization)
			p = pool.map(func, range(len(share_list)))
			result_l = list(p)

	for d in result_l:
		embeddings.extend(d[0])
		images_undetected_faces_list.extend(d[1])

	#print images with no detected face
	images_undetected_faces_list_len = len(images_undetected_faces_list)
	if images_undetected_faces_list_len > 0:
		if images_undetected_faces_list_len == 1:
			print("could not detect a face for {}".format(images_undetected_faces_list[0]))
		elif images_undetected_faces_list_len > 1:
			print("could not detect a face for the following images :")
			for img_path in images_undetected_faces_list:
				print(img_path)

	return embeddings, images_undetected_faces_list

def save_pkl(content = [], exact_path = "representations.pkl"):
	print("Storing in ", exact_path, " file")
	f = open(exact_path, "wb") #this makes a new file or completely overrides an existing one
	pickle.dump(content, f)
	f.close()

def process_frames(cap, face_detector, embeddings_df, threshold, model, detector_backend = 'opencv', align = False, target_size = (224, 224), auto_add = False, db_path = ".", emotion_model = None, normalization = "base", img_type = (".jpg", ".jpeg", ".bmp", ".png")):
	"""
	The output of this function is something like that :
	[
		{
			frame_index: 0,
			detected_faces: [
				face_info1,
				face_info2,
				...
			]
		},
		{
			frame_index: 8, #it doesn't have to be that frame_index is successive since in-between-frames might not have any faces 
			detected_faces: [
				face_info1,
				face_info2,
				...
			]
		},
		...
	]
	"""
	frames_info = []
	frame_index = 0
	
	video_frames_length = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
	if(video_frames_length != -1): #it's -1 if source was 0 (built-in camera)
		pbar = tqdm(range(0, video_frames_length), position= 0)
	
	ret = True
	while(not (cv2.waitKey(1) & 0xFF == ord('q')) and ret == True):	
		ret, img = cap.read() 
		
		if img is None:
			break
		
		pbar.set_description("Processing frame %s " % frame_index)
		
		frame_info = process_frame(frame_index, img, face_detector, embeddings_df, threshold, model, detector_backend = detector_backend, align = align, target_size = (target_size[0], target_size[1]), process_only = True, auto_add = auto_add, db_path = db_path, emotion_model = emotion_model, normalization = normalization, img_type = img_type)
		if(frame_info is not None):
			frames_info.append(frame_info)
		
		frame_index += 1
		pbar.update()
	
	pbar.set_description("Done frames processing.")

	return frames_info

def process_frame(frame_index, img, face_detector, embeddings_df, threshold, model, detector_backend = 'opencv', align = False, target_size = (224, 224), process_only = True, auto_add = False, db_path = ".", emotion_model = None, normalization = 'base', img_type = (".jpg", ".jpeg", ".bmp", ".png")):

	face_detected = False
	
	resolution = img.shape 

	faces = FaceDetector.detect_faces(face_detector, detector_backend, img, align = align)
	
	detected_faces = []
	frame_info = {"frame_index": frame_index}
	for face, (x, y, w, h) in faces:
		#if w > 130: #discard small detected faces				

		face_info = process_face(face, (x, y, w, h), resolution, embeddings_df, threshold, model, emotion_model = emotion_model, detector_backend = detector_backend, target_size = (target_size[0], target_size[1]), normalization = normalization, img_type = img_type, auto_add = auto_add, db_path = db_path)

		if(not process_only): #show the rectangles and texts without returning them.
			if(DeepFace.frame_index == frame_index): #realtime condition which is almost impossible to happen. This won't return anything
				face_inform(face_info, img)
		else:
			#get the rectangles and texts then return them. We won't show them now.
			detected_faces.append(face_info)
			
	if(process_only): 
		if(detected_faces == []):
			return None
		else:			
			frame_info["detected_faces"] = detected_faces
			return frame_info

def process_face(face, pos_dim, resolution, df, threshold, model, emotion_model = None, detector_backend = 'opencv', target_size = (224, 224), normalization = 'base', img_type = (".jpg", ".jpeg", ".bmp", ".png"), auto_add = False, db_path = "."):
	"""
	This will return everything related to the detected face.
	'face' parameter is a numpy array of a cropped face
	face_info = {
		"cropped_array": face, #this "cropped_array" key-value is omitted since it makes the pkl file enormously huge
		"pos_dim": (x, y, w, h),
		"mood": [
			['Angry', percentage], 
			['Disgust', percentage], 
			['Fear', percentage], 
			['Happy', percentage], 
			['Sad', percentage], 
			['Surprise', percentage], 
			['Neutral', percentage]
		],
		"representation": e.g. array([ 0.00743464,  0.00669238, -0.00143091, ...,  0.00044229,
        0.01007347,  0.03110152], dtype=float32), #can be omitted to save space when saving the pkl file
		"most_similar": {
			"name": employee_name,
			"relative_path": employee_relative_path,
			"distance": best_distance
		}
	}
	"""
	#detected_face = img[int(y):int(y+h), int(x):int(x+w)] #crop detected face #Youssef- isn't this same as 'face' variable? yes, if align was set to True 
	detected_face = face
	(x, y, w, h) = pos_dim
	resolution_x = resolution[1]; resolution_y = resolution[0]

	#pivot_img_size = 112 #face recognition result image
	face_info = {
		#"cropped_array": face,
		"pos_dim": (x, y, w, h),
	}

	#process emotion
	if(emotion_model is not None):
		mood_items = get_emotions(detected_face, emotion_model, detector_backend = detector_backend)
		face_info["mood"] = mood_items

	#-------------------------------
	#face recognition

	face = functions.preprocess_face(img = face, target_size = (target_size[0], target_size[1]), hard_detection_failure = False, detector_backend = detector_backend)
	
	#custom normalization. It wasn't originally part of analysis function in realtime.py
	face = functions.normalize_input(img = face, normalization = normalization)

	input_shape = (target_size[1], target_size[0])
	#check preprocess_face function handled
	if(face.shape[1:3] == input_shape and df.shape[0] > 0): #if there are images to verify, apply face recognition
		target_representation = model.predict(face)[0,:]
		#face_info["representation"] = target_representation
		employee_name, employee_relative_path, best_distance = get_most_similar_candidate(df, target_representation, threshold, img_type = img_type)
		face_info["most_similar"] = {
			"name": employee_name,
			"relative_path": employee_relative_path,
			"distance": best_distance
		}
		if(best_distance > threshold and auto_add): #save face to database but don't add it to embeddings now. We give the user the chance to name it as he wishes.
			if not os.path.isdir(db_path + "/auto_add"):
				os.mkdir(db_path + "/auto_add")

			i = 0
			name = str(i) + img_type[0] #it's on purpose that the name is a number. It's a perfect convention, so that in case the user has forgotten to rename it after being saved, it won't hold any name.
			#first name is simply "0.jpg"
			while os.path.isfile(db_path + "/auto_add/" + name):
				i += 1
				name = str(i) + img_type[0]
			
			face = face * 255
			face = np.array(face, dtype="uint8")
			print("\nauto adding unknown face...")
			imageio.imwrite(db_path + "/auto_add/" + name, face[0,:])

	return face_info

def get_most_similar_candidate(df, face_representation, threshold, img_type = (".jpg", ".jpeg", ".bmp", ".png")):
	"""
	Our convention is to return an empty name string in case no close person to our target face was detected in the database 
	"""
	def findDistance(row):
		distance_metric = row['distance_metric']
		db_img_representation = row['embedding']

		distance = 1000 #initialize very large value
		if distance_metric == 'cosine':
			distance = dst.findCosineDistance(face_representation, db_img_representation)
		elif distance_metric == 'euclidean':
			distance = dst.findEuclideanDistance(face_representation, db_img_representation)
		elif distance_metric == 'euclidean_l2':
			distance = dst.findEuclideanDistance(dst.l2_normalize(face_representation), dst.l2_normalize(db_img_representation))
		return distance

	df['distance'] = df.apply(findDistance, axis = 1)
	df = df.sort_values(by = ["distance"])

	candidate = df.iloc[0]
	employee_relative_path = candidate['employee']
	best_distance = candidate['distance']

	#print(candidate[['employee', 'distance']].values)

	if best_distance <= threshold:
		#label = re.sub('([0-9]|.jpg|.png)', '', employee_relative_path.split("/")[-1])
		label = employee_relative_path
		for extension in img_type: #removing .jpg or .png
			label = label.split("/")[-1].replace(extension, "")
		label = re.sub('[0-9]', '', label) #sustitutes any number in the name by a no char, i.e. removes it even if it was repeated anywhere
		label = re.sub('[\- _]*$', '', label) #removes any dash. space, or underscrore found only at the end of the name
	else:
		label = "" #no one in the database is considered matching "face"
	
	return label, employee_relative_path, best_distance
		
def face_inform(face_info, img): #face_info represents only 1 face
	"""
	This will show using cv2 the rectangles and texts after extracting them from face_info. 
	It won't return anything.
	"""
	(x, y, w, h) = face_info["pos_dim"]
	cv2.rectangle(img, (x,y), (x+w,y+h), (67,67,67), 1) #draw rectangle to main image
	
	#-------------------------------
	#transparency
	"""
	overlay = img.copy()
	opacity = 0.4
	pivot_img_size = 112 #face recognition result image
	resolution = img.shape; resolution_x = resolution[1]; resolution_y = resolution[0]

	if x+w+pivot_img_size < resolution_x:
		#right
		cv2.rectangle(img
			#, (x+w,y+20)
			, (x+w,y)
			, (x+w+pivot_img_size, y+h)
			, (64,64,64),cv2.FILLED)

		cv2.addWeighted(overlay, opacity, img, 1 - opacity, 0, img)

	elif x-pivot_img_size > 0:
		#left
		cv2.rectangle(img
			#, (x-pivot_img_size,y+20)
			, (x-pivot_img_size,y)
			, (x, y+h)
			, (64,64,64),cv2.FILLED)

		cv2.addWeighted(overlay, opacity, img, 1 - opacity, 0, img)
	"""
	#-------------------------------
	#emotion
	mood_items = face_info.get("mood")
	if(mood_items is not None):
		emotion_df = pd.DataFrame(mood_items, columns = ["emotion", "score"])
		emotion_df = emotion_df.sort_values(by = ["score"], ascending=False).reset_index(drop=True)

		#show only main emotion
		main_emotion = emotion_df.iloc[0]

		emotion_label = "%s " % (main_emotion['emotion'])
		
		text_location_y = y + h
		text_location_x = x

		cv2.putText(img, emotion_label, (text_location_x, text_location_y), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
		"""
		for index, instance in emotion_df.iterrows():
			emotion_label = "%s " % (instance['emotion'])
			emotion_score = instance['score']/100

			bar_x = 35 #this is the size if an emotion is 100%
			bar_x = int(bar_x * emotion_score)

			if x+w+pivot_img_size < resolution_x:

				text_location_y = y + 20 + (index+1) * 20
				text_location_x = x+w

				if text_location_y < y + h:
					cv2.putText(img, emotion_label, (text_location_x, text_location_y), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)

					cv2.rectangle(img
						, (x+w+70, y + 13 + (index+1) * 20)
						, (x+w+70+bar_x, y + 13 + (index+1) * 20 + 5)
						, (255,255,255), cv2.FILLED)

			#elif x-pivot_img_size > 0:
			else:
				text_location_y = y + 20 + (index+1) * 20
				text_location_x = x-pivot_img_size

				if text_location_y <= y+h:
					cv2.putText(img, emotion_label, (text_location_x, text_location_y), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)

					cv2.rectangle(img
						, (x-pivot_img_size+70, y + 13 + (index+1) * 20)
						, (x-pivot_img_size+70+bar_x, y + 13 + (index+1) * 20 + 5)
						, (255,255,255), cv2.FILLED)
		"""
	
	#-------------------------------
	#face recognition
	most_similar = face_info.get("most_similar")
	if(most_similar is not None):
		name = most_similar["name"]
		#print(name)		
		if(name != ""):
			"""
			display_img = cv2.imread(name)

			display_img = cv2.resize(display_img, (pivot_img_size, pivot_img_size))

			try:
				if y - pivot_img_size > 0 and x + w + pivot_img_size < resolution_x:
					#top right
					freeze_img[y - pivot_img_size:y, x+w:x+w+pivot_img_size] = display_img

					overlay = freeze_img.copy(); opacity = 0.4
					cv2.rectangle(freeze_img,(x+w,y),(x+w+pivot_img_size, y+20),(46,200,255),cv2.FILLED)
					cv2.addWeighted(overlay, opacity, freeze_img, 1 - opacity, 0, freeze_img)

					cv2.putText(freeze_img, label, (x+w, y+10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, text_color, 1)

					#connect face and text
					cv2.line(freeze_img,(x+int(w/2), y), (x+3*int(w/4), y-int(pivot_img_size/2)),(67,67,67),1)
					cv2.line(freeze_img, (x+3*int(w/4), y-int(pivot_img_size/2)), (x+w, y - int(pivot_img_size/2)), (67,67,67),1)

				elif y + h + pivot_img_size < resolution_y and x - pivot_img_size > 0:
					#bottom left
					freeze_img[y+h:y+h+pivot_img_size, x-pivot_img_size:x] = display_img

					overlay = freeze_img.copy(); opacity = 0.4
					cv2.rectangle(freeze_img,(x-pivot_img_size,y+h-20),(x, y+h),(46,200,255),cv2.FILLED)
					cv2.addWeighted(overlay, opacity, freeze_img, 1 - opacity, 0, freeze_img)

					cv2.putText(freeze_img, label, (x - pivot_img_size, y+h-10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, text_color, 1)

					#connect face and text
					cv2.line(freeze_img,(x+int(w/2), y+h), (x+int(w/2)-int(w/4), y+h+int(pivot_img_size/2)),(67,67,67),1)
					cv2.line(freeze_img, (x+int(w/2)-int(w/4), y+h+int(pivot_img_size/2)), (x, y+h+int(pivot_img_size/2)), (67,67,67),1)

				elif y - pivot_img_size > 0 and x - pivot_img_size > 0:
					#top left
					freeze_img[y-pivot_img_size:y, x-pivot_img_size:x] = display_img

					overlay = freeze_img.copy(); opacity = 0.4
					cv2.rectangle(freeze_img,(x- pivot_img_size,y),(x, y+20),(46,200,255),cv2.FILLED)
					cv2.addWeighted(overlay, opacity, freeze_img, 1 - opacity, 0, freeze_img)

					cv2.putText(freeze_img, label, (x - pivot_img_size, y+10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, text_color, 1)

					#connect face and text
					cv2.line(freeze_img,(x+int(w/2), y), (x+int(w/2)-int(w/4), y-int(pivot_img_size/2)),(67,67,67),1)
					cv2.line(freeze_img, (x+int(w/2)-int(w/4), y-int(pivot_img_size/2)), (x, y - int(pivot_img_size/2)), (67,67,67),1)

				elif x+w+pivot_img_size < resolution_x and y + h + pivot_img_size < resolution_y:
					#bottom righ
					freeze_img[y+h:y+h+pivot_img_size, x+w:x+w+pivot_img_size] = display_img

					overlay = freeze_img.copy(); opacity = 0.4
					cv2.rectangle(freeze_img,(x+w,y+h-20),(x+w+pivot_img_size, y+h),(46,200,255),cv2.FILLED)
					cv2.addWeighted(overlay, opacity, freeze_img, 1 - opacity, 0, freeze_img)

					cv2.putText(freeze_img, label, (x+w, y+h-10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, text_color, 1)

					#connect face and text
					cv2.line(freeze_img,(x+int(w/2), y+h), (x+int(w/2)+int(w/4), y+h+int(pivot_img_size/2)),(67,67,67),1)
					cv2.line(freeze_img, (x+int(w/2)+int(w/4), y+h+int(pivot_img_size/2)), (x+w, y+h+int(pivot_img_size/2)), (67,67,67),1)
			except Exception as err:
				print(str(err))
			"""	
			cv2.putText(img, name, (x, y), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 1)
		
	
	pass

def get_emotions(face, emotion_model, detector_backend = 'opencv'):
	gray_img = functions.preprocess_face(img = face, target_size = (48, 48), grayscale = True, hard_detection_failure = False, detector_backend = 'opencv')
	emotion_labels = ['Angry', 'Disgust', 'Fear', 'Happy', 'Sad', 'Surprise', 'Neutral']
	emotion_predictions = emotion_model.predict(gray_img)[0,:]
	sum_of_predictions = emotion_predictions.sum()

	mood_items = []
	for i in range(0, len(emotion_labels)):
		mood_item = []
		emotion_label = emotion_labels[i]
		emotion_prediction = 100 * emotion_predictions[i] / sum_of_predictions
		mood_item.append(emotion_label)
		mood_item.append(emotion_prediction)
		mood_items.append(mood_item)

	return mood_items


def print_license():
	#license to thank Mr. Sefik Ilkin Serengil for his wonderful work which he made available on Github. It's mandatory to put the license, and I hope I can pay him back...
	print("MIT License\n\nCopyright (c) 2019 Sefik Ilkin Serengil\n\nPermission is hereby granted, free of charge, to any person obtaining a copy of this software and associated documentation files (the \"Software\"), to deal in the Software without restriction, including without limitation the rights to use, copy, modify, merge, publish, distribute, sublicense, and/or sell copies of the Software, and to permit persons to whom the Software is furnished to do so, subject to the following conditions:\n\nThe above copyright notice and this permission notice shall be included in all copies or substantial portions of the Software.\n\nTHE SOFTWARE IS PROVIDED \"AS IS\", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.")
	print("\n\nEnd of license\n\n")
	print("Note that further development has been made to customize the code to work in the way we desired. Contact the developer @diraniyoussef on Telegram to ask for something.\n\n")
	pass

def create_representation_file(file_name, model_names, models, represent, db_path = ".", model_name = 'VGG-Face', hard_detection_failure = True, detector_backend = 'opencv', align = True, normalization = 'base', prog_bar = True, img_type = (".jpg", ".jpeg", ".bmp", ".png")):
	
	employees = get_employees(db_path, img_type, path_type = "exact")

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

	print("Saving Representations...")
	save_pkl(representations, db_path+'/'+file_name)
	print("Please delete the representations file when you add new identities in your database.")

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
