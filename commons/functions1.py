import os
from deepface.basemodels import Boosting
from git import Repo
import time

from tqdm import tqdm
import pickle
from deepface.commons import distance as dst, functions

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

def check_change(db_path=".", img_type = (".jpg", ".png")): #this whole function can be called in its own thread
	"""
	The goal of this function check_change is to trace the files without having to read the whole database for any tiny change made by the user like adding an image or deleting another.
	"""
	#User might add an image to his database, might rename, move, delete, etc...
	try:
		repo = Repo(db_path) #this throws an error if it's not git-initialized already
	except Exception:
		repo = Repo.init(db_path, bare=False) #initializing
	
	#img_type = (".jpeg", ".jpg", ".png", ".bmp")
	#check for untracked files
	to_be_added_images_list = []
	for f in repo.untracked_files: 
		if(f.lower().endswith(img_type)):
			to_be_added_images_list.append(f) #images aren't tracked yet
	#check for unstaged files (which are already tracked)
	to_be_removed_images_list = []
	for x in repo.index.diff(None):
		if(x.b_path.endswith(img_type)):#added in case someone messes with the repo, like adds something which isn't an image to the git staging area i.e. tracking it.
			if(x.change_type == 'M'): #not sure how an image file could be modified, anyway
				to_be_added_images_list.append(x.b_path)
				to_be_removed_images_list.append(x.b_path) #this is in case 2 committed files interchanged their names together, so in a best effort manner they must both be removed, then they both must added
			elif(x.change_type == 'D'):
				to_be_removed_images_list.append(x.b_path)
	#Although user might rename an image and it might be worthy to trace such thing, but it is not a direct process in git so postponed. A renamed file is deleted and made new when it comes to git and us.
		
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
	
	return (to_be_added_images_list, to_be_removed_images_list)

def update_embeddings(embeddings, removed_images_list, added_images_list, model, represent, db_path = ".", target_size = (224, 224), hard_detection_failure = False, detector_backend = 'opencv', normalization = 'base'):
	#if(len(removed_images_list)!=0):
	for removed_image in removed_images_list: # processing removed images are made on purpose before added images, for the sake of modified files.
		for i in range(len(embeddings)):					
			if(embeddings[i][0] == removed_image):
				embeddings.pop(i)
				break # even if user has named 2 images in separate folders the same name, it still works since we deal with relative paths
	if(len(added_images_list)!=0):#not needed but anyway.	
		added_embeddings = get_embeddings(added_images_list, model, represent, db_path = db_path, target_size = (target_size[0], target_size[1]), hard_detection_failure = hard_detection_failure, detector_backend = detector_backend, normalization = normalization)
		embeddings.extend(added_embeddings)
		embeddings.sort() #list sort is smart; it sorts first according to first column (which we care only about) then according to second column
	return embeddings

def check_git_and_update_embeddings(embeddings, model, represent, pkl_path, db_path = ".", target_size = (224, 224), hard_detection_failure = False, detector_backend = 'opencv', normalization = 'base', img_type = (".jpg", ".png")):
	#check git for possible user-made changes then commit
	added_images_list, removed_images_list = check_change(db_path = db_path, img_type = img_type)
	print("Supposed updates in database after last save to", pkl_path, "are :")
	print("added images list :", added_images_list)
	print("removed images list :", removed_images_list)
	try:
		embeddings = update_embeddings(embeddings, removed_images_list, added_images_list, model, represent, db_path = db_path, target_size = (target_size[0], target_size[1]), hard_detection_failure = hard_detection_failure, detector_backend = detector_backend, normalization = normalization)
	except Exception as err:
		print(err)
	return embeddings

def get_employees(db_path = ".", img_type = (".jpg", ".png"), path_type = "exact"):
	"""
    if path_type is "relative" the function returns list of relative paths of images i.e. path starting from the db_path excluding it. (any other keyword than "exact" will be considered as "relative")
    elif path_type is "exact" the function returns list of exact paths i.e. path starting from the db_path including.
	"""
	employees = []
	for r, d, f in os.walk(db_path): # r=root, d=directories, f = files
		if(r.split("/")[-1] == ".git"):
			continue
		for file in f:
			for t in img_type:
				if (file.lower().endswith(t)):
					path = r + "/" + file # exact path
                    #exact_path = os.path.join(r, file)
					if(path_type != "exact"): # relative path
						path = path[len(db_path)+1:] # +1 for the '/' after db_path and before relative path
					employees.append(path)
					break
	return employees

def get_embeddings(employees, model, represent, db_path = ".", target_size = (224, 224), hard_detection_failure = False, detector_backend = 'opencv', normalization = 'base'):
    #this is almost like the represent function in DeepFace module, but more fitting to the use case

	pbar = tqdm(range(0,len(employees)), position= 0)

	embeddings = []
	#for employee in employees:
	for index in pbar:
		employee = employees[index] #it's a copy byval, not references
		pbar.set_description("Finding embedding for %s" % (employee.split("/")[-1])) #according to usage employee can be a full exact path or just a path after (without) the db_path. Both cases, .split("/")[-1] works fine. employee may even not contain '/' and it works fine.
		embedding = []
		
		try: #this try-except is useful in case hard_detection_failure was set to True and not face was detected
			img_representation = represent(db_path +'/'+ employee, model, target_size = (target_size[0], target_size[1]), hard_detection_failure = hard_detection_failure, detector_backend = detector_backend, normalization = normalization)
		except Exception as err:
			#print(err) #usual message is as follows : Face could not be detected. Please confirm that the picture is a face photo or consider to set hard_detection_failure param to False.
			print("Could not detect a face in this image :", employee)
			continue

		embedding.append(employee)
		embedding.append(img_representation)
		embeddings.append(embedding)
	pbar.set_description("All embeddings tried.")
	
	return embeddings


def save_pkl(content = [], exact_path = "representations.pkl"):
	print("Representations stored in ", exact_path, " file")
	f = open(exact_path, "wb") #this makes a new file or completely overrides an existing one
	pickle.dump(content, f)
	f.close()

def create_representation_file(file_name, model_names, models, represent, db_path = ".", model_name = 'VGG-Face', hard_detection_failure = True, detector_backend = 'opencv', align = True, normalization = 'base', prog_bar = True, img_type = (".jpg", ".png")):
	
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
