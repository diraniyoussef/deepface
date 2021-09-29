import os
#DeepFace.enhanced_stream(db_path = '/home/youssef/database2', skip_no_face_images = True, source = '/home/youssef/database2/hi.mp4')
#DeepFace.enhanced_stream(db_path = '/home/youssef/database2', source = '/home/youssef/Videos/Webcam/2021-05-20-035437.mp4')
#DeepFace.enhanced_stream(db_path = '/home/youssef/database2', source = '/home/youssef/make a video/Zaher and Issa.mp4')
#DeepFace.enhanced_stream(db_path = '/home/youssef/database2', source = '/home/youssef/make a video/selfie.mp4')
#DeepFace.enhanced_stream(db_path = '/home/youssef/PythonProjects/AI/DeepFace_Project/database2', source = '/home/youssef/PythonProjects/AI/DeepFace_Project/make a video/youssef.mp4', actions = [])
#DeepFace.enhanced_stream(db_path = '/home/youssef/PythonProjects/AI/DeepFace_Project/database2', source = '/home/youssef/PythonProjects/AI/DeepFace_Project/make a video/teta image.mp4', actions = [], number_of_processes = 1)
#DeepFace.enhanced_stream(db_path = '/home/youssef/PythonProjects/AI/DeepFace_Project/database3', source = '/home/youssef/PythonProjects/AI/DeepFace_Project/make a video/6seconds.mp4', actions = [], number_of_processes = 2)
#DeepFace.enhanced_stream(db_path = r'C:\Users\Dirani\ProgrammingProjects\DeepFace_Project\database3', source = r'C:\Users\Dirani\ProgrammingProjects\DeepFace_Project\make a video\6 seconds.mp4', actions = [], number_of_processes = 2)

if __name__ == "__main__":
	from deepface import DeepFace
	DeepFace.enhanced_stream(db_path = 'C:/Users/Dirani/ProgrammingProjects/DeepFace_Project/database3', source = 'C:/Users/Dirani/ProgrammingProjects/DeepFace_Project/make a video/6 seconds.mp4', actions = [], number_of_processes = 2)

def get_embeddings_process(employees, model, represent, index, db_path = ".", target_size = (224, 224), hard_detection_failure = False, detector_backend = 'opencv', normalization = 'base'):
	"""
	For technical reasons which have to do with pool in multiprocessing I had to put this function in the top level; it cannot be nested in get_embeddings function.
	And also for technical reasons, index is the last parameter. This has to do with tqdm with pool.

	This function will return a dictionary e.g. either
	{"embedding": [employee, representation]}
	or
	{"image_undetected_faces_index":index}
	"""
	print("starting process of id {}".format(os.getpid()))
	
	employee = employees[index] #it's a copy byval, not references
	#pbar.set_description("Finding embedding for %s" % (employee.split("/")[-1])) #according to usage employee can be a full exact path or just a path after (without) the db_path. Both cases, .split("/")[-1] works fine. employee may even not contain '/' and it works fine.
	
	try: #this try-except is useful in case hard_detection_failure was set to True and not face was detected
		img_representation = represent(db_path +'/'+ employee, model, target_size = (target_size[0], target_size[1]), hard_detection_failure = hard_detection_failure, detector_backend = detector_backend, normalization = normalization)
		return {"embedding":[employee, img_representation]}
	except Exception as err:
		#print(err) #usual message is as follows : Face could not be detected. Please confirm that the picture is a face photo or consider to set hard_detection_failure param to False.
		print("Could not detect a face in this image : {}".format(employee))
		return {"image_undetected_faces_index":index}

