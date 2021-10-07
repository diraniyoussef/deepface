#DeepFace.enhanced_stream(db_path = 'C:\Users\Dirani\ProgrammingProjects\DeepFace_Project\database2', source = 'C:\Users\Dirani\ProgrammingProjects\DeepFace_Project\make a video\Zaher and Issa.mp4')
#DeepFace.enhanced_stream(db_path = 'C:\Users\Dirani\ProgrammingProjects\DeepFace_Project\database2', source = 'C:\Users\Dirani\ProgrammingProjects\DeepFace_Project\make a video\selfie.mp4')
#DeepFace.enhanced_stream(db_path = 'C:\Users\Dirani\ProgrammingProjects\DeepFace_Project\database2', source = 'C:\Users\Dirani\ProgrammingProjects\DeepFace_Project\make a video\youssef.mp4', actions = [])
#DeepFace.enhanced_stream(db_path = 'C:/Users/Dirani/ProgrammingProjects/DeepFace_Project/database2', source = 'C:/Users/Dirani/ProgrammingProjects/DeepFace_Project/make a video/teta image.mp4', actions = [], number_of_processes = 1)
#DeepFace.enhanced_stream(db_path = r'C:\Users\Dirani\ProgrammingProjects\DeepFace_Project\database3', source = r'C:\Users\Dirani\ProgrammingProjects\DeepFace_Project\make a video\6 seconds.mp4', actions = [], number_of_processes = 2)

if __name__ == "__main__":
	#from deepface import instructions
	from deepface import DeepFace
	#we won't care much for the slash or back-slash in the paths (especially the \U), since they'll be given by the user as string inputs, so they'll be probably well encoded

	#instructions.instructions()
	#DeepFace.enhanced_stream(db_path = 'C:/Users/Dirani/ProgrammingProjects/DeepFace_Project/database', source = 'C:/Users/Dirani/ProgrammingProjects/DeepFace_Project/make a video/me.mp4', model_name = "VGG-Face", detector_backend="dlib", actions = [], number_of_processes = 2, auto_add = True, normalization = "base", distance_metric = 'cosine') #detector_backend="dlib", detector_backend="mtcnn"
	DeepFace.enhanced_stream(db_path = 'C:/Users/Dirani/ProgrammingProjects/DeepFace_Project/database', source = 'https://www.youtube.com/watch?v=s7s4u8a3Ie8', youtube = True, model_name = "VGG-Face", detector_backend="dlib", actions = [], number_of_processes = 2, auto_add = True, normalization = "base", distance_metric = 'cosine') #detector_backend="dlib", detector_backend="mtcnn"
	#DeepFace.play_with_annotations("C:/Users/Dirani/ProgrammingProjects/DeepFace_Project/make a video/me.mp4", "C:/Users/Dirani/ProgrammingProjects/DeepFace_Project/database/frames_info_me.pkl", speed = "slow", fps = 30, im_size = (960, 540))


