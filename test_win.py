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
	#DeepFace.enhanced_stream(db_path = 'C:/Users/Dirani/ProgrammingProjects/DeepFace_Project/database', source = 'C:/Users/Dirani/ProgrammingProjects/DeepFace_Project/make a video/الأهل و التلاميذ و الفصحى.mp4', processing_video_size = (), model_name = "VGG-Face", detector_backend="dlib", actions = [], number_of_processes = 2, auto_add = True, normalization = "base", distance_metric = 'cosine') #detector_backend="dlib", detector_backend="mtcnn"
	#DeepFace.enhanced_stream(db_path = 'C:/Users/Dirani/ProgrammingProjects/DeepFace_Project/database', source = 'https://www.youtube.com/watch?v=g1Aq16fGeeQ&ab_channel=NathanKutz', source_type= "youtube", model_name = "VGG-Face", detector_backend="dlib", actions = [], number_of_processes = 2, auto_add = True, normalization = "base", distance_metric = 'cosine') #detector_backend="dlib", detector_backend="mtcnn", processing_video_size = (960, 540)
	#DeepFace.prepend_imgs_names("C:/Users/Dirani/ProgrammingProjects/DeepFace_Project/database/auto_add", name = "Nathan Kutz")
	#DeepFace.play_with_annotations("C:/Users/Dirani/ProgrammingProjects/DeepFace_Project/make a video/6 seconds.mp4", "C:/Users/Dirani/ProgrammingProjects/DeepFace_Project/database/frames_info_6 seconds.pkl", speed = "slow", fps = 30, output_video_size = (800,600)) # (144, 82)
	#DeepFace.play_with_annotations("C:/Users/Dirani/ProgrammingProjects/DeepFace_Project/make a video/الأهل و التلاميذ و الفصحى.mp4", "C:/Users/Dirani/ProgrammingProjects/DeepFace_Project/database/frames_info_الأهل و التلاميذ و الفصحى.pkl", speed = "slow", fps = 30, processing_video_size = (), output_video_size= (960, 540)) #  (82, 144)
	#DeepFace.play_with_annotations("https://www.youtube.com/watch?v=g1Aq16fGeeQ&ab_channel=NathanKutz", "C:/Users/Dirani/ProgrammingProjects/DeepFace_Project/database/frames_info_programming logic  if and for.pkl", source_type = "youtube", speed = "fast", fps = 24)


