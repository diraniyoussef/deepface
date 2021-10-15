#from deepface import instructions

def search_stream(images_folder, video_file, video_place = "disk", processing_video_size = (), emotion = False, number_of_processes = 2, auto_add = False, model_name = 'DeepFace'):
	from deepface import DeepFace
	if emotion:
		actions = ['emotion']
	else:
		actions = []
	DeepFace.enhanced_stream(db_path = images_folder, source = video_file, source_type = video_place, processing_video_size = processing_video_size, model_name = model_name, detector_backend="dlib", actions = actions, number_of_processes = number_of_processes, auto_add = auto_add, normalization = "base", distance_metric = 'cosine')

def prepend_images_names(images_folder, person_name):
	from deepface import DeepFace
	DeepFace.prepend_imgs_names(imgs_path=images_folder, name = person_name)

def play_with_annotations(video_file, info_file, speed = "normal", video_place = "disk", processing_video_size = (), output_video_size = (), audio = False, color= "dark_gray"):
	from deepface import DeepFace
	DeepFace.play_with_annotations(video_file, info_file, speed= speed, source_type= video_place, processing_video_size= processing_video_size, output_video_size= output_video_size, audio= audio, color= color)


if __name__ == "__main__":
	pass
	#test_win.play_with_annotations("C:/Users/Dirani/ProgrammingProjects/DeepFace_Project/make a video/1.mp4", "C:/Users/Dirani/ProgrammingProjects/DeepFace_Project/database4/frames_info_1.pkl")
	#test_win.search_stream("C:/Users/Dirani/ProgrammingProjects/DeepFace_Project/database6", "https://www.youtube.com/watch?v=z-BgpfYb_uU", video_place="youtube", auto_add = True)
	#test_win.play_with_annotations("https://www.youtube.com/watch?v=z-BgpfYb_uU", "C:/Users/Dirani/ProgrammingProjects/DeepFace_Project/database6/frames_info_prime minister of pakistan imran khan exclusive interview on bbc news.pkl", video_place= "youtube")


	#instructions.instructions()
	#DeepFace.enhanced_stream(db_path = 'C:/Users/Dirani/ProgrammingProjects/DeepFace_Project/database', source = 'C:/Users/Dirani/ProgrammingProjects/DeepFace_Project/make a video/الأهل و التلاميذ و الفصحى.mp4', processing_video_size = (), model_name = "VGG-Face", detector_backend="dlib", actions = [], number_of_processes = 2, auto_add = True, normalization = "base", distance_metric = 'cosine') #detector_backend="dlib", detector_backend="mtcnn"
	#DeepFace.enhanced_stream(db_path = 'C:/Users/Dirani/ProgrammingProjects/DeepFace_Project/database', source = 'https://www.youtube.com/watch?v=g1Aq16fGeeQ&ab_channel=NathanKutz', source_type= "youtube", model_name = "VGG-Face", detector_backend="dlib", actions = [], number_of_processes = 2, auto_add = True, normalization = "base", distance_metric = 'cosine') #detector_backend="dlib", detector_backend="mtcnn", processing_video_size = (960, 540)
	#DeepFace.prepend_imgs_names("C:/Users/Dirani/ProgrammingProjects/DeepFace_Project/database/auto_add", name = "Nathan Kutz")
	#DeepFace.play_with_annotations("C:/Users/Dirani/ProgrammingProjects/DeepFace_Project/make a video/6 seconds.mp4", "C:/Users/Dirani/ProgrammingProjects/DeepFace_Project/database/frames_info_6 seconds.pkl", speed = "slow", fps = 30, output_video_size = (800,600)) # (144, 82)
	#DeepFace.play_with_annotations("C:/Users/Dirani/ProgrammingProjects/DeepFace_Project/make a video/الأهل و التلاميذ و الفصحى.mp4", "C:/Users/Dirani/ProgrammingProjects/DeepFace_Project/database/frames_info_الأهل و التلاميذ و الفصحى.pkl", speed = "slow", fps = 30, processing_video_size = (), output_video_size= (960, 540)) #  (82, 144)
	#DeepFace.play_with_annotations("https://www.youtube.com/watch?v=g1Aq16fGeeQ&ab_channel=NathanKutz", "C:/Users/Dirani/ProgrammingProjects/DeepFace_Project/database/frames_info_programming logic  if and for.pkl", source_type = "youtube", speed = "fast", fps = 24)


