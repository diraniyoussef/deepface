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


