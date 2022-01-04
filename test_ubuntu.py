#DeepFace.enhanced_stream(db_path = '/home/youssef/database2', source = '/home/youssef/Videos/Webcam/2021-05-20-035437.mp4')
#DeepFace.enhanced_stream(db_path = '/home/youssef/database2', source = '/home/youssef/make a video/Zaher and Issa.mp4')
#DeepFace.enhanced_stream(db_path = '/home/youssef/database2', source = '/home/youssef/make a video/selfie.mp4')
#DeepFace.enhanced_stream(db_path = '/home/youssef/ProgrammingProjects/AI/DeepFace_Project/database2', source = '/home/youssef/ProgrammingProjects/AI/DeepFace_Project/make a video/youssef.mp4', actions = [])
#DeepFace.enhanced_stream(db_path = '/home/youssef/ProgrammingProjects/AI/DeepFace_Project/database2', source = '/home/youssef/ProgrammingProjects/AI/DeepFace_Project/make a video/teta image.mp4', actions = [], number_of_processes = 1)
#from commons.functions1 import OptimizeResources


if __name__ == "__main__":
	#from deepface import instructions
	from deepface import DeepFace

	#instructions.instructions()
	DeepFace.enhanced_stream(db_path = '/home/youssef/ProgrammingProjects/AI/DeepFace_Project/database1', source = '/home/youssef/ProgrammingProjects/AI/DeepFace_Project/make a video/me.mp4', model_name = "VGG-Face", detector_backend="dlib", actions = [], number_of_processes = 2, auto_add = True, normalization = "base", distance_metric = 'cosine')
	#DeepFace.play_with_annotations("/home/youssef/ProgrammingProjects/AI/DeepFace_Project/make a video/teta image.mp4", "/home/youssef/ProgrammingProjects/AI/DeepFace_Project/database1/frames_info_teta image.pkl", output_video_size = (540, 960), audio=True)
	"""
	from deepface.commons import functions1
	OR = functions1.OptimizeResources
	optimize_resources = OR()
	optimize_resources.controller()
	"""