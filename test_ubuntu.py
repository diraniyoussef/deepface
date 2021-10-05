#DeepFace.enhanced_stream(db_path = '/home/youssef/database2', source = '/home/youssef/Videos/Webcam/2021-05-20-035437.mp4')
#DeepFace.enhanced_stream(db_path = '/home/youssef/database2', source = '/home/youssef/make a video/Zaher and Issa.mp4')
#DeepFace.enhanced_stream(db_path = '/home/youssef/database2', source = '/home/youssef/make a video/selfie.mp4')
#DeepFace.enhanced_stream(db_path = '/home/youssef/PythonProjects/AI/DeepFace_Project/database2', source = '/home/youssef/PythonProjects/AI/DeepFace_Project/make a video/youssef.mp4', actions = [])
#DeepFace.enhanced_stream(db_path = '/home/youssef/PythonProjects/AI/DeepFace_Project/database2', source = '/home/youssef/PythonProjects/AI/DeepFace_Project/make a video/teta image.mp4', actions = [], number_of_processes = 1)
#  
if __name__ == "__main__":
	from deepface import instructions
	from deepface import DeepFace

	instructions.instructions()
	DeepFace.enhanced_stream(db_path = '/home/youssef/PythonProjects/AI/DeepFace_Project/database1', source = '/home/youssef/PythonProjects/AI/DeepFace_Project/make a video/6seconds.mp4', actions = [], number_of_processes = 2, auto_add = True)
	#DeepFace.play_with_annotations("C:/Users/Dirani/ProgrammingProjects/DeepFace_Project/make a video/trial2.mp4", "C:/Users/Dirani/ProgrammingProjects/DeepFace_Project/database/frames_info_trial2.pkl", speed = "slow", fps = 30, im_size = (960, 540))
