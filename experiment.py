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
	#DeepFace.enhanced_stream(db_path = 'C:/Users/Dirani/ProgrammingProjects/DeepFace_Project/database3', source = 'C:/Users/Dirani/ProgrammingProjects/DeepFace_Project/make a video/6 seconds.mp4', actions = [], number_of_processes = 2)
	DeepFace.enhanced_stream(db_path = '/home/youssef/PythonProjects/AI/DeepFace_Project/database3', source = '/home/youssef/PythonProjects/AI/DeepFace_Project/make a video/6 seconds.mp4', actions = [], number_of_processes = 2)

