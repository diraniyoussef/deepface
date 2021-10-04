#DeepFace.enhanced_stream(db_path = 'C:\Users\Dirani\ProgrammingProjects\DeepFace_Project\database2', source = 'C:\Users\Dirani\ProgrammingProjects\DeepFace_Project\make a video\Zaher and Issa.mp4')
#DeepFace.enhanced_stream(db_path = 'C:\Users\Dirani\ProgrammingProjects\DeepFace_Project\database2', source = 'C:\Users\Dirani\ProgrammingProjects\DeepFace_Project\make a video\selfie.mp4')
#DeepFace.enhanced_stream(db_path = 'C:\Users\Dirani\ProgrammingProjects\DeepFace_Project\database2', source = 'C:\Users\Dirani\ProgrammingProjects\DeepFace_Project\make a video\youssef.mp4', actions = [])
#DeepFace.enhanced_stream(db_path = 'C:/Users/Dirani/ProgrammingProjects/DeepFace_Project/database2', source = 'C:/Users/Dirani/ProgrammingProjects/DeepFace_Project/make a video/teta image.mp4', actions = [], number_of_processes = 1)
#DeepFace.enhanced_stream(db_path = r'C:\Users\Dirani\ProgrammingProjects\DeepFace_Project\database3', source = r'C:\Users\Dirani\ProgrammingProjects\DeepFace_Project\make a video\6 seconds.mp4', actions = [], number_of_processes = 2)

if __name__ == "__main__":
	from deepface import DeepFace
	try:
		#db_path = input("Please input db_path") #'C:\Users\Dirani\ProgrammingProjects\DeepFace_Project\database1'
		DeepFace.enhanced_stream(db_path = 'C:/Users/Dirani/ProgrammingProjects/DeepFace_Project/database1', source = 'C:/Users/Dirani/ProgrammingProjects/DeepFace_Project/make a video/6seconds.mp4', model_name = "DeepFace", actions = [], number_of_processes = 2)
	except SyntaxError as err:
		#if(type(err)==tuple):
		print("that's it")
		#print(err)

