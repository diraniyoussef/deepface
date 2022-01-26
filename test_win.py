#from deepface import DeepFace
#DeepFace.prepend_imgs_names(imgs_path = "C:/Users/Dirani/ProgrammingProjects/DeepFace_Project/database6/auto_add", name = "bcc reporter")
#DeepFace.enhanced_stream(db_path= "C:/Users/Dirani/ProgrammingProjects/DeepFace_Project/database6", source= "https://www.youtube.com/watch?v=z-BgpfYb_uU", source_type="youtube", actions = ['emotion'], auto_add = True, model_name ='DeepFace')
#import test_win
#test_win.play_with_annotations("https://www.youtube.com/watch?v=z-BgpfYb_uU", "C:/Users/Dirani/ProgrammingProjects/DeepFace_Project/database6/frames_info_prime minister of pakistan imran khan exclusive interview on bbc news.pkl", video_place= "youtube", color= "black")

#import test_win
#test_win.play_with_annotations("C:/Users/Dirani/ProgrammingProjects/DeepFace_Project/make a video/1.mp4", "C:/Users/Dirani/ProgrammingProjects/DeepFace_Project/database4/frames_info_1.pkl", audio= True)

from deepface import DeepFace
#DeepFace.enhanced_stream(db_path= "C:/Users/Dirani/ProgrammingProjects/DeepFace_Project/database4", source= "C:/Users/Dirani/ProgrammingProjects/DeepFace_Project/make a video/1.mp4", source_type="disk", actions = ['emotion'], model_name ='DeepFace')

	#test_win.play_with_annotations("C:/Users/Dirani/ProgrammingProjects/DeepFace_Project/make a video/1.mp4", "C:/Users/Dirani/ProgrammingProjects/DeepFace_Project/database4/frames_info_1.pkl")
	#test_win.search_stream("C:/Users/Dirani/ProgrammingProjects/DeepFace_Project/database6", "https://www.youtube.com/watch?v=z-BgpfYb_uU", video_place="youtube", auto_add = True)
	#test_win.play_with_annotations("https://www.youtube.com/watch?v=z-BgpfYb_uU", "C:/Users/Dirani/ProgrammingProjects/DeepFace_Project/database6/frames_info_prime minister of pakistan imran khan exclusive interview on bbc news.pkl", video_place= "youtube")

	#instructions.instructions()

if __name__=="__main__":
	#DeepFace.enhanced_stream(db_path = 'C:/Users/Dirani/ProgrammingProjects/DeepFace_Project/database1', source = 'C:/Users/Dirani/ProgrammingProjects/DeepFace_Project/videos/trial1.mp4', processing_video_size = (), model_name = "VGG-Face", detector_backend="dlib", actions = [], number_of_processes = 1, auto_add = True, normalization = "base", distance_metric = 'cosine', process_rt = True) #detector_backend="dlib", detector_backend="mtcnn"
	DeepFace.play_with_annotations("C:/Users/Dirani/ProgrammingProjects/DeepFace_Project/videos/trial1.mp4", "C:/Users/Dirani/ProgrammingProjects/DeepFace_Project/database1/frames_info_trial1.pkl", speed = "normal", fps = 30, output_video_size = (800,600)) # (144, 82)
	#DeepFace.enhanced_stream(db_path = 'C:/Users/Dirani/ProgrammingProjects/DeepFace_Project/database3', source = 'C:/Users/Dirani/ProgrammingProjects/DeepFace_Project/videos/3.mp4', processing_video_size = (), model_name = "VGG-Face", detector_backend="dlib", actions = [], number_of_processes = 1, auto_add = True, normalization = "base", distance_metric = 'cosine', process_rt = True) #detector_backend="dlib", detector_backend="mtcnn"

	#DeepFace.enhanced_stream(db_path = 'C:/Users/Dirani/ProgrammingProjects/DeepFace_Project/database', source = 'https://www.youtube.com/watch?v=g1Aq16fGeeQ&ab_channel=NathanKutz', source_type= "youtube", model_name = "VGG-Face", detector_backend="dlib", actions = [], number_of_processes = 2, auto_add = True, normalization = "base", distance_metric = 'cosine') #detector_backend="dlib", detector_backend="mtcnn", processing_video_size = (960, 540)
	#DeepFace.prepend_imgs_names("C:/Users/Dirani/ProgrammingProjects/DeepFace_Project/database/auto_add", name = "Nathan Kutz")
	
	#DeepFace.play_with_annotations("C:/Users/Dirani/ProgrammingProjects/DeepFace_Project/make a video/الأهل و التلاميذ و الفصحى.mp4", "C:/Users/Dirani/ProgrammingProjects/DeepFace_Project/database/frames_info_الأهل و التلاميذ و الفصحى.pkl", speed = "slow", fps = 30, processing_video_size = (), output_video_size= (960, 540)) #  (82, 144)
	#DeepFace.play_with_annotations("https://www.youtube.com/watch?v=g1Aq16fGeeQ&ab_channel=NathanKutz", "C:/Users/Dirani/ProgrammingProjects/DeepFace_Project/database/frames_info_programming logic  if and for.pkl", source_type = "youtube", speed = "fast", fps = 24)
