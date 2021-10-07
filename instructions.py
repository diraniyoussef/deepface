def instructions():
	print("\nIn the database, it's better that their names be in English.\n")
	print("\nThese names must not be made of just numbers.\n")
	print("\nenhanced_stream actually does 2 things : it finds representations, or so called embeddings, for the images in the database, and secondly it processes a video. If the user just wants the first function, he can omit the \'source\' argument.\n")
	print("\nWhen processing, it's optional, and probably better not, to use processing_video_size to enter the width and height which will resize the video, but careful when using it because you don't to distort the image which may result in detection failure.\n")
	print("\nWhen playing with annotations, it can be helpful to use processing_video_size if relevant and known to you.\nNote that output_video_size is the replay size which you will see.\n")
	