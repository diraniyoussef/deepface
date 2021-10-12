import moviepy.editor as mp #https://towardsdatascience.com/extracting-audio-from-video-using-python-58856a940fd
from os import path
import wave

audio_path = r"C:/Users/Dirani/ProgrammingProjects/DeepFace_Project/make a video/6 seconds.wav"
if not path.exists(audio_path):
	my_clip = mp.VideoFileClip(r"C:/Users/Dirani/ProgrammingProjects/DeepFace_Project/make a video/6 seconds.mp4")

	my_clip.audio.write_audiofile(audio_path)
else:
	wf = wave.open(audio_path, 'rb')
	