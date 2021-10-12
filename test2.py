import moviepy.editor as mp #https://towardsdatascience.com/extracting-audio-from-video-using-python-58856a940fd
from os import path
import wave
import pyaudio

#audio_path = r"C:/Users/Dirani/ProgrammingProjects/DeepFace_Project/make a video/6 seconds.wav"
audio_path = "/home/youssef/PythonProjects/AI/DeepFace_Project/make a video/6 seconds.wav"

#getting audio file from video file
if not path.exists(audio_path):
	#my_clip = mp.VideoFileClip(r"C:/Users/Dirani/ProgrammingProjects/DeepFace_Project/make a video/6 seconds.mp4")
	my_clip = mp.VideoFileClip("/home/youssef/PythonProjects/AI/DeepFace_Project/make a video/6 seconds.mp4")
	my_clip.audio.write_audiofile(audio_path)

#playing audio file
#CHUNK = 1024
CHUNK = 50
wf = wave.open(audio_path, 'rb')

# instantiate PyAudio (1)
p = pyaudio.PyAudio()

# open stream (2)
stream = p.open(format=p.get_format_from_width(wf.getsampwidth()),
                channels=wf.getnchannels(),
                rate=wf.getframerate(),
                output=True)

# read data
data = wf.readframes(CHUNK)

# play stream (3)
while len(data) > 0:
    stream.write(data)
    data = wf.readframes(CHUNK)

# stop stream (4)
stream.stop_stream()
stream.close()

# close PyAudio (5)
p.terminate()