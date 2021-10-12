import moviepy.editor as mp #https://towardsdatascience.com/extracting-audio-from-video-using-python-58856a940fd
from os import path
import wave
import pyaudio

audio_path = r"C:/Users/Dirani/ProgrammingProjects/DeepFace_Project/make a video/for sure.wav"
#audio_path = "/home/youssef/PythonProjects/AI/DeepFace_Project/make a video/6 seconds.wav"

#getting audio file from video file
if not path.exists(audio_path):
	my_clip = mp.VideoFileClip(r"C:/Users/Dirani/ProgrammingProjects/DeepFace_Project/make a video/Zaher.mp4")
	#my_clip = mp.VideoFileClip("/home/youssef/PythonProjects/AI/DeepFace_Project/make a video/6 seconds.mp4")
	my_clip.audio.write_audiofile(audio_path)

#playing audio file
#CHUNK = 1024 #i was 229
#CHUNK = 2048 #i was 115
#CHUNK = 10240 # i was 23
#CHUNK = 40000 #i was 6
#CHUNK = 100000 #i was 3
CHUNK = 1 #i is 233730
#CHUNK = 30 #i was 7791
wf = wave.open(audio_path, 'rb')

# instantiate PyAudio (1)
p = pyaudio.PyAudio()

# open stream (2)
stream = p.open(format=p.get_format_from_width(wf.getsampwidth()),
                channels=wf.getnchannels(),
                rate=wf.getframerate(),
                output=True)

print("wf.getsampwidth", wf.getsampwidth())
print("p.get_format_from_width", p.get_format_from_width(wf.getsampwidth()))
print("wf.getframerate", wf.getframerate())


# read data
data = wf.readframes(CHUNK)

# play stream (3)
i = 0
while len(data) > 0:
	i+=1
	stream.write(data)
	data = wf.readframes(CHUNK)

print(i)
# stop stream (4)
stream.stop_stream()
stream.close()

# close PyAudio (5)
p.terminate()