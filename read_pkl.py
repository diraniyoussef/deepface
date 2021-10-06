import pickle
#pkl_file = 'C:/Users/Dirani/ProgrammingProjects/DeepFace_Project/database/representations_deepface.pkl'
pkl_file = r'C:\Users\Dirani\ProgrammingProjects\DeepFace_Project\database\frames_info_trial2.pkl'
#pkl_file = '/home/youssef/PythonProjects/AI/DeepFace_Project/database1/frames_info_teta image.pkl'

with open(pkl_file, 'rb') as f:
    pkl_data = pickle.load(f)

print(type(pkl_data), len(pkl_data))

print(pkl_data[8])
"""
i = 0
while i < len(pkl_data):
    print(i, pkl_data[i][0])
    i += 1

"""
person = []
i = -1
for pkl_item in pkl_data:
    i += 1
    for face in pkl_item['detected_faces']:
        name = face['most_similar']['name']
        if name != "": #name == "fp"
            person.append([i, name, pkl_item['frame_index']])

print(person)
