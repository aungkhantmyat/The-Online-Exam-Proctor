import sys
import face_recognition
from concurrent.futures import ThreadPoolExecutor
import cv2
import mediapipe as mp
import numpy as np
import time
import math
import random
import os
import json
import shutil
import keyboard
import pyautogui
import pygetwindow as gw
import webbrowser
import pyperclip
from ultralytics import YOLO
import threading
from multiprocessing import Process
import pyaudio
import struct
import wave
import datetime
import subprocess

#Variables
#All Related
Globalflag = False
Student_Name = ''
start_time = [0, 0, 0, 0, 0]
end_time = [0, 0, 0, 0, 0]
recorded_durations = []
prev_state = ['Verified Student appeared', "Forward", "Only one person is detected", "Stay in the Test", "No Electronic Device Detected"]
flag = [False, False, False, False, False]
capb= cv2.VideoCapture(0)
width= int(capb.get(cv2.CAP_PROP_FRAME_WIDTH))
height= int(capb.get(cv2.CAP_PROP_FRAME_HEIGHT))
capb.release()
capa = cv2.VideoCapture("test_V.mp4")
EDWidth=int(capa.get(cv2.CAP_PROP_FRAME_WIDTH))
EDHeight=int(capa.get(cv2.CAP_PROP_FRAME_HEIGHT))
capa.release()
video = [(str(random.randint(1,50000))+".mp4"), (str(random.randint(1,50000))+".mp4"), (str(random.randint(1,50000))+".mp4"), (str(random.randint(1,50000))+".mp4"), (str(random.randint(1,50000))+".mp4")]
writer = [cv2.VideoWriter(video[0], cv2.VideoWriter_fourcc(*'mp4v'), 20, (width,height)), cv2.VideoWriter(video[1], cv2.VideoWriter_fourcc(*'mp4v'), 20, (width,height)), cv2.VideoWriter(video[2], cv2.VideoWriter_fourcc(*'mp4v'), 20, (width,height)), cv2.VideoWriter(video[3], cv2.VideoWriter_fourcc(*'mp4v'), 15, (1920, 1080)), cv2.VideoWriter(video[4], cv2.VideoWriter_fourcc(*'mp4v'), 20 , (EDWidth,EDHeight))]
#More than One Person Related
mpFaceDetection = mp.solutions.face_detection  # Detect the face
mpDraw = mp.solutions.drawing_utils  # Draw the required Things for BBox
faceDetection = mpFaceDetection.FaceDetection(0.75)# It has 0 to 1 (Change this to make it more detectable) Default is 0.5 and higher means more detection.
#Screen Related
shorcuts = []
active_window = None # Store the initial active window and its title
active_window_title = "Exam — Mozilla Firefox"
exam_window_title = active_window_title
#ED Related
my_file = open("utils/coco.txt", "r") # opening the file in read mode
data = my_file.read() # reading the file
class_list = data.split("\n") # replacing end splitting the text | when newline ('\n') is seen.
my_file.close()
detected_things = []
detection_colors = [] # Generate random colors for class list
for i in range(len(class_list)):
    r = random.randint(0, 255)
    g = random.randint(0, 255)
    b = random.randint(0, 255)
    detection_colors.append((b, g, r))
model = YOLO("yolov8n.pt", "v8") # load a pretrained YOLOv8n model
EDFlag = False
#Voice Related
TRIGGER_RMS = 10  # start recording above 10
RATE = 16000  # sample rate
TIMEOUT_SECS = 3  # silence time after which recording stops
FRAME_SECS = 0.25  # length of frame(chunks) to be processed at once in secs
CUSHION_SECS = 1  # amount of recording before and after sound
SHORT_NORMALIZE = (1.0 / 32768.0)
FORMAT = pyaudio.paInt16
CHANNELS = 1
SHORT_WIDTH = 2
CHUNK = int(RATE * FRAME_SECS)
CUSHION_FRAMES = int(CUSHION_SECS / FRAME_SECS)
TIMEOUT_FRAMES = int(TIMEOUT_SECS / FRAME_SECS)
f_name_directory = 'C:/Users/kaungmyat/PycharmProjects/BestOnlineExamProctor/static/OuputAudios'
# Capture
cap = None


#Database and Files Related
# function to add data to JSON
def write_json(new_data, filename='violation.json'):
    with open(filename,'r+') as file:
        # First we load existing data into a dict.
        file_data = json.load(file)
        # Join new_data with file_data inside emp_details
        file_data.append(new_data)
        # Sets file's current position at offset.
        file.seek(0)
        # convert back to json.
        json.dump(file_data, file, indent = 4)

#Function to move the files to the Output Folders
def move_file_to_output_folder(file_name,folder_name='OutputVideos'):
    # Get the current working directory (project folder)
    current_directory = os.getcwd()
    # Define the paths for the source file and destination folder
    source_path = os.path.join(current_directory, file_name)
    destination_path = os.path.join(current_directory, 'static', folder_name, file_name)
    try:
        # Use 'shutil.move' to move the file to the destination folder
        shutil.move(source_path, destination_path)
        print('Your video is moved to'+folder_name)
    except FileNotFoundError:
        print(f"Error: File '{file_name}' not found in the project folder.")
    except shutil.Error as e:
        print(f"Error: Failed to move the file. {e}")

#Function to reduce video file's data rate to 100 kbps
def reduceBitRate (input_file,output_file):
   target_bitrate = "1000k"  # Set your desired target bitrate here
   # Specify the full path to the FFmpeg executable
   ffmpeg_path = "C:/Users/kaungmyat/Downloads/ffmpeg-2023-08-28-git-b5273c619d-essentials_build/ffmpeg-2023-08-28-git-b5273c619d-essentials_build/bin/ffmpeg.exe"  # Replace with the actual path to ffmpeg.exe on your system
   # Run FFmpeg command to lower the bitrate
   command = [
      ffmpeg_path,
      "-i", input_file,
      "-b:v", target_bitrate,
      "-c:v", "libx264",
      "-c:a", "aac",
      "-strict", "experimental",
      "-b:a", "192k",
      output_file
   ]
   subprocess.run(command, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
   print("Bitrate conversion completed.")

#Recordings related
#Recording Function for Face Verification
def faceDetectionRecording(img, text):
    global start_time, end_time, recorded_durations, prev_state, flag, writer, width, height
    print("Running FaceDetection Recording Function")
    print(text)
    if text != 'Verified Student appeared' and prev_state[0] == 'Verified Student appeared':
        start_time[0] = time.time()
        for _ in range(2):
            writer[0].write(img)
    elif text != 'Verified Student appeared' and str(text) == prev_state[0] and (time.time() - start_time[0]) > 3:
        flag[0] = True
        for _ in range(2):
            writer[0].write(img)
    elif text != 'Verified Student appeared' and str(text) == prev_state[0] and (time.time() - start_time[0]) <= 3:
        flag[0] = False
        for _ in range(2):
            writer[0].write(img)
    else:
        if prev_state[0] != "Verified Student appeared":
            writer[0].release()
            end_time[0] = time.time()
            duration = math.ceil((end_time[0] - start_time[0]) / 3)
            outputVideo = 'FDViolation' + video[0]
            FDViolation = {
                "Name": prev_state[0],
                "Time": time.strftime("%Y-%m-%d %H:%M:%S", time.localtime(start_time[0])),
                "Duration": str(duration) + " seconds",
                "Mark": math.floor(2 * duration),
                "Link": outputVideo,
                "RId": get_resultId()
            }
            if flag[0]:
                recorded_durations.append(FDViolation)
                write_json(FDViolation)
                reduceBitRate(video[0], outputVideo)
                move_file_to_output_folder(outputVideo)
            os.remove(video[0])
            print(recorded_durations)
            video[0] = str(random.randint(1, 50000)) + ".mp4"
            writer[0] = cv2.VideoWriter(video[0], cv2.VideoWriter_fourcc(*'mp4v'), 20, (width, height))
            flag[0] = False
    prev_state[0] = text

#Recording Function for Head Movement Detection
def Head_record_duration(text,img):
    global start_time, end_time, recorded_durations, prev_state, flag,writer, width, height
    print("Running HeadMovement Recording Function")
    print(text)
    if text != "Forward":
        if str(text) != prev_state[1] and prev_state[1] == "Forward":
            start_time[1] = time.time()
            for _ in range(2):
                writer[1].write(img)
        elif str(text) != prev_state[1] and prev_state[1] != "Forward":
            writer[1].release()
            end_time[1] = time.time()
            duration = math.ceil((end_time[1] - start_time[1])/7)
            outputVideo = 'HeadViolation' + video[1]
            HeadViolation = {
                "Name": prev_state[1],
                "Time": time.strftime("%Y-%m-%d %H:%M:%S", time.localtime(start_time[1])),
                "Duration": str(duration) + " seconds",
                "Mark": duration,
                "Link": outputVideo,
                "RId": get_resultId()
            }
            if flag[1]:
                recorded_durations.append(HeadViolation)
                write_json(HeadViolation)
                reduceBitRate(video[1], outputVideo)
                move_file_to_output_folder(outputVideo)
            os.remove(video[1])
            print(recorded_durations)
            start_time[1] = time.time()
            video[1] = str(random.randint(1, 50000)) + ".mp4"
            writer[1] = cv2.VideoWriter(video[1], cv2.VideoWriter_fourcc(*'mp4v'), 20, (width,height))
            flag[1] = False
        elif str(text) == prev_state[1] and (time.time() - start_time[1]) > 3:
            flag[1] = True
            for _ in range(2):
                writer[1].write(img)
        elif str(text) == prev_state[1] and (time.time() - start_time[1]) <= 3:
            flag[1] = False
            for _ in range(2):
                writer[1].write(img)
        prev_state[1] = text
    else:
        if prev_state[1] != "Forward":
            writer[1].release()
            end_time[1] = time.time()
            duration = math.ceil((end_time[1] - start_time[1])/7)
            outputVideo = 'HeadViolation' + video[1]
            HeadViolation = {
                "Name": prev_state[1],
                "Time": time.strftime("%Y-%m-%d %H:%M:%S", time.localtime(start_time[1])),
                "Duration": str(duration) + " seconds",
                "Mark": duration,
                "Link": outputVideo,
                "RId": get_resultId()
            }
            if flag[1]:
                recorded_durations.append(HeadViolation)
                write_json(HeadViolation)
                reduceBitRate(video[1], outputVideo)
                move_file_to_output_folder(outputVideo)
            os.remove(video[1])
            print(recorded_durations)
            video[1] = str(random.randint(1, 50000)) + ".mp4"
            writer[1] = cv2.VideoWriter(video[1], cv2.VideoWriter_fourcc(*'mp4v'), 20, (width,height))
            flag[1] = False
        prev_state[1] = text

#Recording Function for More than one person Detection
def MTOP_record_duration(text, img):
    global start_time, end_time, recorded_durations, prev_state, flag, writer, width, height
    print("Running MTOP Recording Function")
    print(text)
    if text != 'Only one person is detected' and prev_state[2] == 'Only one person is detected':
        start_time[2] = time.time()
        for _ in range(2):
            writer[2].write(img)
    elif text != 'Only one person is detected' and str(text) == prev_state[2] and (time.time() - start_time[2]) > 3:
        flag[2] = True
        for _ in range(2):
            writer[2].write(img)
    elif text != 'Only one person is detected' and str(text) == prev_state[2] and (time.time() - start_time[2]) <= 3:
        flag[2] = False
        for _ in range(2):
            writer[2].write(img)
    else:
        if prev_state[2] != "Only one person is detected":
            writer[2].release()
            end_time[2] = time.time()
            duration = math.ceil((end_time[2] - start_time[2])/3)
            outputVideo = 'MTOPViolation' + video[2]
            MTOPViolation = {
                "Name": prev_state[2],
                "Time": time.strftime("%Y-%m-%d %H:%M:%S", time.localtime(start_time[2])),
                "Duration": str(duration) + " seconds",
                "Mark": math.floor(1.5 * duration),
                "Link": outputVideo,
                "RId": get_resultId()
            }
            if flag[2]:
                recorded_durations.append(MTOPViolation)
                write_json(MTOPViolation)
                reduceBitRate(video[2], outputVideo)
                move_file_to_output_folder(outputVideo)
            os.remove(video[2])
            print(recorded_durations)
            video[2] = str(random.randint(1, 50000)) + ".mp4"
            writer[2] = cv2.VideoWriter(video[2], cv2.VideoWriter_fourcc(*'mp4v'), 20, (width,height))
            flag[2] = False
    prev_state[2] = text

#Recording Function for Screen Detection
def SD_record_duration(text, img):
    global start_time, end_time, prev_state, flag, writer, width, height
    print("Running SD Recording Function")
    print(text)
    if text != "Stay in the Test" and prev_state[3] == "Stay in the Test":
        start_time[3] = time.time()
        print(f"Start SD Recording, start time is {start_time[3]} and array is {start_time}")
        for _ in range(2):
            writer[3].write(img)
    elif text != "Stay in the Test" and str(text) == prev_state[3] and (time.time() - start_time[3]) > 3:
        flag[3] = True
        for _ in range(2):
            writer[3].write(img)
    elif text != "Stay in the Test" and str(text) == prev_state[3] and (time.time() - start_time[3]) <= 3:
        flag[3] = False
        for _ in range(2):
            writer[3].write(img)
    else:
        if prev_state[3] != "Stay in the Test":
            writer[3].release()
            end_time[3] = time.time()
            duration = math.ceil((end_time[3] - start_time[3])/4)
            outputVideo = 'SDViolation' + video[3]
            SDViolation = {
                "Name": prev_state[3],
                "Time": time.strftime("%Y-%m-%d %H:%M:%S", time.localtime(start_time[3])),
                "Duration": str(duration) + " seconds",
                "Mark": (2 * duration),
                "Link": outputVideo,
                "RId": get_resultId()
            }
            if flag[3]:
                recorded_durations.append(SDViolation)
                write_json(SDViolation)
                reduceBitRate(video[3], outputVideo)
                move_file_to_output_folder(outputVideo)
            os.remove(video[3])
            print(recorded_durations)
            video[3] = str(random.randint(1, 50000)) + ".mp4"
            writer[3] = cv2.VideoWriter(video[3], cv2.VideoWriter_fourcc(*'mp4v'), 15, (1920, 1080))
            flag[3] = False
    prev_state[3] = text

# Function to capture the screen using PyAutoGUI and return the frame as a NumPy array
def capture_screen():
    screenshot = pyautogui.screenshot()
    frame = np.array(screenshot)
    frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
    return frame

#Recording Function for Electronic Devices Detection
def EDD_record_duration(text, img):
    global start_time, end_time, prev_state, flag, writer,recorded_Images,EDD_Duration, video, EDWidth, EDHeight
    print(text)
    if text == "Electronic Device Detected" and prev_state[4] == "No Electronic Device Detected":
        start_time[4] = time.time()
        for _ in range(2):
            writer[4].write(img)
    elif text == "Electronic Device Detected" and str(text) == prev_state[4] and (time.time() - start_time[4]) > 0:
        flag[4] = True
        for _ in range(2):
            writer[4].write(img)
    elif text == "Electronic Device Detected" and str(text) == prev_state[4] and (time.time() - start_time[4]) <= 0:
        flag[4] = False
        for _ in range(2):
            writer[4].write(img)
    else:
        if prev_state[4] == "Electronic Device Detected":
            writer[4].release()
            end_time[4] = time.time()
            duration = math.ceil((end_time[4] - start_time[4])/10)
            outputVideo = 'EDViolation' + video[4]
            EDViolation = {
                "Name": prev_state[4],
                "Time": time.strftime("%Y-%m-%d %H:%M:%S", time.localtime(start_time[4])),
                "Duration": str(duration) + " seconds",
                "Mark": math.floor(1.5 * duration),
                "Link": outputVideo,
                "RId": get_resultId()
            }
            if flag[4]:
                write_json(EDViolation)
                reduceBitRate(video[4], outputVideo)
                move_file_to_output_folder(outputVideo)
            os.remove(video[4])
            video[4]= str(random.randint(1, 50000)) + ".mp4"
            writer[4] = cv2.VideoWriter(video[4], cv2.VideoWriter_fourcc(*'mp4v'), 10 , (EDWidth,EDHeight))
            flag[4] = False
    prev_state[4] = text

#system Related
def deleteTrashVideos():
    global video
    video_folder = 'C:/Users/kaungmyat/PycharmProjects/BestOnlineExamProctor'
    # Iterate through files in the folder
    for filename in os.listdir(video_folder):
        if filename.lower().endswith('.mp4'):
            try:
                os.remove(filename)
            except OSError:
                pass

#Models Related
#One: Face Detection Function
def face_confidence(face_distance, face_match_threshold=0.6):
    range = (1.0 - face_match_threshold)
    linear_val = (1.0 - face_distance) / (range * 2.0)

    if face_distance > face_match_threshold:
        return str(round(linear_val * 100, 2)) + '%'
    else:
        value = (linear_val + ((1.0 - linear_val) * math.pow((linear_val - 0.5) * 2, 0.2))) * 100
        return str(round(value, 2)) + '%'

class FaceRecognition:
    face_locations = []
    face_encodings = []
    face_names = []
    known_face_encodings = []
    known_face_names = []
    process_current_frame = True

    def __init__(self):
        self.encode_faces()

    def encode_faces(self):
        for image in os.listdir('static/Profiles'):
            face_image = face_recognition.load_image_file(f"static/Profiles/{image}")
            face_encoding = face_recognition.face_encodings(face_image)[0]

            self.known_face_encodings.append(face_encoding)
            self.known_face_names.append(image)
        print(self.known_face_names)

    def run_recognition(self):
        global Globalflag
        #video_capture = cv2.VideoCapture(0)
        print(f'Face Detection Flag is {Globalflag}')
        text = ""
        if not cap.isOpened():
            sys.exit('Video source not found...')

        while Globalflag:
            ret, frame = cap.read()
            text = "Verified Student disappeared"
            print("Running Face Verification Function")
            # Only process every other frame of video to save time
            if self.process_current_frame:
                # Resize frame of video to 1/4 size for faster face recognition processing
                small_frame = cv2.resize(frame, (0, 0), fx=0.25, fy=0.25)

                # Convert the image from BGR color (which OpenCV uses) to RGB color (which face_recognition uses)
                #rgb_small_frame = small_frame[:, :, ::-1]
                rgb_small_frame = np.ascontiguousarray(small_frame[:, :, ::-1])

                # Find all the faces and face encodings in the current frame of video
                self.face_locations = face_recognition.face_locations(rgb_small_frame)
                self.face_encodings = face_recognition.face_encodings(rgb_small_frame, self.face_locations)

                self.face_names = []
                for face_encoding in self.face_encodings:
                    # See if the face is a match for the known face(s)
                    matches = face_recognition.compare_faces(self.known_face_encodings, face_encoding)
                    name = "Unknown"
                    confidence = '???'

                    # Calculate the shortest distance to face
                    face_distances = face_recognition.face_distance(self.known_face_encodings, face_encoding)

                    best_match_index = np.argmin(face_distances)
                    if matches[best_match_index]:
                        tempname = str(self.known_face_names[best_match_index]).split('_')[0]
                        tempconfidence = face_confidence(face_distances[best_match_index])
                        if tempname == Student_Name and float(tempconfidence[:-1]) >= 84:
                            name = tempname
                            confidence = tempconfidence

                    self.face_names.append(f'{name} ({confidence})')

            self.process_current_frame = not self.process_current_frame

            # Display the results
            for (top, right, bottom, left), name in zip(self.face_locations, self.face_names):
                # Scale back up face locations since the frame we detected in was scaled to 1/4 size
                top *= 4
                right *= 4
                bottom *= 4
                left *= 4
                if "Unknown" not in name:
                    # Create the frame with the name
                    text = "Verified Student appeared"
                cv2.rectangle(frame, (left, top), (right, bottom), (0, 0, 255), 2)
                cv2.rectangle(frame, (left, bottom - 35), (right, bottom), (0, 0, 255), cv2.FILLED)
                cv2.putText(frame, name, (left + 6, bottom - 6), cv2.FONT_HERSHEY_DUPLEX, 0.8, (255, 255, 255), 1)

            # Display the resulting image
           # cv2.imshow('Face Recognition', frame)
            print(text)
            faceDetectionRecording(frame, text)
            # Hit 'q' on the keyboard to quit!

#Second: Head Movement Detection Function
def headMovmentDetection(image, face_mesh):
    print("Running HeadMovement Function")
    # Flip the image horizontally for a later selfie-view display
    # Also convert the color space from BGR to RGB
    image = cv2.cvtColor(cv2.flip(image, 1), cv2.COLOR_BGR2RGB)

    # To improve performance
    image.flags.writeable = False

    # Get the result
    results = face_mesh.process(image)

    # To improve performance
    image.flags.writeable = True

    # Convert the color space from RGB to BGR
    image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)

    img_h, img_w, img_c = image.shape
    face_3d = []
    face_2d = []

    if results.multi_face_landmarks:
        for face_landmarks in results.multi_face_landmarks:
            for idx, lm in enumerate(face_landmarks.landmark):
                if idx == 33 or idx == 263 or idx == 1 or idx == 61 or idx == 291 or idx == 199:
                    if idx == 1:
                        nose_2d = (lm.x * img_w, lm.y * img_h)
                        nose_3d = (lm.x * img_w, lm.y * img_h, lm.z * 8000)

                    x, y = int(lm.x * img_w), int(lm.y * img_h)

                    # Get the 2D Coordinates
                    face_2d.append([x, y])

                    # Get the 3D Coordinates
                    face_3d.append([x, y, lm.z])

                    # Convert it to the NumPy array
            face_2d = np.array(face_2d, dtype=np.float64)

            # Convert it to the NumPy array
            face_3d = np.array(face_3d, dtype=np.float64)

            # The camera matrix
            focal_length = 1 * img_w

            cam_matrix = np.array([[focal_length, 0, img_h / 2],
                                   [0, focal_length, img_w / 2],
                                   [0, 0, 1]])

            # The Distance Matrix
            dist_matrix = np.zeros((4, 1), dtype=np.float64)

            # Solve PnP
            success, rot_vec, trans_vec = cv2.solvePnP(face_3d, face_2d, cam_matrix, dist_matrix)

            # Get rotational matrix
            rmat, jac = cv2.Rodrigues(rot_vec)

            # Get angles
            angles, mtxR, mtxQ, Qx, Qy, Qz = cv2.RQDecomp3x3(rmat)

            # Get the y rotation degree
            x = angles[0] * 360
            y = angles[1] * 360
            # print(y)
            textHead = ''
            # See where the user's head tilting
            if y < -10:
                textHead = "Looking Left"
            elif y > 15:
                textHead = "Looking Right"
            elif x < -8:
                textHead = "Looking Down"
            elif x > 15:
                textHead = "Looking Up"
            else:
                textHead = "Forward"
            # Add the text on the image
            cv2.putText(image, textHead, (20, 20), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
            Head_record_duration(textHead, image)


#Third : More than one person Detection Function
def MTOP_Detection(img):
    print("Running MTOP Function")
    imgRGB = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    results = faceDetection.process(imgRGB)
    textMTOP = ''
    if results.detections:
        for id, detection in enumerate(results.detections):
            bboxC = detection.location_data.relative_bounding_box
            ih, iw, ic = img.shape
            bbox = int(bboxC.xmin * iw), int(bboxC.ymin * ih), \
                int(bboxC.width * iw), int(bboxC.height * ih)
            # Drawing the recantangle
            cv2.rectangle(img, bbox, (255, 0, 255), 2)
            # cv2.putText(img, f'{int(detection.score[0] * 100)}%', (bbox[0], bbox[1] - 20), cv2.FONT_HERSHEY_PLAIN, 3, (255, 255, 255), 10)
        if id > 0:
            textMTOP = "More than one person is detected."
        else:
            textMTOP = "Only one person is detected"
    else:
        textMTOP="Only one person is detected"
    MTOP_record_duration(textMTOP, img)
    print(textMTOP)

#Fourth : Screen Detection Function ( Key-words and Screens)
def shortcut_handler(event):
    if event.event_type == keyboard.KEY_DOWN:
        shortcut = ''
        # Check for Ctrl+C
        if keyboard.is_pressed('ctrl') and keyboard.is_pressed('c'):
            shortcut += 'Ctrl+C'
            print("Ctrl+C shortcut detected!")
        # Check for Ctrl+V
        elif keyboard.is_pressed('ctrl') and keyboard.is_pressed('v'):
            shortcut += 'Ctrl+V'
            print("Ctrl+V shortcut detected!")
        # Check for Ctrl+A
        elif keyboard.is_pressed('ctrl') and keyboard.is_pressed('a'):
            shortcut += 'Ctrl+A'
            print("Ctrl+A shortcut detected!")
        # Check for Ctrl+X
        elif keyboard.is_pressed('ctrl') and keyboard.is_pressed('x'):
            shortcut += 'Ctrl+X'
            print("Ctrl+X shortcut detected!")
        # Check for Alt+Shift+Tab
        elif keyboard.is_pressed('alt') and keyboard.is_pressed('shift') and keyboard.is_pressed('tab'):
            shortcut += 'Alt+Shift+Tab'
            print("Alt+Shift+Tab shortcut detected!")
        # Check for Win+Tab
        elif keyboard.is_pressed('win') and keyboard.is_pressed('tab'):
            shortcut += 'Win+Tab'
            print("Win+Tab shortcut detected!")
        # Check for Alt+Esc
        elif keyboard.is_pressed('alt') and keyboard.is_pressed('esc'):
            shortcut += 'Alt+Esc'
            print("Alt+Esc shortcut detected!")
        # Check for Alt+Tab
        elif keyboard.is_pressed('alt') and keyboard.is_pressed('tab'):
            shortcut += 'Alt+Tab'
            print("Alt+Tab shortcut detected!")
        # Check for Ctrl+Esc
        elif keyboard.is_pressed('ctrl') and keyboard.is_pressed('esc'):
            shortcut += 'Ctrl+Esc'
            print("Ctrl+Esc shortcut detected!")
        # Check for Function Keys F1
        elif keyboard.is_pressed('f1'):
            shortcut += 'F1'
            print("F1 shortcut detected")
        # Check for Function Keys F2
        elif keyboard.is_pressed('f2'):
            shortcut += 'F2'
            print("F2 shortcut detected!")
        # Check for Function Keys F3
        elif keyboard.is_pressed('f3'):
            shortcut += 'F3'
            print("F3 shortcut detected!")
        # Check for Window Key
        elif keyboard.is_pressed('win'):
            shortcut += 'Window'
            print("Window shortcut detected!")
        # Check for Ctrl+Alt+Del
        elif keyboard.is_pressed('ctrl') and keyboard.is_pressed('alt') and keyboard.is_pressed('del'):
            shortcut += 'Ctrl+Alt+Del'
            print("Ctrl+Alt+Del shortcut detected!")
        # Check for Prt Scn
        elif keyboard.is_pressed('print_screen'):
            shortcut += 'Prt Scn'
            print("Prt Scn shortcut detected!")
        # Check for Ctrl+T
        elif keyboard.is_pressed('ctrl') and keyboard.is_pressed('t'):
            shortcut += 'Ctrl+T'
            print("Ctrl+T shortcut detected!")
        # Check for Ctrl+W
        elif keyboard.is_pressed('ctrl') and keyboard.is_pressed('w'):
            shortcut += 'Ctrl+W'
            print("Ctrl+W shortcut detected!")
        # Check for Ctrl+Z
        elif keyboard.is_pressed('ctrl') and keyboard.is_pressed('z'):
            shortcut += 'Ctrl+Z'
            print("Ctrl+Z shortcut detected!")
        shorcuts.append(shortcut) if shortcut != "" else None

def screenDetection():
    global active_window, active_window_title, exam_window_title
    textScreen = ""
    # Get the current active window
    new_active_window = gw.getActiveWindow()
    frame = capture_screen()

    # Check if the active window has changed
    if new_active_window is not None and new_active_window.title != exam_window_title:
        # Check if the active window is a browser or a tab
        if new_active_window.title != active_window_title:
            print("Moved to Another Window: ", new_active_window.title)
            # Update the active window and its title
            active_window = new_active_window
            active_window_title = active_window.title
        textScreen = "Move away from the Test"
    else:
        if new_active_window is not None:
            textScreen = "Stay in the Test"
    SD_record_duration(textScreen, frame)
    print(textScreen)

#Fifth : Electronic Devices Detection Function
def electronicDevicesDetection(frame):
    global model, EDFlag
    # Predict on image
    detect_params = model.predict(source=[frame], conf=0.45, save=False)
    # Convert tensor array to numpy
    DP = detect_params[0].numpy()
    for result in detect_params:  # iterate results
        boxes = result.boxes.cpu().numpy()  # get boxes on cpu in numpy
        for box in boxes:  # iterate boxes
            r = box.xyxy[0].astype(int)  # get corner points as int
            detected_obj = result.names[int(box.cls[0])]
            if (detected_obj == 'cell phone' or detected_obj == 'remote' or detected_obj == 'laptop' or detected_obj == 'laptop,book'): EDFlag = True
    textED = ''
    # Display the resulting frame
    if EDFlag:
        textED = 'Electronic Device Detected'
    else:
        textED = "No Electronic Device Detected"
    EDD_record_duration(textED, frame)
    print(textED)
    EDFlag = False

#Sixth Function : Voice Detection
class Recorder:
    @staticmethod
    def rms(frame):
        count = len(frame) / SHORT_WIDTH
        format = "%dh" % (count)
        shorts = struct.unpack(format, frame)

        sum_squares = 0.0
        for sample in shorts:
            n = sample * SHORT_NORMALIZE
            sum_squares += n * n
        rms = math.pow(sum_squares / count, 0.5)

        return rms * 1000

    def __init__(self):
        self.p = pyaudio.PyAudio()
        self.stream = self.p.open(format=FORMAT,
                                  channels=CHANNELS,
                                  rate=RATE,
                                  input=True,
                                  output=True,
                                  frames_per_buffer=CHUNK)
        self.time = time.time()
        self.quiet = []
        self.quiet_idx = -1
        self.timeout = 0

    def record(self):
        global Globalflag
        print('')
        print(f'Voice Flag is {Globalflag}')
        sound = []
        start = time.time()
        begin_time = None
        while Globalflag:
            data = self.stream.read(CHUNK)
            rms_val = self.rms(data)
            if self.inSound(data):
                sound.append(data)
                if begin_time == None:
                    begin_time = datetime.datetime.now()
            else:
                if len(sound) > 0:
                    duration=math.floor((datetime.datetime.now()-begin_time).total_seconds())
                    self.write(sound, begin_time, duration)
                    sound.clear()
                    begin_time = None
                else:
                    self.queueQuiet(data)

            curr = time.time()
            secs = int(curr - start)
            tout = 0 if self.timeout == 0 else int(self.timeout - curr)
            label = 'Listening' if self.timeout == 0 else 'Recording'
            print('[+] %s: Level=[%4.2f] Secs=[%d] Timeout=[%d]' % (label, rms_val, secs, tout), end='\r')

    # quiet is a circular buffer of size cushion
    def queueQuiet(self, data):
        self.quiet_idx += 1
        # start over again on overflow
        if self.quiet_idx == CUSHION_FRAMES:
            self.quiet_idx = 0

        # fill up the queue
        if len(self.quiet) < CUSHION_FRAMES:
            self.quiet.append(data)
        # replace the element on the index in a cicular loop like this 0 -> 1 -> 2 -> 3 -> 0 and so on...
        else:
            self.quiet[self.quiet_idx] = data

    def dequeueQuiet(self, sound):
        if len(self.quiet) == 0:
            return sound

        ret = []

        if len(self.quiet) < CUSHION_FRAMES:
            ret.append(self.quiet)
            ret.extend(sound)
        else:
            ret.extend(self.quiet[self.quiet_idx + 1:])
            ret.extend(self.quiet[:self.quiet_idx + 1])
            ret.extend(sound)

        return ret

    def inSound(self, data):
        rms = self.rms(data)
        curr = time.time()

        if rms > TRIGGER_RMS:
            self.timeout = curr + TIMEOUT_SECS
            return True

        if curr < self.timeout:
            return True

        self.timeout = 0
        return False

    def write(self, sound, begin_time, duration):
        # insert the pre-sound quiet frames into sound
        sound = self.dequeueQuiet(sound)

        # sound ends with TIMEOUT_FRAMES of quiet
        # remove all but CUSHION_FRAMES
        keep_frames = len(sound) - TIMEOUT_FRAMES + CUSHION_FRAMES
        recording = b''.join(sound[0:keep_frames])
        filename = str(random.randint(1,50000))+"VoiceViolation"
        pathname = os.path.join(f_name_directory, '{}.wav'.format(filename))
        wf = wave.open(pathname, 'wb')
        wf.setnchannels(CHANNELS)
        wf.setsampwidth(self.p.get_sample_size(FORMAT))
        wf.setframerate(RATE)
        wf.writeframes(recording)
        wf.close()
        voiceViolation = {
            "Name": "Common Noise is detected.",
            "Time": begin_time.strftime("%Y-%m-%d %H:%M:%S"),
            "Duration": str(duration) + " seconds",
            "Mark": duration,
            "Link": '{}.wav'.format(filename),
            "RId": get_resultId()
        }
        write_json(voiceViolation)
        print('[+] Saved: {}'.format(pathname))

def cheat_Detection1():
    deleteTrashVideos()
    global Globalflag
    mp_face_mesh = mp.solutions.face_mesh
    face_mesh = mp_face_mesh.FaceMesh(min_detection_confidence=0.5, min_tracking_confidence=0.5)
    print(f'CD1 Flag is {Globalflag}')
    while Globalflag:
        success, image = cap.read()
        headMovmentDetection(image, face_mesh)
    if Globalflag:
        cap.release()
    deleteTrashVideos()

def cheat_Detection2():
    global Globalflag, shorcuts
    print(f'CD2 Flag is {Globalflag}')

    deleteTrashVideos()
    while Globalflag:
        success, image = cap.read()
        image1 = image
        image2 = image
        MTOP_Detection(image1)
        screenDetection()
    deleteTrashVideos()
    if Globalflag:
        cap.release()

#Query Related
#Function to give the next resut id
def get_resultId():
    with open('result.json','r+') as file:
        # First we load existing data into a dict.
        file_data = json.load(file)
        #sort json by ID
        file_data.sort(key=lambda x: x["Id"])
        return file_data[-1]['Id']+1

#Function to give the trust score
def get_TrustScore(Rid):
    with open('violation.json', 'r+') as file:
        # First we load existing data into a dict.
        file_data = json.load(file)
        filtered_data = [item for item in file_data if item["RId"] == Rid]
        total_mark = sum(item["Mark"] for item in filtered_data)
        return total_mark

#Function to give all results
def getResults():
    with open('result.json', 'r+') as file:
        # First we load existing data into a dict.
        result_data = json.load(file)
        return result_data

#Function to give result details
def getResultDetails(rid):
    with open('result.json', 'r+') as file:
        # First we load existing data into a dict.
        result_data = json.load(file)
        filtered_result = [item for item in result_data if item["Id"] == int(rid)]
    with open('violation.json', 'r+') as file:
        # First we load existing data into a dict.
        violation_data = json.load(file)
        filtered_violations = [item for item in violation_data if item["RId"] == int(rid)]
    resultDetails = {
            "Result": filtered_result,
            "Violation": filtered_violations
        }
    return resultDetails

a = Recorder()
fr = FaceRecognition()
