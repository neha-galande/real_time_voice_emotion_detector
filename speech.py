import tkinter as tk
from tkinter import Label, Button
from tensorflow.keras.models import model_from_json
from PIL import Image, ImageTk
import numpy as np
import cv2
import librosa
import joblib
import threading
import speech_recognition as sr

# Load MLPClassifier model for audio
mlp_model = joblib.load("C:/Users/Neha/Desktop/SER_MODEL/mlp_model.pkl")

detected_emotion = ""
transcribed_text= ""

def extract_features_realtime(data, sample_rate):
    # Convert data to floating-point format
    
    data_float = librosa.util.fix_length(data.astype(np.float32), size=len(data))


    # ZCR
    zcr = np.mean(librosa.feature.zero_crossing_rate(y=data_float).T, axis=0)

    # Chroma_stft
    stft = np.abs(librosa.stft(data_float))
    chroma_stft = np.mean(librosa.feature.chroma_stft(S=stft, sr=sample_rate).T, axis=0)

    # MFCC
    mfcc = np.mean(librosa.feature.mfcc(y=data_float, sr=sample_rate).T, axis=0)

    # Root Mean Square Value
    rms = np.mean(librosa.feature.rms(y=data_float).T, axis=0)

    # MelSpectrogram
    mel = np.mean(librosa.feature.melspectrogram(y=data_float, sr=sample_rate).T, axis=0)

    result = np.hstack((zcr, chroma_stft, mfcc, rms, mel))
    return result


def get_features_realtime(data, sample_rate):
    # without augmentation
    res1 = extract_features_realtime(data, sample_rate)
    result = np.array(res1)

    return result



def FacialExpressionModel(json_file, weights_file):
    with open(json_file, "r") as file:
        loaded_model_json = file.read()
        model = model_from_json(loaded_model_json)

    model.load_weights(weights_file)
    model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

    return model

def extract_feature_from_audio(audio):
    audio_np = np.frombuffer(audio.frame_data, dtype=np.int16).astype(np.float32) / np.iinfo(np.int16).max

    if not np.isfinite(audio_np).all():
        raise ValueError("Audio buffer contains non-finite values")

    desired_num_features = 20
    extracted_feature = librosa.feature.mfcc(y=audio_np, sr=audio.sample_rate, n_mfcc=desired_num_features)
    return extracted_feature




def detect_voice_emotion(audio):
    global detected_emotion
    feature = extract_feature_from_audio(audio)
    feature = feature.reshape(1, 40, 1)   #(1,40,1)
    predicted_emotion = mlp_model.predict(feature)

    # Ensure that a valid prediction is made
    if predicted_emotion.shape[1] == len(EMOTIONS_LIST):
        detected_emotion_index = np.argmax(predicted_emotion)

        # Check if the detected_emotion_index is within the valid range
        if 0 <= detected_emotion_index < len(EMOTIONS_LIST):
            detected_emotion = EMOTIONS_LIST[detected_emotion_index]
        else:
            detected_emotion = "Unknown"  # Default to "Unknown" if index is out of range
    else:
        detected_emotion = "Unknown"  # Default to "Unknown" if prediction is invalid

    return detected_emotion





def detect_emotion(frame):
    gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = facec.detectMultiScale(gray_frame, 1.3, 5)

    face_emotions = []  # Store predicted emotions for all faces

    try:
        for (x, y, w, h) in faces:
            fc = gray_frame[y:y + h, x:x + w]
            roi = cv2.resize(fc, (48, 48))
            pred = EMOTIONS_LIST[np.argmax(model.predict(roi[np.newaxis, :, :, np.newaxis]))]
            face_emotions.append(pred)  # Append predicted emotion for each face

        # Pad the emotions list if the number of emotions is less than the number of detected faces
        while len(face_emotions) < len(faces):
            face_emotions.append('No face detected')  # Append 'No face detected' for any additional faces

        return face_emotions if face_emotions else ['No face detected'] * len(
            faces)  # Return emotions for all faces or 'No face detected'
    except:
        return ['Unable to detect'] * len(faces)    

def detect_realtime_voice():
    global transcribed_text, detected_emotion
    r = sr.Recognizer()

    with sr.Microphone() as source:
        print("Listening...")
        audio = r.listen(source)
        try:
            transcribed_text = r.recognize_google(audio)
            print("Transcribed text:", transcribed_text)

            # Convert audio data to numpy array
            audio_data = np.frombuffer(audio.frame_data, dtype=np.int16)
            
            # Extract real-time features
            features = get_features_realtime(audio_data, audio.sample_rate)

            # Make predictions using your trained model (mlp_model)
            predicted_emotion = mlp_model.predict(features.reshape(1, -1))
            detected_emotion = predicted_emotion[0]

        except sr.UnknownValueError:
            print("Google Speech Recognition could not understand audio")
            transcribed_text = "Unable to transcribe"
            detected_emotion = "Unknown"
        except sr.RequestError as e:
            print("Could not request results from Google Speech Recognition service; {0}".format(e))
            transcribed_text = "Error in transcription"
            detected_emotion = "Unknown"



def update_interface():
    global detected_emotion, transcribed_text
    
    # If detected_emotion is an array, extract the index of the maximum value
    if isinstance(detected_emotion, np.ndarray):
        emotion_index = np.argmax(detected_emotion)
        # Check if the emotion_index is within the valid range
        if 0 <= emotion_index < len(EMOTIONS_LIST):
            emotion_label = EMOTIONS_LIST[emotion_index]
            label_emotion.config(text="Detected Emotion: " + emotion_label)
        else:
            label_emotion.config(text="Detected Emotion: Unknown")
    else:
        label_emotion.config(text="Detected Emotion: " + detected_emotion)

    label_text.config(text="Transcribed Text: " + transcribed_text)
    label_emotion.after(1000, update_interface)



def start_realtime_detection():
    video_thread = threading.Thread(target=show_webcam)
    video_thread.start()

def start_audio_detection():
    global transcribed_text, detected_emotion
    transcribed_text = ""  # Clear previous transcription
    detected_emotion = ""  # Clear previous detected emotion
    label_emotion.config(text="Detected Emotion: " + detected_emotion)  # Update GUI
    label_text.config(text="Transcribed Text: " + transcribed_text)  # Update GUI
    top.update()  # Force an immediate GUI update
    audio_thread = threading.Thread(target=detect_realtime_voice)
    audio_thread.start()


def show_webcam():
    face_cascade = cv2.CascadeClassifier('C:/Users/Neha/Desktop/TESS/haarcascade_frontalface_default.xml')
    cap = cv2.VideoCapture(0)

    while True:
        ret, frame = cap.read()
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        predicted_emotions = detect_emotion(frame)

        faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))

        for idx, (x, y, w, h) in enumerate(faces):
            cv2.rectangle(frame, (x, y), (x + w, y + h), (255, 0, 0), 2)

            if idx < len(predicted_emotions):
                emotion_text = predicted_emotions[idx]
                cv2.putText(frame, emotion_text, (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (36, 255, 12), 2)

        cv2.imshow('Video', frame)

        key = cv2.waitKey(1)
        if key == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

# GUI setup
top = tk.Tk()
top.geometry('800x600')
top.title('Emotion Detector')
top.configure(background='#CDCDCD')
label1 = Label(top, background='#CDCDCD', font=('arial', 15, 'bold'))
sign_image = Label(top)

facec = cv2.CascadeClassifier('C:/Users/Neha/Desktop/TESS/haarcascade_frontalface_default.xml')
model = FacialExpressionModel("model_a.json", "model_weights.h5")

EMOTIONS_LIST = ["angry","disgust","fear","happy","neutral","ps","sad"]

label_emotion = Label(top, text="Detected Emotion: ", font=('arial', 12))
label_emotion.pack()

label_text = Label(top, text="Transcribed Text: ", font=('arial', 12))
label_text.pack()

realtime_btn = Button(top, text="Start Real-Time Detection", command=start_realtime_detection, padx=10, pady=5)
realtime_btn.configure(background="#364156", foreground='white', font=('arial', 12, 'bold'))
realtime_btn.pack(side='bottom', pady=20)

audio_btn = Button(top, text="Start Audio Detection", command=start_audio_detection, padx=10, pady=5)
audio_btn.configure(background="#364156", foreground='white', font=('arial', 12, 'bold'))
audio_btn.pack(side='bottom', pady=20)

sign_image.pack(side='bottom', expand='True')
label1.pack(side='bottom', expand='True')
heading = Label(top, text='Emotion Detector', pady=20, font=('arial', 25, 'bold'))
heading.configure(background='#CDCDCD', foreground="#364156")
heading.pack()

top.after(1000, update_interface)  # Update GUI every second
top.mainloop()
