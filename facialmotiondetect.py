import cv2
from fer import FER
import tkinter as tk
from tkinter import messagebox
import threading
emotion_detector = FER()
detection_thread = None
running = False
def initiate_emotion_analysis():
    global running
    video_source = cv2.VideoCapture(0)
    if not video_source.isOpened():
        messagebox.showerror("Error", "Can't open camera")
        return
    running = True
    while running:
        is_frame_captured, video_frame = video_source.read()
        if not is_frame_captured:
            messagebox.showerror("Error", "Could not read video frame.")
            break
        frame_height, frame_width = video_frame.shape[:2]
        target_height = 250
        aspect_ratio = frame_width / frame_height
        target_width = int(target_height * aspect_ratio)
        resized_frame = cv2.resize(video_frame, (target_width, target_height))
        emotion_results = emotion_detector.detect_emotions(resized_frame)
        for detection in emotion_results:
            bounding_box = detection['box']
            emotion_label = detection['emotions']
            cv2.rectangle(resized_frame, 
                          (bounding_box[0], bounding_box[1]), 
                          (bounding_box[0] + bounding_box[2], bounding_box[1] + bounding_box[3]), 
                          (255, 0, 0), 2)
            #
            most_likely_emotion = max(emotion_label, key=emotion_label.get)
            confidence_level = emotion_label[most_likely_emotion]
            cv2.putText(resized_frame, f'{most_likely_emotion}: {confidence_level:.2f}', 
                        (bounding_box[0], bounding_box[1] - 10), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
        cv2.imshow('Varunz Detection Video FRAME', resized_frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    video_source.release()
    cv2.destroyAllWindows()
    running = False
def start_emotion_detection():
    global detection_thread, running
    if not running:
        detection_thread = threading.Thread(target=initiate_emotion_analysis)
        detection_thread.start()
        start_button.config(text="Stop Detection")
    else:
        stop_emotion_detection()
def stop_emotion_detection():
    global running
    running = False
    start_button.config(text="Start Detection")
    if detection_thread:
        detection_thread.join()
def exit_app():
    stop_emotion_detection()
    root.destroy()
def key_press(event):
    if event.char == 's':
        stop_emotion_detection()
root = tk.Tk()
root.title("Vraunz Facial Emotion Detector Program")
root.geometry("490x512")
root.bind('<Key>', key_press)
start_button = tk.Button(root, text="Start Camera & Emotion Detection", command=start_emotion_detection, font=("Helvetica", 14))
start_button.pack(pady=20)
exit_button = tk.Button(root, text="Exit", command=exit_app, font=("Helvetica", 14))
exit_button.pack(pady=20)
root.mainloop()
