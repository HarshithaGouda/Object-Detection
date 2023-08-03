import cv2
from tkinter import *
from PIL import Image, ImageTk
import numpy as np
import pyttsx3

class ObjectDetector:
    def __init__(self):
        self.classNames = []
        with open('coco.names', 'rt') as f:
            self.classNames = [line.rstrip() for line in f]

        self.configPath = 'ssd_mobilenet_v3_large_coco_2020_01_14.pbtxt'
        self.weightsPath = 'frozen_inference_graph.pb'

        self.net = cv2.dnn_DetectionModel(self.weightsPath, self.configPath)
        self.net.setInputSize(320, 320)
        self.net.setInputScale(1.0/ 127.5)
        self.net.setInputMean((127.5, 127.5, 127.5))
        self.net.setInputSwapRB(True)

        self.engine = pyttsx3.init()

        self.root = Tk()
        self.root.title('Object Detector')
        self.root.protocol("WM_DELETE_WINDOW", self.close)

        # Set background color
        self.root.configure(background="#000080")

        # Create a frame to hold the label, buttons and text area
        self.frame = Frame(self.root, bg="#000080")
        self.frame.pack(padx=20, pady=20)

        

        # Create the label to display the video stream
        self.label = Label(self.frame, bg="#000000")
        self.label.pack(pady=10)

        self.last_label=""

        # Create the start button
        self.start_button = Button(self.frame, text="Start", command=self.start, width=10, height=2, bg="#32cd32", fg="#ecf0f1", font=("Helvetica", 12, "bold"))
        self.start_button.pack(side=LEFT, padx=10)

        # Create the stop button
        self.stop_button = Button(self.frame, text="Stop", command=self.stop, state=DISABLED, width=10, height=2, bg="#cd5c5c", fg="#ffffff", font=("Helvetica", 12, "bold"))
        self.stop_button.pack(side=LEFT, padx=10)

        # Create the text area to display last detected object
        self.text_area = Text(self.frame, height=2, width=50, font=("Helvetica", 12, "bold"))
        self.text_area.pack(pady=10)

        self.root.mainloop()

    def start(self):
        self.video_stream = cv2.VideoCapture(0)
        self.is_running = True
        self.start_button.config(state=DISABLED)
        self.stop_button.config(state=NORMAL)
        self.detect_objects()

    def stop(self):
        self.is_running = False
        self.video_stream.release()
        self.start_button.config(state=NORMAL)
        self.stop_button.config(state=DISABLED)

        if self.last_label:
            self.text_area.delete(1.0, END)
            self.text_area.insert(END, f"Last detected object: {self.last_label}")

    def detect_objects(self):
        success, img = self.video_stream.read()
        if success:
            img = cv2.resize(img, (640, 480))
            classIds, confs, bbox = self.net.detect(img, confThreshold=0.5)
            if len(classIds) != 0:
                last_object=None
                for classId, confidence, box in zip(classIds.flatten(),confs.flatten(),bbox):
                    label = self.classNames[classId-1].upper()
                    cv2.rectangle(img, box, color=(0,255,0), thickness=2)
                    cv2.putText(img, label, (box[0]+10, box[1]+30), cv2.FONT_HERSHEY_COMPLEX, 1, (0,255,0), 2)
                    cv2.putText(img, str(round(confidence*100,2)), (box[0]+200, box[1]+30), cv2.FONT_HERSHEY_COMPLEX, 1,(0,255,0), 2)
                    last_object=label
                    self.label.img = ImageTk.PhotoImage(image=Image.fromarray(cv2.cvtColor(img, cv2.COLOR_BGR2RGB)))
                    self.label.configure(image=self.label.img)

                    self.text_to_speech(label)

                if last_object is not None:
                    self.text_area.delete(1.0, END)
                    self.text_area.insert(END, f"Last detected object: {last_object}")
                    self.text_to_speech(last_object)
            if self.is_running:
                self.root.after(10, self.detect_objects)     

        else:
            self.stop()

    def text_to_speech(self, label):
        self.engine.say(label)
        self.engine.runAndWait()

    def close(self):
        self.is_running = False
        self.video_stream.release()
        self.text_area.destroy()
        self.root.destroy()

if __name__ == '__main__':
    obj_detector = ObjectDetector()