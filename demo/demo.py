import os
import threading

import numpy as np
import cv2
import streamlink
from imageai.Detection import VideoObjectDetection, ObjectDetection


streams = streamlink.streams("https://www.earthcam.com/world/england/london/abbeyroad/?cam=abbeyroad_uk")
stream = streams['720p']
url = stream.url

#cap = cv2.VideoCapture(url)
execution_path = os.getcwd()

def forFrame(frame_number, output_array, output_count):
    print("FOR FRAME " , frame_number)
    print("Output for each object : ", output_array)
    print("Output count for unique objects : ", output_count)
    print("------------END OF A FRAME --------------")

def forSeconds(second_number, output_arrays, count_arrays, average_output_count):
    print("SECOND : ", second_number)
    print("Array for the outputs of each frame ", output_arrays)
    print("Array for output count for unique objects in each frame : ", count_arrays)
    print("Output average count for unique objects in the last second: ", average_output_count)
    print("------------END OF A SECOND --------------")

def forMinute(minute_number, output_arrays, count_arrays, average_output_count):
    print("MINUTE : ", minute_number)
    print("Array for the outputs of each frame ", output_arrays)
    print("Array for output count for unique objects in each frame : ", count_arrays)
    print("Output average count for unique objects in the last minute: ", average_output_count)
    print("------------END OF A MINUTE --------------")


def detector_init():
    detector = ObjectDetection()
    detector.setModelTypeAsRetinaNet()
    detector.setModelPath( os.path.join(execution_path , "resnet50_coco_best_v2.0.1.h5"))
    detector.loadModel()
    return detector


def capture_frame_timer(stream_url, seconds=10, dec=None):
    cap = cv2.VideoCapture(stream_url)
    if not cap:
        print("Open webcam [%s] failed." % stream_url)
        return
    else:
        #threading.timer(seconds, capture_frame, [seconds]).start()
        capture_frame(cap, dec)
        print("Start capturing one frame per %s seconds" % seconds)
    
    
def capture_frame(cv_cap, detector):
    if cv_cap:
        ret, frame = cv_cap.read()
        print('Captured one frame.')
        #cv2.imshow('frame',frame)
        custom_objects = detector.CustomObjects(person=True)
        detector.detectCustomObjectsFromImage(custom_objects=custom_objects, input_type="array", input_image=frame, output_image_path=os.path.join(execution_path , "image3custom.jpg"), minimum_percentage_probability=30)
        #detections = detector.detectObjectsFromImage(input_type="array", input_image=frame, output_image_path=os.path.join(execution_path , "image2new.jpg"), minimum_percentage_probability=30)
    else:
        print("Capture camera invalid.")

detector = detector_init()
        
capture_frame_timer(url, 5, detector)