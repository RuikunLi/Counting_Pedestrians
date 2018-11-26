import os
import time
from datetime import timedelta

import cv2
import streamlink
from imageai.Detection import VideoObjectDetection, ObjectDetection
from apscheduler.schedulers.blocking import BlockingScheduler


streams = streamlink.streams("https://www.earthcam.com/world/england/london/abbeyroad/?cam=abbeyroad_uk")
stream = streams['720p']
url = stream.url

target_img_path = os.getcwd()


def detector_init():
    det = ObjectDetection()
    det.setModelTypeAsRetinaNet()
    det.setModelPath(os.path.join(target_img_path, "resnet50_coco_best_v2.0.1.h5"))
    det.loadModel()
    return det
    
def capture_frame(video_cap, detector, object_types, execution_path, every_frames=20):
    if not video_cap:
        print("Open webcam [%s] failed." % url)
        return
    else:
        while every_frames > 0:
            ret, frame = video_cap.read()
            every_frames -= 1

        if not ret:
            print("Captured frame is broken.")
            return 
        else:
            print("Captured one frame.")

        detector.detectCustomObjectsFromImage(custom_objects=object_types, 
                                              input_type="array", 
                                              input_image=frame, 
                                              output_image_path=os.path.join(execution_path , "image.jpg"), 
                                              minimum_percentage_probability=30)
        img = cv2.imread(os.path.join(execution_path, "image.jpg"))
        cv2.imshow("image", img)
        cv2.waitKey(0)
        cv2.destroyAllWindows()
        #detections = detector.detectObjectsFromImage(input_type="array", input_image=frame, output_image_path=os.path.join(execution_path , "image2new.jpg"), minimum_percentage_probability=30)
if __name__ == "__main__":
    scheduler = BlockingScheduler()
    cap = cv2.VideoCapture(url)
    detect = detector_init()
    custom_objects = detect.CustomObjects(person=True)
    while True:
        capture_frame(cap, detect, custom_objects, target_img_path)
        time.sleep(10)
        
    print('###exit.')
        
#  raises errors, nor ready for using.
#     scheduler.add_job(capture_frame, 'interval', seconds=5, args=[cap, detect, custom_objects, target_img_path])
#     scheduler.start()

