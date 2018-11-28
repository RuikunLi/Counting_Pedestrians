import os
import time
from datetime import timedelta

import cv2
import streamlink
from imageai.Detection import VideoObjectDetection, ObjectDetection
from apscheduler.schedulers.blocking import BlockingScheduler
from selenium import webdriver


class CountingObject(object):
    """
    A class of counting objects
    """
    
    algos = {"resnet": "resnet50_coco_best_v2.0.1.h5", "yolov3": "yolo.h5", "yolo_tiny": "yolo-tiny.h5"}
    
    def __init__(self, stream_link):
        self.stream_link = stream_link
        self.streams = streamlink.streams(stream_link)
        if self.streams is None:
            raise ValueError("cannot open the stream link %s" % stream_link)
            
        self.stream = self.streams['720p']
        self.target_img_path = os.getcwd()
        
        self.detector = ObjectDetection()
        if self.detector is None:
            raise ValueError("Detector of objects is None")
        
        
    def detector_init(self, algo="resnet", speed="fast"):
        """
        Must be invoked after instantiate for initialize a object detector. 
        
        Args:
            algo (str): The algorithm of object detection tasks.
            speed (str): The detection speed for object detetion tasks. "normal"(default), "fast", "faster" , "fastest" and "flash".
        
        Returns:
            void
        
        """
        
        if algo == "resnet":
            self.detector.setModelTypeAsRetinaNet()
            self.detector.setModelPath(os.path.join(self.target_img_path, self.algos["resnet"]))
        elif algo == "yolov3":
            self.detector.setModelTypeAsYOLOv3()
            self.detector.setModelPath(os.path.join(self.target_img_path, self.algos["yolov3"]))
        elif algo == "yolo_tiny":
            self.detector.setModelTypeAsTinyYOLOv3()
            self.detector.setModelPath(os.path.join(self.target_img_path, self.algos["yolo_tiny"])) 
        else:
            print("Given algorithm of object detection is invalid.")
            return
        
        self.detector.loadModel()
        self.detector.loadModel(detection_speed=speed)
        self.custom_objects = self.detector.CustomObjects(person=True)

    def put_text_to_img(self, img, text):
        if img is None:
            print("Put text to a none image.")
            return
        
        font                  = cv2.FONT_HERSHEY_SIMPLEX
        upperLeftCornerOfText = (50,50)
        fontScale             = 1
        fontColor             = (255,255,255)
        lineType              = 2

        cv2.putText(img, text,  
                    upperLeftCornerOfText, 
                    font, 
                    fontScale,
                    fontColor,
                    lineType)

    def capture_frame_by_stream(self):
        video_cap = cv2.VideoCapture(self.stream.url)
        if video_cap is None:
            print("Open webcam [%s] failed." % self.stream.url)
            return
        else:
            ret, frame = video_cap.read()

            if not ret:
                print("Captured frame is broken.")
                return 
            else:
                print("Captured one frame.")

            detections = self.detector.detectCustomObjectsFromImage(custom_objects=self.custom_objects, 
                                                  input_type="array", 
                                                  input_image=frame, 
                                                  output_image_path=os.path.join(self.target_img_path , "image.jpg"), 
                                                  minimum_percentage_probability=30)
            


            img = cv2.imread(os.path.join(self.target_img_path, "image.jpg"))
            # put the number of persons to the image
            self.put_text_to_img(img, str(len(detections)))

            cv2.imshow("image", img)
            cv2.waitKey(0) # blocked until pressing Enter key
            cv2.destroyAllWindows()
            video_cap.release()
            #detections = detector.detectObjectsFromImage(input_type="array", input_image=frame, output_image_path=os.path.join(execution_path , "image2new.jpg"), minimum_percentage_probability=30)

    def capture_frame_by_screenshot(self):
        if self.driver is None:
            print("Web driver is none.")
            return 
        
        self.driver.save_screenshot(os.path.join(self.target_img_path , "screenshot.png"))
        print("Take a screenshot.")
        detections = self.detector.detectCustomObjectsFromImage(custom_objects=self.custom_objects, 
                                              input_image=os.path.join(self.target_img_path , "screenshot.png"), 
                                              output_image_path=os.path.join(self.target_img_path , "screenshot_result.png"), 
                                              minimum_percentage_probability=30)
        img = cv2.imread(os.path.join(self.target_img_path, "screenshot_result.png"))
        # put the number of persons to the image
        self.put_text_to_img(img, str(len(detections)))

        cv2.imshow("image", img)
        cv2.waitKey(0) # blocked until pressing Enter key
        cv2.destroyAllWindows()

    def init_webdriver(self):#nim: number of image, tin: time interval
        self.driver = webdriver.Chrome(os.path.join(self.target_img_path, "chromedriver.exe"))  # Optional argument, if not specified will search path.
        self.driver.get(self.stream_link)
        time.sleep(15) # Jump over the ads
        
if __name__ == "__main__":
#     scheduler = BlockingScheduler()
    print("Starting")
    counting_person = CountingObject("https://www.earthcam.com/world/england/london/abbeyroad/?cam=abbeyroad_uk")
    counting_person.detector_init("yolov3")

#     counting_person.init_webdriver()

    while True:
        counting_person.capture_frame_by_stream()
#         counting_person.capture_frame_by_screenshot()
        time.sleep(10)
        
    print('###exit.')
        
#  raises errors, nor ready for using.
#     scheduler.add_job(capture_frame, 'interval', seconds=5, args=[cap, detect, custom_objects, target_img_path])
#     scheduler.start()

