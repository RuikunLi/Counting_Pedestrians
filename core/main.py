# coding: utf-8

import os
import time
from pytz import timezone
from datetime import datetime

import cv2
import numpy as np
import streamlink
from imageai.Detection import ObjectDetection

from selenium import webdriver
import pandas as pd


######################################################################################################
#list of webcam
london = 'https://www.earthcam.com/world/england/london/abbeyroad/?cam=abbeyroad_uk'
timesquare = 'https://www.earthcam.com/usa/newyork/timessquare/?cam=tsrobo1'
dublin = 'https://www.earthcam.com/world/ireland/dublin/?cam=templebar'
######################################################################################################
#list of timezone
Dublin = 'Europe/Dublin'
London = 'Europe/London'
NYC = 'America/New_York'
######################################################################################################


class CountingObject(object):
    """
    A class of counting objects
    """

    algos = {
        "resnet": "resnet50_coco_best_v2.0.1.h5",
        "yolov3": "yolo.h5",
        "yolo_tiny": "yolo-tiny.h5"
    }

    def __init__(self, stream_link):
        self.stream_link = stream_link
        self.streams = streamlink.streams(stream_link)
        if self.streams is None:
            raise ValueError("cannot open the stream link %s" % stream_link)

        q = list(self.streams.keys())[0]
        self.stream = self.streams['%s' % q]

        self.target_img_path = os.getcwd()

        self.detector = ObjectDetection()
        if self.detector is None:
            raise ValueError("Detector of objects is None")

    def detector_init(self, algo="resnet", speed="nomal"):
        """
        Must be invoked after instantiate for initialize a object detector. 
        
        Args:
            algo (str): The algorithm of object detection tasks. "resnet"(default), "yolov3", "yolo_tiny".
            speed (str): The detection speed for object detetion tasks. "normal"(default), "fast", "faster" , "fastest" and "flash".
        
        Returns:
            void
        
        """

        if algo == "resnet":
            self.detector.setModelTypeAsRetinaNet()
            self.detector.setModelPath(
                os.path.join(self.target_img_path, self.algos["resnet"]))
        elif algo == "yolov3":
            self.detector.setModelTypeAsYOLOv3()
            self.detector.setModelPath(
                os.path.join(self.target_img_path, self.algos["yolov3"]))
        elif algo == "yolo_tiny":
            self.detector.setModelTypeAsTinyYOLOv3()
            self.detector.setModelPath(
                os.path.join(self.target_img_path, self.algos["yolo_tiny"]))
        else:
            print("Given algorithm of object detection is invalid.")
            return

        self.detector.loadModel(detection_speed=speed)
        self.custom_objects = self.detector.CustomObjects(person=True)

    def put_text_to_img(self, img, text, pos = (50,50), fontColor=(0,0,255), lineType=2):
        """
        Put text to an image.
        
        Args:
            img : An image represented by numpy array. You can use cv2.imread(path_to_iamge) to read an image in the filesystem by
                    giving the image path.
            text (str): The text what you want to put to the image.
            pos (tuple): x and y position relative to the origin (0,0) at the top left.
            fontColor (tuple): R G B channel.
            lineType (int): Type of line.
        
        Returns:
            void
        
        """
        if img is None:
            print("Put text to a none image.")
            return

        font = cv2.FONT_HERSHEY_SIMPLEX
        fontScale = 1

        cv2.putText(img, text, pos, font, fontScale, fontColor, lineType)

    def capture_frame_by_stream_wrapper(self,
                                        image_prefix="stream",
                                        mprob=30,
                                        num_im=6,
                                        time_interval=10,
                                        tz=None):
        """
        A wrapper of the function capture_frame_by_stream.
        
        Args:
            image_prefix (str): Prefix of target images. The postfix is numerated by numbers.
            mprob (int): Minimum probability to be a person.
            num_im (int): How many images will be taken.
            time_interval (int): Time interval of taking next image, the unit is second.
			tz (str): Time zone from package pytz. Default is None, then apply utc time. Use function pytz.all_timezones to get the list of timezones.
        
        Returns:
            void
        
        """
        print("The current conuting function is based on capture frame by stream.")

        dir_path = os.path.join(self.target_img_path, image_prefix)
        if not os.path.isdir(dir_path):
            os.makedirs(dir_path)
        frames_res = []
        if num_im <= 0:
            try:
                i = 0
                while True:
                    i = i + 1
                    frame_res = self.capture_frame_by_stream(
                        image_prefix, i, mprob, tz)
                    frames.res.append(frame_res)
                    time.sleep(time_interval)
            except KeyboardInterrupt:
                return frames_res
                print('Abort by key interrupt.')
        else:
            for i in range(num_im):
                frame_res = self.capture_frame_by_stream(
                    image_prefix, i, mprob, tz)
                frames_res.append(frame_res)
                time.sleep(time_interval)

            return frames_res

    def capture_frame_by_stream(self,
                                image_prefix="stream",
                                image_index=0,
                                mprob=30,
                                tz=None) -> int:
        """
        capture a frame from a online stream, namely webcam.
        
        Args:
            image_prefix (str): Prefix of target images. The postfix is numerated by numbers.
            image_index (int): The postfix of target images. By default, numerated from 0.
            mprob (int): Minimum probability to be a person.
		    tz (str): Time zone from package pytz. Default is None, then apply utc time. Use function pytz.all_timezones to get the list of timezones.

		
        Returns:
            tuple: The name of target image, the number of persons in an image detected by the model and the current time.
        """
		
        video_cap = cv2.VideoCapture(self.stream.url)
        dir_path = os.path.join(self.target_img_path, image_prefix)

        if video_cap is None:
            print("Open webcam [%s] failed." % self.stream.url)
            return None
        else:
            ret, frame = video_cap.read()

            if not ret:
                print("Captured frame is broken.")
                video_cap.release()
                return None
            else:
                print("-----------------------------------------------------")
                

                if tz is None:
                    current_time = datetime.utcnow().strftime(
                        "%a %Y-%m-%d %H:%M:%S")
                    print('### time zone is None, therefore use utc time###')
                else:
                    current_time = datetime.now(
                        timezone(tz)).strftime("%a %Y-%m-%d %H:%M:%S")

                print("Capturing frame %d." % image_index)
                target_img_name = "{}{}.png".format(image_prefix, image_index)
                # frame = crop_frame(frame, target_img_name)  # comment to unuse the crop function.
                
                cv2.imwrite(os.path.join(dir_path, target_img_name), frame)

                detections = self.detector.detectCustomObjectsFromImage(
                    custom_objects=self.custom_objects,
                    input_image=os.path.join(dir_path, target_img_name),
                    output_image_path=os.path.join(dir_path, target_img_name),
                    minimum_percentage_probability=mprob)

                print(
                    "The number of person in frame %d (%s):" %
                    (image_index, target_img_name), len(detections))
                print(
                    "The current time in frame %d (%s):" %
                    (image_index, target_img_name), current_time)

                img = cv2.imread(os.path.join(dir_path, target_img_name))
                # put the number of persons to the image and put timestamp to the image
                self.put_text_to_img(
                    img, "The number of person:%s " % str(len(detections)))
                img_height, img_width = img.shape[0:2]
                self.put_text_to_img(
                    img, "The current time:%s " % current_time, pos=(int(img_width*0.1), int(img_height*0.9)))

                cv2.imwrite(os.path.join(dir_path, target_img_name), img)
                video_cap.release()

                return target_img_name, len(detections), current_time

    def capture_frame_by_screenshot_wrapper(self,
                                            image_prefix="screenshot",
                                            mprob=30,
                                            num_im=6,
                                            time_interval=10,
                                            tz=None):
        """
        A wrapper of the function capture_frame_by_screenshot.
        
        Args:
            image_prefix (str): Prefix of target images. The postfix is numerated by numbers.
            mprob (int): Minimum probability to be a person.
            num_im (int): How many images will be taken.
            time_interval (int): Time interval of taking next image, the unit is second.
			tz (str): Time zone from package pytz. Default is None, then apply utc time. Use function pytz.all_timezones to get the list of timezones.

        
        Returns:
            void
        
        """
        print(
            "The current conuting function is based on capture frame by screenshot."
        )

        frames_res = []
        dir_path = os.path.join(self.target_img_path, image_prefix)
        if not os.path.isdir(dir_path):
            os.makedirs(dir_path)
        if num_im <= 0:
            try:
                i = 0
                while True:
                    i = i + 1
                    frame_res = self.capture_frame_by_screenshot(
                        image_prefix, i, mprob, tz)
                    frames_res.append(frame_res)
                    time.sleep(time_interval)
            except KeyboardInterrupt:
                if self.driver is not None:
                    self.driver.quit()
                return frames_res
                print('Abort by key interrupt.')
        else:
            for i in range(num_im):
                frame_res = self.capture_frame_by_screenshot(
                    image_prefix, i, mprob, tz)
                frames_res.append(frame_res)
                time.sleep(time_interval)

            if self.driver is not None:
                self.driver.quit()

            return frames_res

    def capture_frame_by_screenshot(self,
                                    image_prefix="screenshot",
                                    image_index=0,
                                    mprob=30,
                                    num_im=6,
                                    tz=None) -> int:
        """
       capture an image by taking a screenshot on an opened website via browser.
        
        Args:
            image_prefix (str): Prefix of target images. The postfix is numerated by numbers.
            image_index (int): The postfix of target images. By default, numerated from 0.
            mprob (int): Minimum probability to be a person.
			tz (str): Time zone from package pytz. Default is None, then apply utc time. Use function pytz.all_timezones to get the list of timezones.

        
        Returns:
            tuple: The name of target image, the number of persons in an image detected by the model and the current time.
        
        """
		
        dir_path = os.path.join(self.target_img_path, image_prefix)

        if self.driver is None:
            print("Web driver is none.")
            return None
        else:
            print("-----------------------------------------------------")

            if tz is None:
                current_time = datetime.utcnow().strftime(
                    "%a %Y-%m-%d %H:%M:%S")
                print('### time zone is None, therefore use utc time###')
            else:
                current_time = datetime.now(
                    timezone(tz)).strftime("%a %Y-%m-%d %H:%M:%S")

            target_img_name = "{}{}.png".format(image_prefix, image_index)
            print("Taking screenshot %d..." % image_index)
            self.driver.save_screenshot(
                os.path.join(dir_path, target_img_name))
            detections = self.detector.detectCustomObjectsFromImage(
                custom_objects=self.custom_objects,
                input_image=os.path.join(dir_path,
                                         target_img_name),
                output_image_path=os.path.join(dir_path, target_img_name),
                minimum_percentage_probability=mprob)

            print(
                "The number of person in frame %d (%s):" % (image_index,
                                                            target_img_name),
                len(detections))
            print(
                "The current time in frame %d (%s):" %
                (image_index, target_img_name), current_time)

            img = cv2.imread(os.path.join(dir_path, target_img_name))
            # put the number of persons to the image
            self.put_text_to_img(
                img, "The number of person is:%s" % str(len(detections)))
            img_height, img_width = img.shape[0:2]
            self.put_text_to_img(
                img, "The current time:%s " % current_time, pos=(int(img_width*0.1), int(img_height*0.9)))

            cv2.imwrite(os.path.join(dir_path, target_img_name), img)

            return target_img_name, len(detections), current_time

    def init_webdriver(self):
        """
       Initialize the webdriver of Chrome by using the python lib selenium.
        
        Args:
            Void
        
        Returns:
            Void
        """
		
        self.driver = webdriver.Chrome(
        )  # Optional argument, if not specified will search path.
        self.driver.get(self.stream_link)
        time.sleep(15)  # Jump over the ads
        
    def stroe_info_in_df_csv(self, image_prefix="counting_person"):
        """
       Collect test dataset by storing the image name and the detected number of persons in a csv file.
        
        Args:
            info():  
        
        Returns:
            Void
        """
		
        df = pd.DataFrame(
            np.array(res), columns=['image_name', 'detected_num', 'time'])
        # df["counted_num"] = ""  #only for baseline
        df.to_csv(
            path_or_buf=os.path.join(self.target_img_path, "%s.csv" %
                                     image_prefix))
        return df



	
def crop_frame(frame, target_img_name, y1=150, y2=500, x1=0, x2=1000):
    """
    only crop frame to evaluate the baseline. not a offical function, therefore outsied the class. will be deleted after evaluation.
    """
    frame = frame[y1:y2, x1:x2]

    if not os.path.isdir('.\\dublin_day_baseline'):
        os.makedirs('.\\dublin_day_baseline')
    path = '.\\dublin_day_baseline'
    cv2.imwrite(os.path.join(path, target_img_name), frame)
    return frame



if __name__ == "__main__":
    #     scheduler = BlockingScheduler()
    print("Starting...")
    counting_person = CountingObject(dublin)
    counting_person.detector_init()

    by_stream_flag = True
    img_prefix = "dublin_day"
    res = []
    if by_stream_flag:
        res = counting_person.capture_frame_by_stream_wrapper(
            image_prefix=img_prefix, num_im=50, time_interval=180, tz=Dublin)

    else:
        counting_person.init_webdriver()
        res = counting_person.capture_frame_by_screenshot_wrapper(num_im=2)

#     counting_person.store_baseline_info_in_csv(res)
    df = counting_person.stroe_info_in_df_csv(image_prefix=img_prefix)
    display(df)

    print('###Exit...')

