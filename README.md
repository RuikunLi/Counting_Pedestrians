# Counting_Objects

This project tries to detect objects in images captured by online webcams. It supports two ways to capture images/frames, one is to directly capture frames from a live stream, another is to take a screenshot by opening the live stream via a web browser. The class CountingObject contains methods that you might use. Here a simple example to use this module is the following.

```python
# step.1 new a object of class CountingObject 
counting_person = CountingObject(stream_link)  # passing a valid stream link (url)
# step.2 invoke the init method to finish the init things
counting_person.detector_init()	
# step.3 capture by stream, creating a folder named "target_imgs" at current working directory, in which all captured frames are stored as images. The parameter "tz" specifies the time zone in which the webcam locates, the concrete time zone list is available in the python lib pytz.
counting_person.capture_frame_by_stream_wrapper(image_prefix="target_imgs", num_im=50, time_interval=180, tz='Europe/Dublin')
# step.3 capture by screenshot
counting_person.init_webdriver()
counting_person.capture_frame_by_screenshot_wrapper(num_im=2)
```

# Dependencies

## Dependencies of ImageAI (source: https://github.com/OlafenwaMoses/ImageAI#installation)

- Python 3.5.1 (and later versions) 
- pip3
- Tensorflow 1.4.0 (and later versions)
 pip3 install --upgrade tensorflow 
- Numpy 1.13.1 (and later versions)
 pip3 install numpy 
- SciPy 0.19.1 (and later versions)
 pip3 install scipy 
- OpenCV 
 pip3 install opencv-python 
- Pillow
 pip3 install pillow 
- Matplotlib
 pip3 install matplotlib 
- h5py 
 pip3 install h5py 
- Keras 2.x
 pip3 install keras 
- ImageAI
 pip3 install https://github.com/OlafenwaMoses/ImageAI/releases/download/2.0.2/imageai-2.0.2-py3-none-any.whl 

 ## Dependencies of the rest

- Pandas
 pip3 install pandas
- Streamlink
 pip3 install -U streamlink
- Selenium
 pip3 install selenium

