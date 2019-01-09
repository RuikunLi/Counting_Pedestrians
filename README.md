# Counting_Objects

This project tries to detect objects in images captured by online webcams. It supports two ways to capture images/frames, one is to directly capture frames from a live stream, another is to take a screenshot by opening the live stream via a web browser. The class CountingObject contains methods that you might use. Here a simple example to use this module is the following.

```python
# step.1 new a object of class CountingObject 
counting_person = CountingObject(stream_link)  # passing a valid stream link (url)
# step.2 invoke the init method to finish the init things
counting_person.detector_init()	
# step.3 capture by stream, creating a folder named "target_imgs" at current working directory, in which all captured frames are stored as images. 
# "tz": specifies the time zone in which the webcam locates, the concrete time zone list is available in the python lib pytz.
# "num_im": specified how many images you expect to be captured.
# "time_interval": the interval of capturing an image in unit of second.
counting_person.capture_frame_by_stream_wrapper(image_prefix="target_imgs", num_im=50, time_interval=180, tz='Europe/Dublin')
# step.3 capture by screenshot
counting_person.init_webdriver()
counting_person.capture_frame_by_screenshot_wrapper(num_im=2)
```

# Dependencies

## Dependencies of ImageAI ([source](https://github.com/OlafenwaMoses/ImageAI#installation))

- Python 3.5.1 (and later versions) 
- pip3
- Tensorflow 1.4.0 (and later versions)
  - pip3 install --upgrade tensorflow 
- Numpy 1.13.1 (and later versions)
  - pip3 install numpy 
- SciPy 0.19.1 (and later versions)
  - pip3 install scipy 
- OpenCV 
  - pip3 install opencv-python 
- Pillow
  - pip3 install pillow 
- Matplotlib
  - pip3 install matplotlib 
- h5py 
  - pip3 install h5py 
- Keras 2.x
  - pip3 install keras 
- ImageAI
  - pip3 install https://github.com/OlafenwaMoses/ImageAI/releases/download/2.0.2/imageai-2.0.2-py3-none-any.whl 

 ## Dependencies of the rest

- Pandas
  - pip3 install pandas
- Streamlink
  - pip3 install -U streamlink
- Selenium
  - pip3 install selenium

# Prerequisites

## Detection Models
Download the dection model you would like to apply via https://github.com/OlafenwaMoses/ImageAI/tree/master/imageai/Detection. Place them in the folder of the main file in which the main program is executed. Three models are available, for instance listed below:

1. resnet50_coco_best_v2.0.1
2. yolo.h5
3. yolo-tiny.h5

## webdrivers for the way of taking screenshots [optional]

We apply chrome to do our experiment, so here we provide the source to download chromedriver via http://chromedriver.chromium.org/downloads
The web driver is only required if you use the way of taking screenshots when opening the webcam by chrome. Once you download it, please put it into the folder where the main python file "main.py" is located.

# Run Code

If you prepare all things mentioned above, then executing `python main.py`.

## Passinng command line args

By default, the target images are stored in the folder named as `target_imgs` that is also the image prefix. The default way of capturing images is by a live stream other than by taking screenshots. By passing command line args you can adapt it.

### Command line args

You can print the help message by `python main.py -h`.

> usage: main.py \[-h\]\[-s\]\[-l LINK\]\[-p prefix\]\[-n NUMBER\]\[-i INTERVAL\]\[-z TIMEZONE\]

```wiki
optional arguments:
  -h, --help            show this help message and exit
  -s                    Capture images by taking screenshot.
  -l LINK, --link LINK  The stream link indicates a webcam.
  -p prefix, --prefix prefix
                        The prefix of target image.
  -n NUMBER, --number NUMBER
                        The number of images you want to collect.
  -i INTERVAL, --interval INTERVAL
                        The interval of capturing two consecutive images.
  -z TIMEZONE, --timezone TIMEZONE
                        The timezone since the current time will be placed in
                        the captured images.
```