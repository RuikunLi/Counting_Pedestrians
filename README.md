# Counting_Objects

This project tries to detect objects in images captured by online webcams. It supports two ways to capture images/frames, one is to directly capture frames from a live stream, another is to take a screenshot by opening the live stream via a web browser. The class CountingObject contains methods that you might use. Here a simple example to use this module is the following.

```python
# step.1 new a object of class CountingObject 
counting_person = CountingObject(stream_link)  # passing a valid stream link (url)
# step.2 invoke the init method to finish the init things
counting_person.detector_init()	
# step.3 capture by stream
counting_person.capture_frame_by_stream_wrapper(image_prefix="dublin_night", num_im=50, time_interval=180, tz='Europe/Dublin')
# step.3 capture by screenshot
counting_person.init_webdriver()
counting_person.capture_frame_by_screenshot_wrapper(num_im=2)
```

