# Raspberry Pi face recognition code for the dorm door.
This project provides students living in a dorm a convenient way to establish a face recognition system to control the door.

Click the image to watch the video. (In Chinese)
[![Watch the video](https://github.com/p0werHu/Face-Recognition-by-Raspberry-Pi/blob/master/image.jpg?raw=true)](https://www.youtube.com/watch?v=aTo2CXa4aSs&feature=youtu.be)

## How to use the system
Step 1: You should purchase a Raspberry Pi and install an ubuntu system in it.  
Step 2: Our face recognition algorithm is from [FaceNet](https://github.com/davidsandberg/facenet). Watch how to train your model and get checkpoint from this link. There are two checkpoints, the first one is provided in the link; and the second one is trained by yourself.  
Step 3: We lost the code in Raspberry Pi, thus, you could find any camera code of Raspberry Pi and send the image to the server when using it.  
Step 4: Change paths in the code to your config including IP and model path.  

## Tip
Any code except test.py shouldn't be modified.

## Acknowledgment
The model come from [FaceNet](https://github.com/davidsandberg/facenet).
