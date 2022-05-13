# README: Pose Gesture Analysis
The goal of this project is to take in video data, extract pose information from it, embed the pose data into an embedded vector and visualize the data in 2-dimensional space. This is done to extract meaningful information from the pose data of vide input.
## Dependencies
* python 3
* numpy
* pandas
* scikit learn
* cv2
* mediapipe
* plotly

## How to Run
1. Make sure all dependencies are downloaded and available in the working directory.
2. In the terminal, run...

        py main.py *videofilename1*

    for single video, cluster mode which will cluster the data using Gaussian Mixture clustering

    **OR** run...
    
        py main.py *videofilename1* *videofilename2* etc...

    to run with multiple videos and without clustering.