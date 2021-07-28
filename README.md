# YOLOv5 and Simple Online Realtime Tracking for Counting Cars 
This repository for Detecting and Counting Cars of top-view using YOLOv5 detecting and Deep-SORT tracking


<div align="center">
<p>
<img src="eval/input.gif" width="400"/> <img src="eval/output.gif" width="400"/> 
</p>

## Introduction

This implementation for detecting, tracking and counting cars, YOLOv5 for training, i labeled Dataset of more than 300 images of cars by top-view using Roboflow.
Then by using augmentation for these images, i created more than 3000 images ready for training. 
for tracking the cars, i used Deep-Sort Algorithm which uses the Kalman Filter to track objects, then i stored the IDs of the detected cars in a lsit and print its length

Input Video: https://drive.google.com/file/d/1V9ngeP0G8FSpe13VFjN62R4KWXADLMCM/view?usp=sharing

Dataset: https://app.roboflow.com/ds/43Gg8QfcNi?key=WPiscHczVq

Model Weights: https://drive.google.com/file/d/1EnPc04tFzUDlWDAO4Oi7nDlSUp4MhFEB/view?usp=sharing

Output Video: https://drive.google.com/file/d/1vLfAxRClU-iJnZjOkU6JVtlxsnC7JQIo/view?usp=sharing


## References
  - YOLOv5: https://github.com/ultralytics/yolov5.git
  - Deep-SORT: https://github.com/mikel-brostrom/Yolov5_DeepSort_Pytorch.git
