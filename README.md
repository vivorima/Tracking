# Tracking Project
 ## Stages of object tracking in video sequences.
The classification steps are :
- extraction of moving objects (by background subtraction).
- extraction of object descriptors.
- object tracking by matching.

## The output of the tracking system will be:
- A trajectory of each moving object in the scene,
- The video with a bounding box for each object with the same color for the same object.
- A list (or array) containing the objects with their trajectories (coordinates of the bounding box
in each frame of the video).
