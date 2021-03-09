import numpy as np
import cv2
import os
import random

# im fine :)
images_path = "dataset/groundtruth/"

video_path = "dataset/input/"

# load images from a specific folder and returns an np array


def loadImages(path):

    images = []

    for filename in os.listdir(path):

        # dont remove this coz gray scale is needed pour que detect contours accept hadak
        # avec le ground truth
        if(path == "dataset/groundtruth/"):

            img = cv2.imread(os.path.join(path, filename),
                             cv2.IMREAD_GRAYSCALE)
        else:
            img = cv2.imread(os.path.join(path, filename))

        if img is not None:

            images.append(img)

    print("Video loaded ...")
    return np.array(images)


def readVideo(video, ground, stop):

    # this has to change when we track objects
    #  we'll have to give a color to every object
    color = (119, 119, 2)

    for i in range(0, len(video)):

        # detects contours on ground images
        contours, _ = cv2.findContours(
            ground[i], cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        # all bounding boxes of a frame
        frame_bb = np.zeros((len(contours), 4))

        for k, c in enumerate(contours):

            x, y, w, h = cv2.boundingRect(c)

            # that s to keep track of each frame's bounding box later we group
            # theme depending on ttracker we gonna use
            frame_bb[k] = [x, y, w, h]

            # je dessine les contours
            cv2.drawContours(video[i], c, -1, (224, 227, 231), 2)

            # je dessine les bb
            cv2.rectangle(video[i], (x, y), (x+w, y+h), color, 2)

        cv2.imshow("Edges", ground[i])
        cv2.imshow('image', video[i])

        # equivalent to 60fps
        if(cv2.waitKey(16) & 0xFF == ord('q')):
            stop = True
            break

    return stop


# Main Function
def main():
    # video to display
    video = loadImages(video_path)
    # ground truth
    images = loadImages(images_path)

    # repeat my video endlessly
    stop = False

    while(not stop):

        stop = readVideo(video, images, stop)

        cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
