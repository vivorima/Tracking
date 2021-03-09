import numpy as np
import cv2
import matplotlib as plt
import glob

# im fine :)
path = "Dataset/highway/groundtruth/*.png"


def ArrayImages(path):
    # array contenant les images de la vid: FRAMES
    input_array = []
    for frame in glob.glob(path):
        img = cv2.imread(frame)
        size = (img.shape[1], img.shape[0])
        input_array.append(img)
    print("images loaded and put in an array..")
    return input_array


def CreateVideoFromImages(array_images):
    video = cv2.VideoWriter('video.avi', cv2.VideoWriter_fourcc(*'DIVX'), 15,
                            (array_images[0].shape[1], array_images[0].shape[0]))
    for i in range(len(array_images)):
        video.write(array_images[i])
    video.release()
    print("video created..")
    return video


# - extraction d'objets
def ExtractObjectsFromFrame(frame):
    objects = []
    pass

# cv2.waitKey(0)

# Main Function
def main():
    frames = ArrayImages(path)
    CreateVideoFromImages(frames)
    # - extraction de descripteurs des objets.
    # - suivi des objets par appareillement (matching).

if __name__ == "__main__":
    main()

print('Program Completed!')