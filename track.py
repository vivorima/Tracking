import os

import cv2
import numpy as np
from shapely import geometry

from Object import Object

# PATHS
images_path = "Dataset/PETS2006/groundtruth/"
video_path = "Dataset/PETS2006/input/"


def load_images(path):
    """load images from a specific folder and returns an np array"""
    images = []
    for filename in os.listdir(path):
        # avec le ground truth en grayscale
        if path == images_path:
            img = cv2.imread(os.path.join(path, filename), cv2.IMREAD_GRAYSCALE)
        else:
            img = cv2.imread(os.path.join(path, filename))
        if img is not None:
            images.append(img)
    if path == images_path:
        print("Images loaded...")
    else:
        print("Video Loaded...")
    return np.array(images)


# detects objects in first frame, search for the same objects in second frame
def matching(my_objects, video, ground):
    """reads the video,detects contours and affects a bounding box to every object in a frame"""

    # get the objects in the first frame only
    ref_objects = extract_objects(ground[0], 0)
    # since all these objects are new i will affect a new ID to all of them and insert them to my video objects
    for obj in ref_objects:
        add_object(my_objects, obj)

    # for every frame
    for i in range(1, len(video)):
        # get the objects in frame i
        frame_objects = extract_objects(ground[i], i)

        # search the objects of frame i-1 (ref_objects) in the new frame i
        for obj in ref_objects:
            # get the x,y of my object in the frame i-1
            coors = obj.getFrame(i - 1)
            # cropping the object to give to template
            img = cv2.cvtColor(video[i - 1][coors[1]:coors[1] + coors[3], coors[0]:coors[0] + coors[2]],
                               cv2.COLOR_BGR2GRAY)
            # search for the object (img) it in frame i
            res = cv2.matchTemplate(img, cv2.cvtColor(video[i], cv2.COLOR_BGR2GRAY), cv2.TM_SQDIFF_NORMED)
            # get the min location of the object "found" in frame i
            _, _, min_loc, _ = cv2.minMaxLoc(res)

            # because these coords are not the best: we will search in frame i objects (extracted with contours)
            # the closest object to these coords: we define un voisinage (kinda like meanshift)
            point_1 = geometry.Point(min_loc)
            for o in range(len(frame_objects)):
                x = frame_objects[o].getFrame(i)[0]
                y = frame_objects[o].getFrame(i)[1]
                point_2 = geometry.Point(x, y)
                circle_buffer = point_2.buffer(30)  # radius of 30

                # to ensure we are taking the right object: voisinage
                if point_1.within(circle_buffer):
                    print(i, ": match", point_1, point_2)
                    # then the object in i is the same as the object in i-1
                    # update object position in frame i

                    # update coords in the frame i for obj
                    if obj in my_objects:
                        frame_objects[o].set_id(obj.get_id())
                        my_objects[my_objects.index(obj)].appears_frame(i, frame_objects[o].getFrame(i))
                        # draw bb and
                        bounding_box(my_objects[my_objects.index(obj)], video[i], i)
                        # we found a match so we break
                        break
                    else:
                        print("--------------------------", obj)

                # else:
                #     print(i,"Not a match", point_1,point_2)

            #         # DISTANCE ENTRE PONTS -------------------------------------------------------------> BAD RESULTS
            #         # search for these coords in my frame_objects using minimal distance-------------------> OPTIMIZE
            #         # dis = 1000000
            #         # temp = -1
            #         # for o in range(len(frame_objects)):
            #         #     x = frame_objects[o].getFrame(i)[0]
            #         #     y = frame_objects[o].getFrame(i)[1]
            #         #     value = math.sqrt((x - min_loc[0])**2 + (y - min_loc[1])** 2)
            #         #     # print(i, value)
            #         #     if value < dis:
            #         #         dis = value
            #         #         temp = o

            # LOOP:  we do this for all objects in frame i-1

            # if there are still objects in frame i that werent in frame i-1 they either are new or they left the shot
            for j in frame_objects:
                if j.get_id() == -1:  # objects not found in i-1
                    # search for them  in  objects of frames [0:i-2] (my_objects) (maybe they left the shot and came
                    # back later)
                    point_1 = geometry.Point(j.getFrame(i))
                    found = False
                    v = 0
                    while v < len(my_objects) and not found:
                        # voisinage dans la last frame but we really shouldnt use this: want to use un descripteur
                        x = my_objects[v].get_last_frame()[0]
                        y = my_objects[v].get_last_frame()[1]
                        point_2 = geometry.Point(x, y)
                        circle_buffer = point_2.buffer(10)
                        # if they exist update my_objects
                        if point_1.within(circle_buffer):
                            j.set_id(my_objects[v].get_id())
                            my_objects[v].appears_frame(i, j.getFrame(i))
                            # since obj is already in my_objects it will be updated auto
                            bounding_box(my_objects[v], video[i], i)
                            found = True
                        v += 1
                    # if they dont exist then its a new object
                    # when we reach the end of our list :3
                    if not found:
                        j.set_id(getID(my_objects))
                        add_object(my_objects, j)
                        bounding_box(j, video[i], i)

        cv2.imshow('video with bbs', video[i])
        cv2.waitKey(1)
        # clearing the old frame objects
        ref_objects.clear()
        # adding the new ones with the new IDs
        ref_objects.extend(frame_objects)
        # frame ++


def extract_objects(ground_frame, frame_num):
    # OLD OPENCV VERSION, RANIA JUST DELETE ret
    ret, contours, hierarchy = cv2.findContours(ground_frame, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    # this will contain the frames's objects
    objects = []
    # randomize color of all the objects boxes
    colors = list(np.random.random(size=3) * 256 for i in range(len(contours)))

    '''pour chaque contour on va créer un bounding box'''
    for k, c in enumerate(contours):
        x, y, w, h = cv2.boundingRect(c)
        # pour contourner le probleme des petites windows
        if w > 30 and h > 30:
            # je dessine les contours
            # cv2.drawContours(frame, c, -1, (224, 227, 231), 2)
            '''creating my object'''
            o = Object(-1, colors[k])
            o.appears_frame(frame_num, [x, y, w, h])
            objects.append(o)
    return objects


def bounding_box(object, frame, frame_num):
    """va me dessiner sur une frame le bb de mon objet"""
    frame_bb = object.getFrame(frame_num)
    cv2.rectangle(frame, (frame_bb[0], frame_bb[1]), (frame_bb[0] + frame_bb[2], frame_bb[1] + frame_bb[3]),
                  object.get_color(), 2)
    cv2.putText(frame, str(object.get_id()), (frame_bb[0], frame_bb[1] - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.5,
                object.get_color(), 1, cv2.LINE_AA)


# me retourne an id for the new objects
def getID(my_objects):
    return len(my_objects) + 1


def add_object(my_objects, obj):
    obj.set_id(len(my_objects) + 1)
    my_objects.append(obj)


# Main Function
def main():
    my_objects = []
    # video to display
    video = load_images(video_path)
    # ground truth
    images = load_images(images_path)
    print("Waiting for Matching...")
    matching(my_objects, video, images)

    # OLD RESULTS
    # for i in range(1, len(video)):
    #     # get the objects in frame i
    #     frame_objects = extract_objects(images[i], i)
    #     for obj in frame_objects:
    #         bounding_box(obj, video[i], i)
    #     cv2.imshow('video with bbs', video[i])
    #     cv2.waitKey(1)

    print("Objects Detected:")
    for detected in my_objects:
        print(detected)


if __name__ == "__main__":
    main()
