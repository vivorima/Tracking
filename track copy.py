import os

import cv2
import numpy as np

from Object import Object

# PATHS
images_path = "Dataset/groundtruth/"
video_path = "Dataset/input/"


def load_images(path):
    """load images from a specific folder and returns an np array"""
    images = []
    for filename in os.listdir(path):
        # avec le ground truth en grayscale
        if path == images_path:
            img = cv2.imread(os.path.join(path, filename),
                             cv2.IMREAD_GRAYSCALE)
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
    # -----------------------------------------------REF FRAME----------------------------------------------------------
    # get the objects in the first frame only (frame de ref)
    ref_objects = extract_objects(ground[0], 0)
    # since all these objects are new i will affect a new ID to all of them and insert them to my video objects
    for obj in ref_objects:
        add_object(my_objects, obj)

    # -------------------------------------FOR EACH FRAME IN THE VIDEO--------------------------------------------------
    for i in range(1, len(video)):
        # DETECT OBJECTS IN FRAME i
        frame_objects = extract_objects(ground[i], i)

        # ---------------------------------------------search the objects of frame i-1 (ref_objects) in the new frame i
        # they either dont exist(left), we find them (match) or  error (mismatch)
        for obj in ref_objects:
            # get the x,y of my object in the frame i-1
            coors = obj.getFrame(i - 1)
            # cropping the object to give to template
            img = cv2.cvtColor(video[i - 1][coors[1]:coors[1] + coors[3], coors[0]:coors[0] + coors[2]],
                               cv2.COLOR_BGR2GRAY)
            # search for the object (img) it in frame i
            res = cv2.matchTemplate(img, cv2.cvtColor(
                video[i], cv2.COLOR_BGR2GRAY), cv2.TM_SQDIFF_NORMED)
            # get the min location of the object "found" in frame i
            _, _, min_loc, _ = cv2.minMaxLoc(res)

            # SEARCHING FOR THE OBJECT NEAR MIN_LOC  (voisinage)
            point_1 = geometry.Point(min_loc)
            found = False
            for o in range(len(frame_objects)):
                x = frame_objects[o].getFrame(i)[0]
                y = frame_objects[o].getFrame(i)[1]
                point_2 = geometry.Point(x, y)
                circle_buffer = point_2.buffer(30)  # radius of 30
                if point_1.within(circle_buffer):
                    # WE FOUND THE OBJECT IN FRAME I
                    found = True
                    # update object position in frame i & coords in the frame i for obj
                    if obj in my_objects:
                        frame_objects[o].set_id(obj.get_id())
                        my_objects[my_objects.index(obj)].appears_frame(
                            i, frame_objects[o].getFrame(i))
                        # draw bb and
                        bounding_box(
                            my_objects[my_objects.index(obj)], video[i], i)
                        # we found a match so we break
                        break
                    else:
                        # should (and will) never happen but just in case we made a mistake
                        print(
                            i, " : ERROR updating an object that doesnt exist: ", obj)

            # if a mismatch occured for any reason: (merge / occlusion / descriptor error)
            if not found:
                # either they dont exist (they left) or we made a mistake
                # we suppose it left the frame because we cant really do anything here?????????????????????????????????
                print(i, " : No match found or mismatch for: ", obj)

        # RESULT TILL NOW: SOME OBJETCS IN REF WERE FOUND IN FRAME i, SOME WERENT (not found -> idk what to do about it)

        # REF <- FRAME I: so affect id to all objects in FRAME I AS WELL

        # some objects in frame i werent matched in frame i-1 --------->
        # they either are new or they left the shot (0,i-2) and came back in frame i
        for j in frame_objects:
            # search for them  in  objects of frames [0:i-2] ---------> using un voisinage aussi (latest frame)

            if j.get_id() == -1:  # if object was not found in i-1 (id not updated)
                # Initialisations
                point_1 = geometry.Point(j.getFrame(i))
                found = False
                v = 0
                # compare x,y with each object in the lastest frame
                while v < len(my_objects) and not found:
                    # voisinage dans la last frame -----> this way we dont search in all the frames
                    x = my_objects[v].get_last_frame()[0]
                    y = my_objects[v].get_last_frame()[1]
                    point_2 = geometry.Point(x, y)
                    circle_buffer = point_2.buffer(10)

                    # if we find a match then we update (could be wrong but whatever: occlusion problem)
                    if point_1.within(circle_buffer):
                        j.set_id(my_objects[v].get_id())
                        my_objects[v].appears_frame(i, j.getFrame(i))
                        # since obj is already in my_objects it will be updated auto
                        bounding_box(my_objects[v], video[i], i)
                        found = True
                    v += 1

                # if we dont find anything then its a new object
                if not found:
                    j.set_id(getID(my_objects))
                    add_object(my_objects, j)
                    bounding_box(j, video[i], i)

        cv2.imshow('video with bbs', video[i])
        cv2.waitKey(100)
        # clearing the old frame objects
        ref_objects.clear()
        # adding the new ones with the new IDs
        ref_objects.extend(frame_objects)


def extract_objects(ground_frame, frame_num):
    # OLD OPENCV VERSION, RANIA JUST DELETE ret
    contours, hierarchy = cv2.findContours(
        ground_frame, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
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

    print("Objects Detected:")
    for detected in my_objects:
        print(detected)


if __name__ == "__main__":
    main()
