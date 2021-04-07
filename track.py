import os

import cv2
import numpy as np
import math
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

# remove unidentified frames


def processFrames(frames):

    for i in range(0, len(frames)):

        objs = []

        for obj in frames[i]:

            if (obj.get_id() != -1):

                if(not(obj.get_id() == 1 and obj.get_ratio() > 0.9)):

                    objs.append(obj)

        frames[i] = np.array(objs)


# detects objects in first frame, search for the same objects in second frame
def matching(video, ground, video_speed):

    frames = []

    ID_OBJ = 1

    seuil_fixe = 15

    # for each frame i extract objects
    for image in ground:

        frames.append(extract_objects(image))

    # i gave ids to 1st frame's objects
    for detected_obj in frames[0]:

        detected_obj.set_id(ID_OBJ)
        ID_OBJ += 1

    # pour chaque frame de la video
    for i in range(1, len(frames)):

        # pour chaque objet de la frame i
        for obj1 in frames[i]:

            distances = np.zeros(len(frames[i-1]))

            # je calcule la distance entre les centroid des objets de ma frame actuelle "i"
            # avec ceux des objets de la frame precedente "i-1"

            for k, obj2 in enumerate(frames[i-1]):

                distances[k] = math.dist(
                    obj1.get_centroid(), obj2.get_centroid())

            # je reecupere l'indice de la distance minimale ça correspond aussi
            # a l indice de l'objet (probablement recherché) dans la frame i - 1
            if(len(distances) > 0):

                min_index = np.argmin(distances)

                # s'il y a matching grace au seuil
                # l'objet a le meme id et couleur que l'objet trouvé
                if(np.min(distances) <= seuil_fixe):

                    matching_obj = frames[i-1][min_index]

                    obj1.set_id(matching_obj.get_id())

                else:
                    # si je ne trouve pas le match
                    # je cherche dans les 34 frames precedentes
                    #  ( au max 34 frames psk sinon la distance devient trop grande )
                    found = False

                    found_ratio = False

                    seuil = 44

                    frame_obj = []

                    for j in range(i-2, i-35, -1):

                        if(i-2 > 0):

                            distances = np.zeros(len(frames[j]))

                            for v, obj2 in enumerate(frames[j]):

                                distances[v] = math.dist(
                                    obj1.get_centroid(), obj2.get_centroid())

                                # on save l'indice de la frame et l'indice du dernier objet qui a le
                                # meme ratio que l'objet qu'on recherche
                                if(abs(obj1.get_ratio() - obj2.get_ratio() == 0)):

                                    frame_obj = [j, v, distances[v]]
                                    found_ratio = True

                            if(len(distances) > 0):

                                # je reecupere l'indice du min
                                min_index = np.argmin(distances)

                                matching_obj = frames[j][min_index]

                                # Si j'arrive a matcher j'arrete de rechercher dans les frames precedentes
                                if(np.min(distances) <= seuil):

                                    obj1.set_id(matching_obj.get_id())

                                    found = True
                                    break

                                seuil += 1

                    # si je trouve pas de match
                    if(not found):
                        # soit c est l'objet avec un BB du meme ratio (bien sur selon la dist aussi)
                        if(found_ratio and frame_obj[2] <= 2*seuil):

                            obj1.set_id(frames[frame_obj[0]]
                                        [frame_obj[1]].get_id())

                        # ou bien c'est un nouvel objet
                        else:
                            obj1.set_id(ID_OBJ)
                            ID_OBJ += 1

    # elimine les BB du cartable qu'on peut pas eliminer au debut
    # grace au seuillage
    processFrames(frames)

    # lance la video
    displayVideo(frames, video, video_speed)

    return frames


def displayVideo(frames, video, video_speed):

    # pour chaque frame on dessine les objets qu'elle contient
    for i in range(len(frames)):

        for detected in frames[i]:

            cv2.rectangle(video[i], (detected.get_x(), detected.get_y()), (detected.get_x()+detected.get_w(), detected.get_y() + detected.get_h()),
                          detected.get_color(), 2)

            cv2.putText(video[i], str(detected.get_id()), (detected.get_x(),  detected.get_y() - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.5,
                        detected.get_color(), 1, cv2.LINE_AA)

        # alors ça c est pour draw la trajectoire
        # il commence a dessiner a partir de la frame i-100 ( pour clean le chemin avec le temps)
        # si tu le mets a 0 ça va garder le path (give it a try, it s cool)
        drawPath(video, frames, i-100, i+1)

        cv2.imwrite("bounding_boxes/image" + str(i)+".jpg", video[i])

        cv2.imshow('video with bbs', video[i])

        cv2.waitKey(video_speed)


def drawPath(video, frames, minVal, maxVal):

    for k in range(minVal, maxVal):

        if(k > 0):
            # pour chaque objet d'une frame on dessine le centroid
            for detected2 in frames[k]:

                cv2.circle(video[maxVal-1], (int(detected2.get_centroid()[0]), int(detected2.get_centroid()[
                    1])), 2, detected2.get_color())


def splitObject(obj):

    obj1 = Object(obj.get_id(), [obj.get_x(), obj.get_y(), int(
        obj.get_w()/2), obj.get_h()])

    obj2 = Object(obj.get_id(), [obj.get_x()+int(obj.get_w()/2),
                                 obj.get_y(), int(obj.get_w()/2), obj.get_h()])
    return obj1, obj2


def extract_objects(frame):

    # returns a list of contours
    contours, _ = cv2.findContours(
        frame, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    # this will contain the frames's objects
    objects = []
    seuilMinimal = 30

    '''pour chaque contour on va créer un bounding box'''
    for k, c in enumerate(contours):

        x, y, w, h = cv2.boundingRect(c)
        # pour contourner le probleme des faux contours
        if (w > seuilMinimal and h > seuilMinimal):

            obj = Object(-1, [x, y, w, h])

            # pour diviser les trop grands BB en 2 objets
            if(w > 100 and (160 < h < 174 or h > 175)):

                obj1, obj2 = splitObject(obj)
                objects.append(obj1)
                objects.append(obj2)

            else:

                # pour eliminer la plupart des BB du cartable
                if(obj.get_ratio() != 1.04 and obj.get_ratio() != 1.02):
                    '''creating my object'''
                    objects.append(obj)

    return objects


# Main Function
def main():

    video_speed = 50

    my_objects = []
    # video to display
    video = load_images(video_path)
    # ground truth
    images = load_images(images_path)

    print("Waiting for Matching...")
    frames = matching(video, images, video_speed)

    for i in range(len(frames)):
        print("frame ", i, "\n", frames[i])


if __name__ == "__main__":
    main()
