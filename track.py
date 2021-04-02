import os
import cv2
import numpy as np
import math
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
            img = cv2.imread(os.path.join(path, filename),cv2.IMREAD_GRAYSCALE)
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
def matching(video, ground):
    frames = []

    ID_OBJ = 1

    # for each frame i extract objects
    for image in ground:
        frames.append(extract_objects(image))

    # i gave ids to 1st frame objects
    for detected_obj in frames[0]:
        detected_obj.set_id(ID_OBJ)
        ID_OBJ += 1

    # pour chaque frame de la video
    for i in range(1, len(frames)):

        # print("FRAME", i)

        # pour chaque frame i
        for obj1 in frames[i]:
            distances = np.zeros(len(frames[i - 1]))

            # je calcule la distance entre les centroid des objets de ma frame actuelle "i"
            # avec ceux des objets de la frame precedente "i-1"

            for k, obj2 in enumerate(frames[i - 1]):
                # print("Objet", obj2.get_id())

                distances[k] = math.sqrt(sum((px - qx) ** 2.0 for px, qx in zip(obj1.get_centroid(), obj2.get_centroid())))
                # distances[k] = math.dist(obj1.get_centroid(), obj2.get_centroid())

                # print("distance = ", distances[k])

            # je reecupere l'indice de la distance minimale ça correspond aussi
            # a l indice de l'objet (probablement recherché) dans la frame i - 1
            min_index = np.argmin(distances)

            # print("Min dist = ", np.min(distances))

            # s'il y a matching grace au seuil
            # l'objet a le meme id et couleur que l'objet trouvé
            if np.min(distances) <= 15:

                matching_obj = frames[i - 1][min_index]

                obj1.set_id(matching_obj.get_id())

            else:
                # si je ne trouve pas le match
                # je cherche dans les 28 frames precedentes
                #  ( au max 32 frame psk sinon la distance devient trop grande )
                found = False
                seuil = 45
                for j in range(i - 2, i - 30, -1):

                    distances = np.zeros(len(frames[j]))

                    for v, obj2 in enumerate(frames[j]):
                        # print(obj2)

                        distances[v] = math.sqrt(
                            sum((px - qx) ** 2.0 for px, qx in zip(obj1.get_centroid(), obj2.get_centroid())))
                    # distances[v] = math.dist( obj1.get_centroid(), obj2.get_centroid())

                        # print("distance = ", distances[v])

                    # je reecupere l'indice du min
                    min_index = np.argmin(distances)

                    # print("Min dist = ", np.min(distances))

                    # Si j'arrive a matcher j'arrete de rechercher dans les frames precedentes
                    if np.min(distances) <= seuil:
                        matching_obj = frames[j][min_index]

                        obj1.set_id(matching_obj.get_id())

                        found = True
                        break

                    seuil += 1

                # si je trouve pas de match c est un nouvel objet
                if not found:
                    obj1.set_id(ID_OBJ)
                    ID_OBJ += 1

    displayVideo(frames, video)

    return frames


def displayVideo(frames, video):
    # pour chaque frame on dessine les objets qu'elle contient
    for i in range(len(frames)):

        for detected in frames[i]:
            cv2.rectangle(video[i], (detected.get_x(), detected.get_y()),
                          (detected.get_x() + detected.get_w(), detected.get_y() + detected.get_h()),
                          detected.get_color(), 2)

            cv2.putText(video[i], str(detected.get_id()), (detected.get_x(), detected.get_y() - 5),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5,
                        detected.get_color(), 1, cv2.LINE_AA)

        # alors ça c est pour draw la trajectoire
        # il commence a dessiner a partir de la frame i-100 ( pour clean le chemin avec le temps)
        # si tu le mets a 0 ça va garder le path (give it a try, it s cool)
        drawPath(video, frames, i - 100, i + 1)

        # cv2.imwrite("BB/image" + str(i)+".jpg", video[i])

        cv2.imshow('video with bbs', video[i])

        cv2.waitKey(50)


def drawPath(video, frames, minVal, maxVal):
    for k in range(minVal, maxVal):

        if (k > 0):
            # pour chaque objet d'une frame on dessine le centroid
            for detected2 in frames[k]:
                cv2.circle(video[maxVal - 1], (int(detected2.get_centroid()[0]), int(detected2.get_centroid()[
                                                                                         1])), 2, detected2.get_color())


def extract_objects(frame):
    # OLD OPENCV VERSION, RANIA JUST DELETE ret
    ret, contours, hierarchy = cv2.findContours(
        frame, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    # this will contain the frames's objects
    objects = []

    '''pour chaque contour on va créer un bounding box'''
    for k, c in enumerate(contours):

        x, y, w, h = cv2.boundingRect(c)
        # pour contourner le probleme des petites windows
        if w > 30 and h > 30:
            '''creating my object'''
            objects.append(Object(-1, [x, y, w, h]))

    return objects


# Main Function
def main():
    my_objects = []
    # video to display
    video = load_images(video_path)
    # ground truth
    images = load_images(images_path)

    print("Waiting for Matching...")
    frames = matching(video, images)


if __name__ == "__main__":
    main()
