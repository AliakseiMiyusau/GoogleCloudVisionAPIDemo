import io
from google.cloud import vision
from google.cloud.vision.feature import Feature
from google.cloud.vision.feature import FeatureTypes
import cv2
import math
import glob
import os

FRAMES_DIR = "/home/aliaksei/Documents/faces/frames/"
FRAME_EXTENSION = "bmp"
PROCESSED_FRAMES_DIR = "/home/aliaksei/Documents/faces/processed_frames/"
BATCH_SIZE_LIMIT = 10485760


def distance(point1, point2):
    return math.sqrt((point1[0]-point2[0])**2 + (point1[1]-point2[1])**2)


def process_batch(image_list):
 
    client = vision.Client()
    batch = client.batch()
    face_feature = Feature(FeatureTypes.FACE_DETECTION, len(image_list))
    for image in image_list:
        image_data = io.open(image, 'rb')
        content = image_data.read()
        im = client.image(content=content)
        batch.add_image(im, [face_feature])
    results = batch.detect()
    return results


def blur_face(image, top_left, bot_right):
    face = image[top_left[1]:bot_right[1], top_left[0]:bot_right[0]]
    blurred_face = cv2.GaussianBlur(face,(23, 23), 30)
    image[top_left[1]:bot_right[1], top_left[0]:bot_right[0]] = blurred_face

    return image


#STUB METHOD
def simple_classifier():
    return True


def process_video_frames(frame_list):

    result = process_batch(frame_list)
    i = 0
    for image in result:
        idx = frame_list[i].split("/")[-1].split(".")[0]
        img = cv2.imread(frame_list[i])
        print frame_list[i]


        for face in image.faces:

            top_left_face_point = (face.bounds.vertices[0].x_coordinate, face.bounds.vertices[0].y_coordinate)
            bot_right_face_point = (face.bounds.vertices[2].x_coordinate, face.bounds.vertices[2].y_coordinate)

            left_eye = (int(face.landmarks.left_eye.position.x_coordinate), int(face.landmarks.left_eye.position.y_coordinate))
            right_eye = (int(face.landmarks.right_eye.position.x_coordinate), int(face.landmarks.right_eye.position.y_coordinate))
            chin = (int(face.landmarks.chin_gnathion.position.x_coordinate), int(face.landmarks.chin_gnathion.position.y_coordinate))
            left_eyebrow = (int(face.landmarks.left_eyebrow_upper_midpoint.position.x_coordinate), int(face.landmarks.left_eyebrow_upper_midpoint.position.y_coordinate))
            right_eyebrow = (int(face.landmarks.right_eyebrow_upper_midpoint.position.x_coordinate), int(face.landmarks.right_eyebrow_upper_midpoint.position.y_coordinate))
            mouth = (int(face.landmarks.mouth_center.position.x_coordinate), int(face.landmarks.mouth_center.position.y_coordinate))

            print("Top Left Face Point: " + str(top_left_face_point))
            print("Bot Right Face Point: " + str(bot_right_face_point))

            print("Left Eye: " + str(left_eye))
            print("Right Eye: " + str(right_eye))
            print("Chin: " + str(chin))
            print("Left Eyebrow: " + str(left_eyebrow))
            print("Right Eyebrow: " + str(right_eyebrow))
            print("Mouth: " + str(mouth))

            if simple_classifier():
                img = blur_face(img, top_left_face_point, bot_right_face_point)

            cv2.circle(img, left_eye, 3, (255, 255, 255))
            cv2.circle(img, right_eye, 3, (255, 255, 255))    
            cv2.circle(img, chin, 3, (255, 255, 255))    
            cv2.circle(img, left_eyebrow, 3, (255, 255, 255))    
            cv2.circle(img, right_eyebrow, 3, (255, 255, 255))
            cv2.circle(img, mouth, 3, (255, 255, 255))
            #cv2.rectangle(img, top_left_face_point, bot_right_face_point, (255, 255, 0))
        if not os.path.exists(PROCESSED_FRAMES_DIR):
            os.makedirs(PROCESSED_FRAMES_DIR)
        cv2.imwrite("processed_frames/" + idx + ".bmp", img)
        i+=1


def batch_video_frames(video_frames, BATCH_SIZE_LIMIT, FRAMES_PER_BATCH_LIMIT):
    #assuming all frames have same byte size

    frame_size = cv2.imread(video_frames[0]).shape
    frame_byte_size = frame_size[0]*frame_size[1]*frame_size[2]
    frames_per_batch = min(FRAMES_PER_BATCH_LIMIT, BATCH_SIZE_LIMIT / frame_byte_size)
    batch_number = len(video_frames) * frame_byte_size / BATCH_SIZE_LIMIT
    frame_batches = [[] for _ in range(batch_number)]
    
    i = 0
    for i in range(batch_number):
        frame_batches[i] = video_frames[i*frames_per_batch:(i+1)*frames_per_batch]
        i = i+1
    print frame_batches
    return frame_batches
    


def main():

    video_frames = glob.glob(FRAMES_DIR + "*." + FRAME_EXTENSION)
    frame_batches = batch_video_frames(video_frames, BATCH_SIZE_LIMIT, FRAMES_PER_BATCH)
    for batch in frame_batches:
        process_video_frames(batch)
            
if __name__ == "__main__":
    main()
