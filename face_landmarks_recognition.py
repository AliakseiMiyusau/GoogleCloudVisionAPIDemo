import io
from google.cloud import vision
import cv2

photo_file = "/home/aliaksei/Documents/SemanticTube/test.jpg"

def process_one(photo_file):
 
    vision_client = vision.Client()
    with io.open(photo_file, 'rb') as image_file:
        content = image_file.read()

    image = vision_client.image(content=content)
    faces = image.detect_faces()
    return faces

def main():


    faces = process_one(photo_file)
    for face in faces:
        left_eye = (int(face.landmarks.left_eye.position.x_coordinate), int(face.landmarks.left_eye.position.y_coordinate))
        right_eye = (int(face.landmarks.right_eye.position.x_coordinate), int(face.landmarks.right_eye.position.y_coordinate))
        chin = (int(face.landmarks.chin_gnathion.position.x_coordinate), int(face.landmarks.chin_gnathion.position.y_coordinate))
        left_eyebrow = (int(face.landmarks.left_eyebrow_upper_midpoint.position.x_coordinate), int(face.landmarks.left_eyebrow_upper_midpoint.position.y_coordinate))
        right_eyebrow = (int(face.landmarks.right_eyebrow_upper_midpoint.position.x_coordinate), int(face.landmarks.right_eyebrow_upper_midpoint.position.y_coordinate))
        mouth = (int(face.landmarks.mouth_center.position.x_coordinate), int(face.landmarks.mouth_center.position.y_coordinate))

        print("Left Eye: " + str(left_eye))
        print("Right Eye: " + str(right_eye))
        print("Chin: " + str(chin))
        print("Left Eyebrow: " + str(left_eyebrow))
        print("Right Eyebrow: " + str(right_eyebrow))
        print("Mouth: " + str(mouth))

    img = cv2.imread(photo_file)
    cv2.circle(img, left_eye, 3, (255, 255, 255))
    cv2.circle(img, right_eye, 3, (255, 255, 255))    
    cv2.circle(img, chin, 3, (255, 255, 255))    
    cv2.circle(img, left_eyebrow, 3, (255, 255, 255))    
    cv2.circle(img, right_eyebrow, 3, (255, 255, 255))
    cv2.circle(img, mouth, 3, (255, 255, 255))
    cv2.imwrite("output.jpg", img)
if __name__ == "__main__":
    main()
