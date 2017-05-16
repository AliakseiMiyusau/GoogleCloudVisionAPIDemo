import io
from google.cloud import vision
from google.cloud.vision.feature import Feature
from google.cloud.vision.feature import FeatureTypes
import cv2

image1 = "/home/aliaksei/Documents/SemanticTube/test.jpg"
image2 = "/home/aliaksei/Documents/SemanticTube/test2.jpg"

image_list = [image1, image2]

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

def main():


    result = process_batch(image_list)
    i = 0
    for image in result:
        img = cv2.imread(image_list[i])
        print image_list[i]
        for face in image.faces:
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

            cv2.circle(img, left_eye, 3, (255, 255, 255))
            cv2.circle(img, right_eye, 3, (255, 255, 255))    
            cv2.circle(img, chin, 3, (255, 255, 255))    
            cv2.circle(img, left_eyebrow, 3, (255, 255, 255))    
            cv2.circle(img, right_eyebrow, 3, (255, 255, 255))
            cv2.circle(img, mouth, 3, (255, 255, 255))
        i+=1
        cv2.imwrite("output" + str(i) + ".jpg", img)
            
if __name__ == "__main__":
    main()
