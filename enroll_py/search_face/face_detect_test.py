import cv2
import sys
sys.path.insert(0, "./modules/inference")
import time
from modules.face.face_detection import PCNFaceDetector

if __name__ == '__main__':
    # time.sleep(120)
    # model = FaceDetection('172.16.1.36', 8500, 'retinaface_mbnet', image_size=480, debug=True)
    model = PCNFaceDetector()

    i1 = cv2.imread("1.jpg")


    start_time = time.time()
    a = model.predict(i1)
    end_time = time.time()
    print("run time:", end_time - start_time)
    print(a)

    cv2.imwrite("output.jpg", a[3][0])
    time.sleep(120)


