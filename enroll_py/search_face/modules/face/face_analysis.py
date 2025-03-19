import time
from collections import namedtuple

import cv2

from modules.face.face_detection import FastFaceDetection, AccuracyFaceDetection, PCNFaceDetector
from modules.face.face_embedding import FaceEmbedding
from modules.utils import compute_sim

Face = namedtuple("Face", ['det_score', 'image', 'box', 'landmark', 'embedding', 'rotated_angle'])
Face.__new__.__defaults__ = (None,) * len(Face._fields)


class NotSupportException(Exception):
    pass


class FaceAnalysis:
    def __init__(self, service_host, service_port, options=[], timeout=5, debug=False):
        self.fast_face_detector = FastFaceDetection(service_host=service_host,
                                                    service_port=service_port,
                                                    image_size=320,
                                                    options=options,
                                                    timeout=timeout,
                                                    debug=debug)
        self.acc_face_detector = AccuracyFaceDetection(service_host=service_host,
                                                       service_port=service_port,
                                                       image_size=320,
                                                       options=options,
                                                       timeout=timeout,
                                                       debug=debug)
        self.pcn_face_detector = PCNFaceDetector()

        self.face_embedding = FaceEmbedding(service_host=service_host,
                                            service_port=service_port,
                                            image_size=(112, 112),
                                            # keep_prob=1.0,
                                            options=options,
                                            timeout=timeout,
                                            debug=debug)

    def get(self, img, size_threshold=0, max_faces=10, receive_mode="meta", fast=True, mode="detect",
            face_image_size=112, crop_mode='arcface'):
        assert img is not None, "Input image is None!"
        if mode == 'embed_only':
            embeddings = self.face_embedding.predict(img)
            return embeddings
            # faces = []
            # for embedding in embeddings:
            #     face = Face(embedding=embedding)
            #     faces.append(face)
            # return faces

        face_detector = self.pcn_face_detector
        bboxes, landmarks, confidences, images = face_detector.predict(img, {"size_threshold": size_threshold,
                                                                             "max_faces": max_faces,
                                                                             "receive_mode": receive_mode,
                                                                             "fast": fast,
                                                                             "face_image_size": face_image_size,
                                                                             "crop_mode": crop_mode})
        if bboxes is None or len(bboxes) == 0:
            face_detector = self.fast_face_detector
            if not fast:
                face_detector = self.acc_face_detector
            bboxes, landmarks, confidences, images = face_detector.predict(img,
                                                                           {"size_threshold": size_threshold,
                                                                            "max_faces": max_faces,
                                                                            "receive_mode": receive_mode,
                                                                            "fast": fast,
                                                                            "face_image_size": face_image_size,
                                                                            "crop_mode": crop_mode})
        if bboxes is None or len(bboxes) == 0:
            return []

        if mode == 'embed':
            if receive_mode == 'meta':
                raise NotSupportException(f"When embedding face, outputs of detection must be image or all")
            elif receive_mode == 'image':
                embeddings = self.face_embedding.predict(images)
        else:
            embeddings = [None] * len(bboxes)

        faces = []
        for i in range(len(bboxes)):
            box = bboxes[i]
            landmark = landmarks[i]
            det_score = confidences[i]
            face_img = images[i]
            embedding = embeddings[i]
            face = Face(det_score=det_score, image=face_img, box=box, landmark=landmark, embedding=embedding,
                        rotated_angle=0)
            faces.append(face)
        return faces


if __name__ == '__main__':

    img = cv2.imread("/home/thiennt/Desktop/0.jpg")
    if img is None:
        exit(0)
    fa = FaceAnalysis("localhost", 8500)

    start_time = time.time()
    faces = fa.get(img, receive_mode="image", fast=False, mode='embed')
    print("time inference:", time.time() - start_time)
    if len(faces) > 1:
        sim = compute_sim(faces[0].embedding, faces[1].embedding, new_range=True)
        print("Sim:", sim)

    cam = cv2.VideoCapture(0)
    start_time = time.time()
    i = 0
    while True:
        _, frame = cam.read()
        # frame = cv2.flip(frame, 1)
        # frame = cv2.flip(frame, 0)
        if frame is None:
            print("no cam input")

        origin = frame.copy()
        # faces = fd.get(frame, receive_mode="image")
        faces = fa.get(frame, receive_mode="image", fast=False, mode='embed')
        if len(faces) > 2:
            sim = compute_sim(faces[0].embedding, faces[1].embedding, new_range=True)
            i += 1
            print(f"{i} Sim: {sim}")

        # for i in range(len(faces)):
        #     cv2.imshow(f"face_{i}", faces[i])

        # calculate fps
        fps_str = "FPS: %.2f" % (1 / (time.time() - start_time))
        start_time = time.time()
        cv2.putText(frame, fps_str, (25, 25),
                    cv2.FONT_HERSHEY_DUPLEX, 0.75, (0, 255, 0), 2)

        cv2.imshow("frame", frame)
        if cv2.waitKey(1) == ord('q'):
            exit()
