from typing import Dict, Tuple, Any, List, Optional

import cv2
import numpy as np

from modules.base import InputFormatException
from modules.base.classifier import Classifier
from modules.face.face_detection import FaceDetection


class FaceEmotion(Classifier):
    def __init__(self, service_host: str, service_port: int, model_name: str = 'emotion', version: int = 1,
                 image_size: int = 64,
                 threshold: float = 0.5, options: Optional[List] = None, timeout: int = 5, debug: bool = False):
        super(FaceEmotion, self).__init__(service_host, service_port, model_name, version=version,
                                          model_input_size=image_size, threshold=threshold,
                                          classes=['angry', 'disgust', 'fear', 'happy', 'sad', 'surprise',
                                                   'neutral'], options=options,
                                          timeout=timeout, debug=debug)

    def _pre_process(self, x: list) -> Tuple[Dict[Any, np.ndarray], dict]:
        """
        :param x: List of image
        :return: input dict for inference and dict of process parameter
        """
        if x is None or not isinstance(x, list):
            raise InputFormatException("{}: {}".format(self.__class__.__name__, "input must be list."))
        if len(x) == 0:
            raise InputFormatException("{}: {}".format(self.__class__.__name__, "input must be not empty."))
        try:
            imgs = []
            for im in x:
                im = cv2.cvtColor(im, cv2.COLOR_BGR2GRAY)
                im = np.expand_dims(im, axis=-1)
                imgs.append(im)
            imgs = cv2.dnn.blobFromImages(imgs, scalefactor=self.scale_factor,
                                          size=(self.model_input_size, self.model_input_size),
                                          mean=[0, 0, 0], swapRB=self.swapRB, crop=False)
            imgs = np.transpose(imgs, (0, 2, 3, 1))
            imgs = imgs - 0.5
            imgs = imgs * 2.0
        except Exception as e:
            self.LOGGER.exception(e)
            raise InputFormatException("{}: {}".format(self.__class__.__name__, "input is wrong format"))

        return {self._input_signature_key[0][0]: imgs}, {}


class FaceEmotionV2(Classifier):
    def __init__(self, service_host: str, service_port: int, model_name: str = 'emotion', version: int = 2,
                 image_size: int = 48, threshold: float = 0.5, options: Optional[List] = None, timeout: int = 5,
                 debug: bool = False):
        super(FaceEmotionV2, self).__init__(service_host, service_port, model_name, version=version,
                                            model_input_size=image_size, threshold=threshold,
                                            scale_factor=1.0,
                                            classes=['neutral', 'happy'], options=options,
                                            timeout=timeout, debug=debug)

    def _pre_process(self, x: list) -> Tuple[Dict[Any, np.ndarray], dict]:
        """
        :param x: List of image
        :return: input dict for inference and dict of process parameter
        """
        if x is None or not isinstance(x, list):
            raise InputFormatException("{}: {}".format(self.__class__.__name__, "input must be list."))
        if len(x) == 0:
            raise InputFormatException("{}: {}".format(self.__class__.__name__, "input must be not empty."))
        try:
            imgs = []
            for im in x:
                im = cv2.cvtColor(im, cv2.COLOR_BGR2GRAY)
                im = np.expand_dims(im, axis=-1)
                imgs.append(im)
            imgs = cv2.dnn.blobFromImages(imgs, scalefactor=self.scale_factor,
                                          size=(self.model_input_size, self.model_input_size),
                                          mean=[0, 0, 0], swapRB=self.swapRB, crop=False)
            imgs = np.transpose(imgs, (0, 2, 3, 1))
        except Exception as e:
            self.LOGGER.exception(e)
            raise InputFormatException("{}: {}".format(self.__class__.__name__, "input is wrong format"))

        return {self._input_signature_key[0][0]: imgs}, {}


if __name__ == '__main__':
    face_emotion = FaceEmotionV2('172.16.1.36', 8500, 'emotion', version=2, timeout=30)
    face_detection = FaceDetection('172.16.1.36', 8500, 'retinaface_mbnet', image_size=120, debug=True)

    import glob

    for i, file in enumerate(glob.glob('/media/data_it/Data_set/database_image/face/smile/Anh_smile/Fail/*.jpg')):
        img = cv2.imread(file)
        faces = face_detection.predict(img, {'receive_mode': 'image'})

        boxes, landmarkds, confs, face_imgs = faces
        if face_imgs:
            emotion_pred = face_emotion.predict(face_imgs)
            if emotion_pred['label'][0] == 'neutral':
                print(i, file, emotion_pred)

    exit(0)
    cv2.namedWindow('window_frame')
    video_capture = cv2.VideoCapture(0)
    i = 1
    while True:
        bgr_image = video_capture.read()[1]
        faces = face_detection.predict(bgr_image, {'receive_mode': 'image'})

        boxes, landmarkds, confs, face_imgs = faces
        if face_imgs:
            emotion_pred = face_emotion.predict(face_imgs)
            print(emotion_pred)
            # if emotion_pred['label'][0] == 'happy':
            #     cv2.imwrite(f'/media/thiennt/projects/face_lvt/application/test/data/liveness/smile/{i}.jpg', bgr_image)
            #     i += 1

        # for face_coordinates in faces:
        #
        #     x1, x2, y1, y2 = apply_offsets(face_coordinates, emotion_offsets)
        #     gray_face = gray_image[y1:y2, x1:x2]
        #     try:
        #         gray_face = cv2.resize(gray_face, (emotion_target_size))
        #     except:
        #         continue
        #
        #     gray_face = preprocess_input(gray_face, True)
        #     gray_face = np.expand_dims(gray_face, 0)
        #     gray_face = np.expand_dims(gray_face, -1)
        #     emotion_prediction = emotion_classifier.predict(gray_face)
        #     emotion_probability = np.max(emotion_prediction)
        #     emotion_label_arg = np.argmax(emotion_prediction)
        #     emotion_text = emotion_labels[emotion_label_arg]
        #     emotion_window.append(emotion_text)
        #
        #     if len(emotion_window) > frame_window:
        #         emotion_window.pop(0)
        #     try:
        #         emotion_mode = mode(emotion_window)
        #     except:
        #         continue
        #
        #     if emotion_text == 'angry':
        #         color = emotion_probability * np.asarray((255, 0, 0))
        #     elif emotion_text == 'sad':
        #         color = emotion_probability * np.asarray((0, 0, 255))
        #     elif emotion_text == 'happy':
        #         color = emotion_probability * np.asarray((255, 255, 0))
        #     elif emotion_text == 'surprise':
        #         color = emotion_probability * np.asarray((0, 255, 255))
        #     else:
        #         color = emotion_probability * np.asarray((0, 255, 0))
        #
        #     color = color.astype(int)
        #     color = color.tolist()
        #
        #     draw_bounding_box(face_coordinates, rgb_image, color)
        #     draw_text(face_coordinates, rgb_image, emotion_mode,
        #               color, 0, -45, 1, 1)
        #
        # bgr_image = cv2.cvtColor(rgb_image, cv2.COLOR_RGB2BGR)
        cv2.imshow('window_frame', bgr_image)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    video_capture.release()
    cv2.destroyAllWindows()
