import logging
from typing import Dict, Tuple, Any
import os
import cv2
import dlib
import numpy as np
from imutils import face_utils

from modules.base import BaseModelTF, InputFormatException
from modules.utils import download_model

LOGGER = logging.getLogger("model")
LOGGER.setLevel(logging.INFO)


class FaceLandmark(BaseModelTF):
    def __init__(self, service_host: str, service_port: int, version: int = None,
                 image_size: int = 128,
                 options: list = [], timeout: int = 5, debug: bool = False):
        super(FaceLandmark, self).__init__(service_host, service_port, model_name='landmark', version=version,
                                           options=options, timeout=timeout, debug=debug)
        self.image_size = image_size

        # dlib_model_url = download_model(
        #     'https://github.com/tienthienhd/CenterNet-1/releases/download/face_landmark/shape_predictor_68_face_landmarks.dat')
        #
        dir_path = os.path.dirname(os.path.realpath(__file__))
        dlib_model_url = os.path.join(dir_path, 'liveness', 'shape_predictor_68_face_landmarks.dat')
        self._dlib_model = dlib.shape_predictor(dlib_model_url)

    def _pre_process(self, x: np.ndarray) -> Tuple[Dict[Any, np.ndarray], dict]:
        if x is None or not isinstance(x, np.ndarray):
            raise InputFormatException("{}: {}".format(self.__class__.__name__, "input must be not None."))
        im_height, im_width = x.shape[:2]
        img = cv2.resize(x, (self.image_size, self.image_size))
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img = np.expand_dims(img, axis=0)
        return {
                   self._input_signature_key[0][0]: img
               }, {'im_height': im_height, 'im_width': im_width}

    def _post_process(self, predict_response: dict, params_process: dict) -> Dict:
        landmark = predict_response['output']
        marks = np.array(landmark).flatten()[:136]
        marks = np.reshape(marks, (-1, 2))
        return {'mark': marks}

    def _normalize_output(self, post_processed: dict, params: dict):
        return post_processed['mark']

    def get_dlib(self, img, bbox):
        rect = dlib.rectangle(bbox[0], bbox[1], bbox[2], bbox[3])
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        shape = self._dlib_model(gray, rect)
        shape = face_utils.shape_to_np(shape)
        shape = shape.astype('double')
        return shape


def get_square_box(box):
    """Get a square box out of the given box, by expanding it."""
    left_x = box[0]
    top_y = box[1]
    right_x = box[2]
    bottom_y = box[3]

    box_width = right_x - left_x
    box_height = bottom_y - top_y

    # Check if box is already a square. If not, make it a square.
    diff = box_height - box_width
    delta = int(abs(diff) / 2)

    if diff == 0:  # Already a square.
        return box
    elif diff > 0:  # Height > width, a slim box.
        left_x -= delta
        right_x += delta
        if diff % 2 == 1:
            right_x += 1
    else:  # Width > height, a short box.
        top_y -= delta
        bottom_y += delta
        if diff % 2 == 1:
            bottom_y += 1

    # Make sure box is always square.
    assert ((right_x - left_x) == (bottom_y - top_y)), 'Box is not square.'

    return [left_x, top_y, right_x, bottom_y]
