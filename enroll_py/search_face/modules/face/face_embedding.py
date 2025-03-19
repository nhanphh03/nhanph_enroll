from typing import Dict, Tuple, Any, List

import cv2
import numpy as np

from modules.base import BaseModelTF, InputFormatException


class FaceEmbedding(BaseModelTF):
    def __init__(self, service_host: str, service_port: int, version: int = None,
                 image_size: tuple = (112, 112),
                 # keep_prob=0.4,
                 options: list = [], timeout: int = 5, debug: bool = False):
        super(FaceEmbedding, self).__init__(service_host, service_port, model_name='arcface', version=version,
                                            options=options, timeout=timeout, debug=debug)

        self.image_size = image_size
        self._debug = debug
        if self._version == 1:
            self._input_mean = 0.0
            self._input_std = 1.0
        if self._version == 2:
            self._input_mean = 127.5
            self._input_std = 127.5

    def _pre_process(self, x: List[np.ndarray]) -> Tuple[Dict[Any, np.ndarray], dict]:
        if x is None or not isinstance(x, List):
            raise InputFormatException("{}: {}".format(self.__class__.__name__, "input must be not None."))

        imgs = cv2.dnn.blobFromImages(x, 1.0 / self._input_std, self.image_size,
                                      (self._input_mean, self._input_mean, self._input_mean), swapRB=True)

        if self._version == 1:
            imgs = np.transpose(imgs, (0, 2, 3, 1))
        return {self._input_signature_key[0][0]: imgs}, {}

    def _post_process(self, predict_response: dict, params_process: dict) -> Dict:
        embeddings = predict_response[self._output_signature_key[0][0]]
        embeddings = np.reshape(embeddings, (-1, 512))
        return {'embeddings': embeddings}

    def _normalize_output(self, post_processed: dict, params: dict):
        return post_processed['embeddings']
