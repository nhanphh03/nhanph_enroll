from typing import Tuple, Dict, Any
import cv2
import numpy as np

from modules.base import BaseModelTF, InputFormatException, ResponseFormatException


class Classifier(BaseModelTF):
    def __init__(self, service_host: str, service_port: int, model_name: str, version: int = None,
                 model_input_size=(256, 256),
                 threshold: float = 0.7,
                 classes: list = None,
                 unknown_class='unknown',
                 scale_factor: float = 0.00392156862745098,
                 swap_rb: int = 1,
                 options: list = None,
                 timeout: int = 5,
                 debug=False):
        super(Classifier, self).__init__(service_host, service_port, model_name, version, options, timeout, debug)
        self.model_input_size = model_input_size
        self.threshold = threshold
        self.scale_factor = scale_factor
        self.swapRB = swap_rb
        self._debug = debug
        self.classes = classes
        self.unknown_class = unknown_class

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
            imgs = cv2.dnn.blobFromImages(x, scalefactor=self.scale_factor,
                                          size=(self.model_input_size, self.model_input_size),
                                          mean=[0, 0, 0], swapRB=self.swapRB, crop=False)
            imgs = np.transpose(imgs, (0, 2, 3, 1))
        except Exception as e:
            self.LOGGER.exception(e)
            raise InputFormatException("{}: {}".format(self.__class__.__name__, "input is wrong format"))

        return {self._input_signature_key[0][0]: imgs}, {}

    def _post_process(self, predict_response: dict, params_process: dict) -> Dict:
        """
        post process response from server model
        :param predict_response: dict of output key
        :param params_process: dict of param from preprocess
        :return:
        """
        if len(self._output_signature_key) == 0:
            raise ResponseFormatException("{}: {}".format(self.__class__.__name__, "Response metadata not contain key"))
        if self._output_signature_key[0][0] not in predict_response:
            raise ResponseFormatException("{}: {}".format(self.__class__.__name__, "Response not contain meta key"))
        outputs = predict_response[self._output_signature_key[0][0]]
        idx_max_scores = np.argmax(outputs, axis=-1).tolist()
        max_scores = np.amax(outputs, axis=-1).tolist()
        return {'idx': idx_max_scores, 'score': max_scores}

    def _normalize_output(self, post_processed: dict, params: dict):
        """
        :param post_processed: dict of idx and score
        :param params: parameter to format
        :return: List of class name
        """
        labels = []
        for i in range(len(post_processed['idx'])):
            idx = post_processed['idx'][i]
            score = post_processed['score'][i]

            if score < self.threshold:
                labels.append(self.unknown_class)
                post_processed['idx'][i] = -1
            else:
                labels.append(self.classes[idx])

        return {
            'label': labels,
            'score': post_processed['score'],
            'idx': post_processed['idx']
        }


if __name__ == '__main__':
    import glob

    model_name = 'test'
    model_input_size = (160, 160)
    classes = ['real', 'fake']
    input_folder = 'twes/*.jpg'
    max_test = 10
    model = Classifier("172.16.1.36", 8500, model_name, model_input_size=model_input_size, threshold=0.7,
                       classes=classes, unknown_class='other', scale_factor=1, swap_rb=1)
    list_path = list(
        glob.glob(input_folder))

    imgs = []
    for path in list_path:
        img = cv2.imread(path)
        if img is not None:
            imgs.append(img)
            if len(imgs) == max_test:
                break

    result = model.predict(imgs)

    report = {}
    for path, score, label, index in zip(list_path, result['score'], result['label'], result['idx']):
        print(path, index, label, score)
        if label in report:
            report[label]['count'] += 1
            report[label]['total_score'] += score
        else:
            report[label] = {
                'count': 1,
                'total_score': score
            }

    print("\n\n=================================== Report ======================================")
    for k, v in report.items():
        print(f'{k:<15}: count: {v["count"]:<5}    avg_score: {v["total_score"]/v["count"]:<10}',)
    print("=================================================================================")
