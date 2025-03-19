import logging
from typing import Dict, Tuple, Any, List

import cv2
import numpy as np

from modules.base import BaseModelTF, InputFormatException, ResponseFormatException
from modules.base.classifier import Classifier


class FaceAntiSpoof(BaseModelTF):
    def __init__(self, service_host: str, service_port: int, model_name: str, version: int = None,
                 image_size: int = 260,
                 threshold: float = 0.63, options: list = [], timeout: int = 5, debug: bool = False):
        super(FaceAntiSpoof, self).__init__(service_host, service_port, model_name, version=version, options=options,
                                            timeout=timeout, debug=debug)
        self.image_size = image_size
        self.threshold = threshold

    def _pre_process(self, x: np.ndarray) -> Tuple[Dict[Any, np.ndarray], dict]:
        if x is None or not isinstance(x, np.ndarray):
            raise InputFormatException("{}: {}".format(self.__class__.__name__, "input must be not None."))
        x = cv2.cvtColor(x, cv2.COLOR_BGR2RGB)
        x = cv2.resize(x, (self.image_size, self.image_size))
        x = x / 255.0
        x = x.astype(np.float32)
        x = np.expand_dims(x, axis=0)

        return {
                   self._input_signature_key[0][0]: x
               }, {}

    def _post_process(self, predict_response: dict, params_process: dict) -> Dict:
        if len(self._output_signature_key) == 0:
            raise ResponseFormatException("{}: {}".format(self.__class__.__name__, "Response metadata not contain key"))
        if self._output_signature_key[0][0] not in predict_response:
            raise ResponseFormatException("{}: {}".format(self.__class__.__name__, "Response not contain meta key"))
        score = predict_response[self._output_signature_key[0][0]]

        fake_prob = float(score[0][0])
        real_prob = float(score[0][1])
        return {
            'fake_prob': fake_prob,
            'real_prob': real_prob,
            'is_fake': fake_prob > self.threshold
        }

    def _normalize_output(self, post_processed: dict, params: dict):
        return post_processed


class FaceAntiSpoofV2(BaseModelTF):
    def __init__(self, service_host: str, service_port: int, model_name: str, version: int = None,
                 image_size: int = 224,
                 threshold: float = 0.7, options: list = [], timeout: int = 5, debug: bool = False):
        super(FaceAntiSpoofV2, self).__init__(service_host, service_port, model_name, version=version, options=options,
                                              timeout=timeout, debug=debug)
        self.image_size = image_size
        self.threshold = threshold

    def _pre_process(self, x: np.ndarray) -> Tuple[Dict[Any, np.ndarray], dict]:
        if x is None or not isinstance(x, np.ndarray):
            raise InputFormatException("{}: {}".format(self.__class__.__name__, "input must be not None."))
        x = cv2.cvtColor(x, cv2.COLOR_BGR2RGB)
        x = cv2.resize(x, (self.image_size, self.image_size))
        x = np.expand_dims(x, axis=0)
        x = x.astype(np.float32)
        x = x / 255.0

        return {
                   self._input_signature_key[0][0]: x
               }, {}

    def _post_process(self, predict_response: dict, params_process: dict) -> Dict:
        score = predict_response['output']

        real_prob = float(score[0][0])
        fake_prob = float(score[0][1])
        return {
            'fake_prob': fake_prob,
            'real_prob': real_prob,
            'is_fake': fake_prob > self.threshold
        }

    def _normalize_output(self, post_processed: dict, params: dict):
        return post_processed


class FaceAntiSpoofV3(Classifier):
    def __init__(self, service_host: str, service_port: int, model_name: str, version: int = None,
                 image_size: int = 260,
                 threshold: float = 0.8, options: list = [], timeout: int = 5, debug: bool = False):
        super(FaceAntiSpoofV3, self).__init__(service_host, service_port, model_name, version=version,
                                              model_input_size=image_size, threshold=0.4,
                                              classes=['fake', 'real'], options=options,
                                              timeout=timeout, debug=debug)
        self.anti_spoof_thresh = threshold

    def _normalize_output(self, post_processed: dict, params: dict):
        preds = super()._normalize_output(post_processed, params)
        is_fake = []
        fake_prob = []
        for l, s in zip(preds['label'], preds['score']):
            if l == 'real':
                score = 1 - s
                if score >= self.anti_spoof_thresh:
                    is_fake.append(True)
                else:
                    is_fake.append(False)
                fake_prob.append(score)
            else:
                score = s
                if score >= self.anti_spoof_thresh:
                    is_fake.append(True)
                else:
                    is_fake.append(False)
                fake_prob.append(score)
        return {
            'is_fake': is_fake,
            'fake_prob': fake_prob
        }


if __name__ == '__main__':
    model = FaceAntiSpoofV3('localhost', 8500, 'anti_spoof', version=3, threshold=0.4, timeout=30)

    import glob, os

    labels = {
        # 'real': [
        #     '/media/thiennt/projects/face_lvt/application/test/data/anti_spoof/real/*',
        #     '/media/data_it/Data_set/database_image/face/data_fake/real/*'
        # ],
        'fake': [
            # '/media/data_it/Data_set/database_image/face/data_fake/paper_image/*',
            # '/media/data_it/Data_set/database_image/face/data_fake/screen/*',
            '/media/data_it/Data_set/database_image/face/CelebA_Spoof/filter_data/train/spoof/*',
        ]
    }
    batch_size = 64
    count_true = 0
    count_false = 0
    output_path = '/media/thiennt/projects/face_lvt/application/test/data/anti_spoof/model_v6_04.csv'
    output_f = open(output_path, 'a')
    if not os.path.exists(output_path):
        output_f.write("filename,label,score\n")
    batches = []
    filenames = []
    for label, dir_paths in labels.items():
        for dir_path in dir_paths:
            for f in glob.glob(dir_path):
                filename = os.path.basename(f)
                img = cv2.imread(f)
                batches.append(img)
                filenames.append(filename)
                if len(batches) >= batch_size:

                    output = model.predict(batches)
                    for filename, pred, score in zip(filenames, output['is_fake'], output['fake_prob']):
                        if pred:
                            count_true += 1
                        else:
                            count_false += 1
                        line = f'{filename},{label},{score}'
                        output_f.write(line + "\n")
                        print(line)
                    batches.clear()
                    filenames.clear()

    if len(batches) >= 0:
        output = model.predict(batches)
        for filename, pred, score in zip(filenames, output['is_fake'], output['fake_prob']):
            if pred:
                count_true += 1
            else:
                count_false += 1
            line = f'{filename},{label},{score}'
            output_f.write(line + "\n")
            print(line)
        batches.clear()
        filenames.clear()
    print(f'fraud: {count_true}')
    print(f'real: {count_false}')
    output_f.close()
