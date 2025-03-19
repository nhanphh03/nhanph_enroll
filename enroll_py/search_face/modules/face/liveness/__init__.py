import glob
import os
from typing import List

import cv2
import numpy as np

from config_run import Config
from modules.face.liveness.action_recognition import ActionRecognition
from modules.face.liveness.command import Command


class LivenessChecking:
    def __init__(self, service_host: str, service_port: int, thresh_config: Config, timeout: int = 5, debug: bool = False):
        self.action_recognition = ActionRecognition(service_host, service_port, thresh_config=thresh_config, img_size=(1280, 720), timeout=timeout,
                                                    debug=True)
        self.debug = debug

    def check_action(self, imgs: List[np.ndarray], command: Command) -> bool:
        n_frame_per_action = len(imgs)
        if n_frame_per_action == 0:
            return  # TODO
        rxs = np.zeros(n_frame_per_action)
        rys = np.zeros(n_frame_per_action)
        rzs = np.zeros(n_frame_per_action)
        res = np.zeros(n_frame_per_action)
        rms = np.zeros(n_frame_per_action)

        if command == Command.SMILE:
            return self.action_recognition.check_smile_action(imgs)
        else:
            for idx, frame in enumerate(imgs):
                values = self.action_recognition.get_values(frame)
                if not values:
                    rxs[idx] = 0
                    rys[idx] = 0
                    rzs[idx] = 0
                    res[idx] = 0
                    rms[idx] = 0
                else:
                    rx, ry, rz, eye_ratio, mouth_ratio = values
                    rxs[idx] = rx
                    rys[idx] = ry
                    rzs[idx] = rz
                    res[idx] = eye_ratio
                    rms[idx] = mouth_ratio

            return self.action_recognition.check_action_discrete(rxs, rys, rzs, res, rms, command)


if __name__ == '__main__':
    model = LivenessChecking("localhost", 8500, debug=True, thresh_config=Config())
    data_dir = "/media/thiennt/projects/face_lvt/application/test/data/liveness"
    max_img = 10
    for folder in os.listdir(data_dir):
        imgs = []
        i = 0
        for file in glob.glob(os.path.join(data_dir, folder) + "/*.jpg"):
            imgs.append(cv2.imread(file))
            i += 1
            if i == max_img:
                break
        try:
            print(folder, model.check_action(imgs, Command[folder.upper()]))
        except Exception as e:
            print(e)
            continue
