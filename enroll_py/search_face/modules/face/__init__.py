import logging
from typing import List, Dict

import numpy as np

from config_run import Config
from modules.face.face_analysis import FaceAnalysis
from modules.face.liveness import LivenessChecking, Command
from modules.face.liveness.anti_spoof import FaceAntiSpoofV2, FaceAntiSpoofV3, FaceAntiSpoof
from modules.utils import compute_sim, compile_images

LOGGER = logging.getLogger("model")


class NotEnoughFaceCompare(Exception):
    pass


class NoFaceDetection(Exception):
    pass


face_analysis: FaceAnalysis = None

face_liveness: LivenessChecking = None

face_anti_spoof: FaceAntiSpoofV2 = None

face_anti_spoof_v1: FaceAntiSpoofV3 = None

face_anti_spoof_v3: FaceAntiSpoofV3 = None


def init_module(config):
    global face_analysis, face_liveness, face_anti_spoof, face_anti_spoof_v1, face_anti_spoof_v3
    if face_analysis is None:
        face_analysis = FaceAnalysis(service_host=config.service_host,
                                     service_port=config.service_port,
                                     timeout=config.service_timeout,
                                     debug=config.DEBUG)
    if face_liveness is None:
        face_liveness = LivenessChecking(service_host=config.service_host,
                                         service_port=config.service_port,
                                         thresh_config=config,
                                         timeout=config.service_timeout,
                                         debug=config.DEBUG)
    if face_anti_spoof is None:
        face_anti_spoof = FaceAntiSpoofV2(service_host=config.service_host,
                                          service_port=config.service_port,
                                          model_name='anti_spoof',
                                          version=2,
                                          threshold=config.face_anti_spoof_threshold,
                                          timeout=config.service_timeout,
                                          debug=config.DEBUG)
    if face_anti_spoof_v1 is None:
        face_anti_spoof_v1 = FaceAntiSpoofV3(service_host=config.service_host,
                                             service_port=config.service_port,
                                             model_name='anti_spoof',
                                             version=1,
                                             image_size=260,
                                             threshold=config.face_anti_spoof_threshold,
                                             timeout=config.service_timeout,
                                             debug=config.DEBUG)
    if face_anti_spoof_v3 is None:
        face_anti_spoof_v3 = FaceAntiSpoofV3(service_host=config.service_host,
                                             service_port=config.service_port,
                                             model_name='anti_spoof',
                                             version=3,
                                             image_size=260,
                                             threshold=config.face_anti_spoof_threshold,
                                             timeout=10,
                                             debug=config.DEBUG)


def detect_face(img, size_threshold=0.0, max_faces=10, receive_mode="meta", fast=True):
    """
    Detect face in image input. User size threshold to filter small face.
    :param img: numpy array image
    :param size_threshold: area threshold has range from 0 - 1. 0 is not filter and 1 is filter all.
    :param max_faces: Maximum of faces has response
    :param receive_mode: type of response 'meta': bounding boxes and landmarks;     'images': image of faces
    :param fast: boolean - mode inference
    :return:
    """

    return face_analysis.get(img,
                             size_threshold=size_threshold,
                             max_faces=max_faces,
                             receive_mode=receive_mode,
                             fast=fast,
                             mode="detect")


def embed_face(img, size_threshold=0.0, max_faces=10, fast=False):
    """
    Detect face in image input. User size threshold to filter small face.
    :param img: numpy array image
    :param size_threshold: area threshold has range from 0 - 1. 0 is not filter and 1 is filter all.
    :param max_faces: Maximum of faces has response
    :param fast: boolean - mode inference
    :return:
    """
    faces = face_analysis.get(img,
                              size_threshold=size_threshold,
                              max_faces=max_faces,
                              receive_mode="image",
                              fast=fast,
                              mode="embed")
    if len(faces) == 0:
        raise NoFaceDetection('Can\'t detect face. Please update another picture')
    return faces


def embed_face_only(images: List[np.ndarray], size_threshold=0.0, max_faces=10, fast=False) -> np.ndarray:
    """
        Get embedding of list face
        :param images: list numpy array image
        :param size_threshold: area threshold has range from 0 - 1. 0 is not filter and 1 is filter all.
        :param max_faces: Maximum of faces has response
        :param fast: boolean - mode inference
        :return: List embedding
        """
    embeddings = face_analysis.get(images, mode="embed_only")
    return embeddings


def compare_face(img1: np.ndarray, img2s: List[np.ndarray], threshold: float = Config.similar_thresh_compare):
    """
    Detect and embed face in each image and compare them.
    :raise NotEnoughFaceCompare if image don't have faces
    :param img1: Input 1st image
    :param img2: Input 2nd image
    :return:
    """
    face1 = face_analysis.get(img1,
                              size_threshold=0.0,
                              max_faces=1,
                              receive_mode="image",
                              fast=True,
                              mode="embed")
    num_face1 = len(face1)
    if num_face1 < 1:
        raise NoFaceDetection('Can\'t detect face. Please update another picture')
    face1 = face1[0]

    face2s = []
    for img2 in img2s:
        face2 = face_analysis.get(img2,
                                  size_threshold=0.0,
                                  max_faces=1,
                                  receive_mode="image",
                                  fast=True,
                                  mode="embed")
        num_face2 = len(face2)
        if num_face2 < 1:
            raise NoFaceDetection('Can\'t detect face. Please update another picture')
        face2s.append(face2[0])

    matches = []
    similars = []
    for j in range(len(face2s)):
        embedding1 = face1.embedding
        embedding2 = face2s[j].embedding
        sim = compute_sim(embedding1, embedding2, new_range=True)
        match = sim >= threshold
        similars.append(sim)
        matches.append(match)

    return matches, similars


def compare_face_v2(img1: np.ndarray, img2: np.ndarray, threshold: float = Config.similar_thresh_compare):
    img = compile_images(img1, img2)

    faces = face_analysis.get(img,
                              size_threshold=0,
                              max_faces=2,
                              receive_mode='image',
                              fast=True,
                              mode="embed")

    if len(faces) < 2:
        raise NoFaceDetection('Can\'t detect face. Please update another picture')

    embedding1 = faces[0].embedding
    embedding2 = faces[1].embedding
    sim = compute_sim(embedding1, embedding2, new_range=True)
    match = sim >= threshold
    return match, sim


def check_liveness(matches: List[bool], sims: List[float]) -> bool:
    if Config.min_liveness > 0:
        return sum(matches) >= Config.min_liveness
    else:
        return sum(matches) / len(matches) >= Config.thresh_liveness


def get_similars(emb1, emb2s, threshold: float = Config.similar_thresh_liveness):
    matches = []
    similars = []
    for j in range(len(emb2s)):
        sim = compute_sim(emb1, emb2s[j], new_range=True)
        match = sim >= threshold
        similars.append(sim)
        matches.append(match)

    return matches, similars


def check_action(imgs: List[np.ndarray], cmd: Command) -> bool:
    return face_liveness.check_action(imgs, cmd)


def is_fake(img: np.ndarray, version: int = 1) -> Dict[str, List]:
    if version == 1:
        res = face_anti_spoof_v1.predict([img])
    elif version == 2:
        faces = face_analysis.get(img,
                                  size_threshold=0.0,
                                  max_faces=1,
                                  receive_mode='image',
                                  fast=True,
                                  mode="detect",
                                  face_image_size=Config.face_anti_spoof_image_size,
                                  crop_mode='crop')
        if faces is not None and len(faces) > 0:
            res = face_anti_spoof.predict(faces[0].image)
        else:
            raise NoFaceDetection("Not found face to check")
    else:
        raise ValueError(f'Face anti spoof: unknown version {version}')
    return res
    # faces = face_analysis.get(img,
    #                           size_threshold=0.0,
    #                           max_faces=1,
    #                           receive_mode='image',
    #                           fast=True,
    #                           mode="detect",
    #                           face_image_size=Config.face_anti_spoof_image_size,
    #                           crop_mode='crop')
    # if faces is not None and len(faces) > 0:
    #     res = face_anti_spoof.predict(faces[0].image)
    #     if 0.5 < res['fake_prob'] < 0.99:
    #         res = face_anti_spoof_v1.predict(img)
    #         return res
    #     return res
    # else:
    #     raise NoFaceDetection("Not found face to check")


def check_fake(imgs: List[np.ndarray], version: int = 3) -> Dict[str, List]:
    res = None
    if version == 3:
        res = face_anti_spoof_v3.predict(imgs)
    elif version == 4:
        res = face_anti_spoof_v1.predict(imgs)
        is_fakes = []
        fake_probs = []
        i = 0
        for is_fake, fake_prob in zip(res['is_fake'], res['fake_prob']):
            LOGGER.debug(f'Model 1: {fake_prob}')
            if 0.0714 < fake_prob < 0.9957:
                faces = face_analysis.get(imgs[i],
                                          size_threshold=0.0,
                                          max_faces=1,
                                          receive_mode='image',
                                          fast=True,
                                          mode="detect",
                                          face_image_size=Config.face_anti_spoof_image_size,
                                          crop_mode='crop')
                if faces is not None and len(faces) > 0:
                    res = face_anti_spoof.predict(faces[0].image)
                    LOGGER.debug(f'Model 2: {res["fake_prob"]}')
                    fake_prob = (res['fake_prob'] + fake_prob) / 2
                    LOGGER.debug(f'Model final: {fake_prob}')
                    is_fake = fake_prob >= 0.75

            is_fakes.append(is_fake)
            fake_probs.append(fake_prob)
            i += 1
        res = {
            'is_fake': is_fakes,
            'fake_prob': fake_probs
        }
    return res
