#!/usr/bin/python3
from ctypes import *
import cv2
import numpy as np
import sys
import os
import time
# from ipdb import set_trace as dbg
from enum import IntEnum
import glob


this_dir = os.path.dirname(os.path.realpath(__file__))

class CPoint(Structure):
    _fields_ = [("x", c_int),
                ("y", c_int)]


FEAT_POINTS = 14


class CWindow(Structure):
    _fields_ = [("x", c_int),
                ("y", c_int),
                ("width", c_int),
                ("angle", c_int),
                ("score", c_float),
                ("points", CPoint * FEAT_POINTS)]


class FeatEnam(IntEnum):
    CHIN_0 = 0
    CHIN_1 = 1
    CHIN_2 = 2
    CHIN_3 = 3
    CHIN_4 = 4
    CHIN_5 = 5
    CHIN_6 = 6
    CHIN_7 = 7
    CHIN_8 = 8
    NOSE = 9
    EYE_LEFT = 10
    EYE_RIGHT = 11
    MOUTH_LEFT = 12
    MOUTH_RIGHT = 13
    FEAT_POINTS = 14

lib = CDLL(os.path.join(this_dir, "libPCN.so"))

init_detector = lib.init_detector
# void *init_detector(const char *detection_model_path,
#            const char *pcn1_proto, const char *pcn2_proto, const char *pcn3_proto,
#            const char *tracking_model_path, const char *tracking_proto,
#            int min_face_size, float pyramid_scale_factor, float detection_thresh_stage1,
#            float detection_thresh_stage2, float detection_thresh_stage3, int tracking_period,
#            float tracking_thresh, int do_smooth)
init_detector.argtypes = [
    c_char_p, c_char_p, c_char_p,
    c_char_p, c_char_p, c_char_p,
    c_int, c_float, c_float, c_float,
    c_float, c_int, c_float, c_int]
init_detector.restype = c_void_p

# CWindow* detect_faces(void* pcn, unsigned char* raw_img,size_t rows, size_t cols, int *lwin)
detect_faces = lib.detect_faces
detect_faces.argtypes = [c_void_p, POINTER(c_ubyte), c_size_t, c_size_t, POINTER(c_int)]
detect_faces.restype = POINTER(CWindow)

#CWindow* detect_track_faces(void* pcn, unsigned char* raw_img,size_t rows, size_t cols, int *lwin)
detect_track_faces = lib.detect_track_faces
detect_track_faces.argtypes = [c_void_p, POINTER(c_ubyte),c_size_t,c_size_t,POINTER(c_int)]
detect_track_faces.restype = POINTER(CWindow)

# void free_faces(CWindow* wins)
free_faces = lib.free_faces
free_faces.argtypes = [c_void_p]

# void free_detector(void *pcn)
free_detector = lib.free_detector
free_detector.argtypes = [c_void_p]

CYAN = (255, 255, 0)
BLUE = (255, 0, 0)
RED = (0, 0, 255)
GREEN = (0, 255, 0)
YELLOW = (0, 255, 255)


def DrawFace(win, img):
    width = 2
    x1 = win.x
    y1 = win.y
    x2 = win.width + win.x - 1
    y2 = win.width + win.y - 1
    centerX = (x1 + x2) / 2
    centerY = (y1 + y2) / 2
    angle = win.angle
    R = cv2.getRotationMatrix2D((centerX, centerY), angle, 1)
    pts = np.array([[x1, y1, 1], [x1, y2, 1], [x2, y2, 1], [x2, y1, 1]], np.int32)
    pts = (pts @ R.T).astype(int)  # Rotate points
    pts = pts.reshape((-1, 1, 2))
    cv2.polylines(img, [pts], True, CYAN, width)
    cv2.line(img, (pts[0][0][0], pts[0][0][1]), (pts[3][0][0], pts[3][0][1]), BLUE, width)
    cv2.rectangle(img, (x1, y1), (x2, y2), (0, 255, 255))


def DrawPoints(win, img):
    width = 3
    f = FeatEnam.NOSE
    cv2.circle(img, (win.points[f].x, win.points[f].y), width, GREEN, -1)
    f = FeatEnam.EYE_LEFT
    cv2.circle(img, (win.points[f].x, win.points[f].y), width, YELLOW, -1)
    f = FeatEnam.EYE_RIGHT
    cv2.circle(img, (win.points[f].x, win.points[f].y), width, YELLOW, -1)
    f = FeatEnam.MOUTH_LEFT
    cv2.circle(img, (win.points[f].x, win.points[f].y), width, RED, -1)
    f = FeatEnam.MOUTH_RIGHT
    cv2.circle(img, (win.points[f].x, win.points[f].y), width, RED, -1)
    for i in range(9):
        cv2.circle(img, (win.points[i].x, win.points[i].y), width, BLUE, -1)

    for i in range(FeatEnam.FEAT_POINTS):
        cv2.putText(img, '{}'.format(i), (win.points[i].x, win.points[i].y), cv2.FONT_HERSHEY_SIMPLEX, 1,BLUE)


def SetThreadCount(threads):
    os.environ['OMP_NUM_THREADS'] = str(threads)


def c_str(str_in):
    return c_char_p(str_in.encode('utf-8'))


class PCNDetector:
    def __init__(self):
        path = os.path.join(this_dir, 'model/')
        detection_model_path = c_str(path + "PCN.caffemodel")
        pcn1_proto = c_str(path + "PCN-1.prototxt")
        pcn2_proto = c_str(path + "PCN-2.prototxt")
        pcn3_proto = c_str(path + "PCN-3.prototxt")
        tracking_model_path = c_str(path + "PCN-Tracking.caffemodel")
        tracking_proto = c_str(path + "PCN-Tracking.prototxt")

        self.detector = init_detector(detection_model_path, pcn1_proto, pcn2_proto, pcn3_proto,
                                 tracking_model_path, tracking_proto,
                                 40, 1.414, 0.37, 0.43, 0.90, 30, 0.9, 0)
        # print('pcn detector')

    def detect_face(self, face_img, debug=False, tracking=False):
        width = face_img.shape[1]
        height = face_img.shape[0]
        start = time.time()
        face_count = c_int(0)
        raw_data = face_img.ctypes.data_as(POINTER(c_ubyte))

        if tracking:
            windows = detect_track_faces(self.detector, raw_data,
                                         int(height), int(width),
                                         pointer(face_count))
        else:
            # print('Track')
            windows = detect_faces(self.detector, raw_data,
                                   int(height), int(width),
                                   pointer(face_count))

        faces_det = []
        landmarks = []
        face_area = 0
        face_angle = 0
        for i in range(face_count.value):
            win = windows[i]
            score = win.score

            # find biggest face angle
            if face_area < win.width * win.width:
                face_area = win.width * win.width
                face_angle = win.angle

            # find face location by landmark point
            x1 = face_img.shape[1]
            y1 = face_img.shape[0]
            x2 = 0
            y2 = 0
            for p in win.points:
                x1 = p.x if x1 > p.x else x1
                y1 = p.y if y1 > p.y else y1
                x2 = p.x if x2 < p.x else x2
                y2 = p.y if y2 < p.y else y2
            angle = (win.angle + 360) % 360
            if 45 <= angle < 135:
                x1 = win.x
            elif 135 <= angle < 225:
                y2 = win.y + win.width
            elif 225 <= angle < 315:
                x2 = win.x + win.width
            else:
                y1 = win.y
            faces_det.append(np.array((x1, y1, x2, y2, score)))

            # landmark
            landmark = np.zeros((5, 2))
            for j, f in enumerate([FeatEnam.EYE_LEFT, FeatEnam.EYE_RIGHT, FeatEnam.NOSE,
                                   FeatEnam.MOUTH_LEFT, FeatEnam.MOUTH_RIGHT]):
                landmark[j][0] = win.points[f].x
                landmark[j][1] = win.points[f].y

            landmarks.append(landmark)
        end = time.time()
        if debug:
            for i in range(face_count.value):
                win = windows[i]
                DrawFace(win, face_img)
                DrawPoints(win, face_img)
                cv2.rectangle(face_img,
                              (int(faces_det[i][0]), int(faces_det[i][1])),
                              (int(faces_det[i][2]), int(faces_det[i][3])),
                              (0, 0, 255))
            fps = 1 / (end-start)
            cv2.putText(face_img, str(fps) + "fps", (20, 45), 4, 1, (0, 0, 125))

            cv2.imshow('pcn', face_img)
            # cv2.waitKey(1)

        # print("rotation detection: ", (end- start) * 1000)
        if len(faces_det) > 0:
            return np.array(faces_det), np.array(landmarks), face_angle
        else:
            return None

    def get_face_rotation(self, face_img):
        s = time.time()
        width = face_img.shape[1]
        height = face_img.shape[0]
        face_count = c_int(0)
        raw_data = face_img.ctypes.data_as(POINTER(c_ubyte))

        windows = detect_faces(self.detector, raw_data,
                               int(height), int(width),
                               pointer(face_count))
        face_area = 0
        face_angle = 0
        x1 = 0
        y1 = 0
        x2 = face_img.shape[1] - 1
        y2 = face_img.shape[0] - 1
        for i in range(face_count.value):
            win = windows[i]
            if face_area < win.width * win.width:
                face_area = win.width * win.width
                face_angle = win.angle
                x1 = win.x - win.width // 2
                y1 = win.y - win.width // 2
                x2 = x1 + win.width + win.width // 2
                y2 = y1 + win.width + win.width // 2
        x1 = x1 if x1 >= 0 else 0
        y1 = y1 if y1 >= 0 else 0
        x2 = x2 if x2 < face_img.shape[1] else face_img.shape[1] - 1
        y2 = y2 if y2 < face_img.shape[0] else face_img.shape[0] - 1
        e = time.time()
        # print("rotation detection: ",(e-s) * 1000)
        return face_angle, x1, y1, x2, y2


if __name__ == '__main__':
    face_img = cv2.imread('/workspace/img/1.jpg')
    pcn_detector = PCNDetector()
    print(pcn_detector.detect_face(face_img, debug=True))

    video = '/home/vvn/Downloads/testface.mp4'
    cap = cv2.VideoCapture(video)
    while cap.isOpened():
        ret, frame = cap.read()
        if frame.shape[0] == 0:
            break
        start = time.time()
        pcn_detector.detect_face(frame, debug=True, tracking=False)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

