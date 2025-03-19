import math
import logging
import cv2
import numpy as np
from scipy.spatial import distance as dist

from config_run import Config
from modules.face.face_detection import FastFaceDetection
from modules.face.face_emotion import FaceEmotion, FaceEmotionV2
from modules.face.face_landmark import FaceLandmark
from modules.face.liveness.command import Command

LOGGER = logging.getLogger("model")


class ActionRecognition:
    def __init__(self, service_host, service_port, thresh_config: Config, img_size=(480, 640), timeout: int = 5, debug=False):
        self.face_detector = FastFaceDetection(service_host=service_host, service_port=service_port, image_size=128,
                                               timeout=timeout)
        self.landmark_model = FaceLandmark(service_host=service_host, service_port=service_port, image_size=128,
                                           timeout=timeout)
        self.pose_estimator = PoseEstimator(img_size=img_size)
        # self.emotion_model = FaceEmotion(service_host=service_host, service_port=service_port, timeout=timeout)
        self.emotion_model = FaceEmotionV2(service_host=service_host, service_port=service_port, timeout=timeout)

        self.thresh_up = thresh_config.thresh_up
        self.thresh_down = thresh_config.thresh_down
        self.thresh_left = thresh_config.thresh_left
        self.thresh_right = thresh_config.thresh_right
        self.thresh_tilting_left = thresh_config.thresh_tilting_left
        self.thresh_tilting_right = thresh_config.thresh_tilting_right
        self.thresh_blink = thresh_config.thresh_blink
        self.thresh_mouth = thresh_config.thresh_mouth

        self.thresh_action = thresh_config.thresh_action

        self.debug = debug

    def check_action_discrete(self, rxs, rys, rzs, res, rms, command):
        if command == Command.HEAD_UP:
            count = (rxs < self.thresh_up).sum()
            LOGGER.debug(f"Liveness: Count head up: {count}")
            if count >= self.thresh_action:
                return True
        elif command == Command.HEAD_DOWN:
            count = (rxs > self.thresh_down).sum()
            LOGGER.debug(f"Liveness: Count head down: {count}")
            if count >= self.thresh_action:
                return True
        elif command == Command.HEAD_LEFT:
            count = (rys > self.thresh_left).sum()
            LOGGER.debug(f"Liveness: Count head left: {count}")
            if count >= self.thresh_action:
                return True
        elif command == Command.HEAD_RIGHT:
            count = (rys < self.thresh_right).sum()
            LOGGER.debug(f"Liveness: Count head right: {count}")
            if count >= self.thresh_action:
                return True
        elif command == Command.HEAD_TILTING_LEFT:
            count = (rzs > self.thresh_tilting_left).sum()
            LOGGER.debug(f"Liveness: Count head tilting left: {count}")
            if count >= self.thresh_action:
                return True
        elif command == Command.HEAD_TILTING_RIGHT:
            count = (rzs < self.thresh_tilting_right).sum()
            LOGGER.debug(f"Liveness: Count head tilting right: {count}")
            if count >= self.thresh_action:
                return True
        elif command == Command.MOUTH_OPEN:
            count = (rms > self.thresh_mouth).sum()
            LOGGER.debug(f"Liveness: Count mouth open: {count}")
            if count >= self.thresh_action:
                return True
        elif command == Command.EYE_BLINK:
            count = (res > self.thresh_blink).sum()
            LOGGER.debug(f"Liveness: Count eye blink: {count}")
            if count >= self.thresh_action:
                return True
        else:
            raise Exception("Command not found!")
        return False

    def check_smile_action(self, imgs):
        face_imgs = []
        for img in imgs:
            bboxes, landmarks, confidences, images = self.face_detector.predict(img, {"max_faces": 1,
                                                                                  "receive_mode": "image",
                                                                                  "fast": True})
            if images:
                face_imgs.append(images[0])
        res = self.emotion_model.predict(face_imgs)
        LOGGER.debug(f'Emotion output: {res}')
        count = (np.array(res['label']) == 'happy').sum()
        LOGGER.debug(f'Liveness: Count smile: {count}')
        if count >= self.thresh_action:
            return True
        return False

    def get_values(self, img):
        faces = self.get_landmark_2(img)
        if faces and len(faces) > 0:
            box, landmark = faces
            rx, ry, rz = self.pose_estimator.get_angles(landmark)
            if rz > 0:
                rz -= 180
            else:
                rz += 180
            eye_ratio = self.get_eye_ratio(landmark)
            mouth_ratio = self.get_mouth_ratio(landmark)
            if self.debug:
                cv2.putText(img, "X:{:.2f}   Y:{:.2f}   Z:{:.2f}".format(rx, ry, rz), (10, 50),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0),
                            2)
                cv2.putText(img, "ER:{:.2f}".format(eye_ratio), (50, 150), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0),
                            2)
                cv2.putText(img, "MR:{:.2f}".format(mouth_ratio), (50, 200), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0),
                            2)

            return rx, ry, rz, eye_ratio, mouth_ratio

    @staticmethod
    def moving_average(a, n=3):
        ret = np.cumsum(a, dtype=float)
        ret[n:] = ret[n:] - ret[:-n]
        return ret[n - 1:] / n

    @staticmethod
    def increase_decrease_counts(a):
        a = ActionRecognition.moving_average(a, n=5)
        mask = a[1:] >= a[:-1]
        idx = np.flatnonzero(mask[1:] != mask[:-1])
        if len(idx) == 0:
            max_count = len(mask)
            return 0, 0, 0, 0, 0, 0
        count = np.concatenate(([idx[0] + 1], idx[1:] - idx[:-1] + 1, [a.size - 1 - idx[-1]]))

        idx_max_count = np.argmax(count)  # 1
        max_count = count[idx_max_count]  # 3
        end_idx = count[0]
        for i in range(1, idx_max_count + 1):
            end_idx += count[i] - 1

        start_idx = end_idx - count[idx_max_count] + 1
        # print("Start idx: {}   Start value: {}   end idx: {}   end value: {}    max_count:{}".format(start_idx,
        #                                                                                              a[start_idx],
        #                                                                                              end_idx,
        #                                                                                              a[end_idx],
        #                                                                                              max_count))

        return start_idx, a[start_idx], end_idx, a[end_idx], max_count, a[end_idx] - a[start_idx]

    @staticmethod
    def get_mouth_ratio(landmark):
        mouth = landmark[48:69]

        # compute the euclidean distances between the two sets of
        # vertical mouth landmarks (x, y)-coordinates
        A = dist.euclidean(mouth[2], mouth[10])  # 51, 59
        B = dist.euclidean(mouth[4], mouth[8])  # 53, 57

        # compute the euclidean distance between the horizontal
        # mouth landmark (x, y)-coordinates
        C = dist.euclidean(mouth[0], mouth[6])  # 49, 55

        # compute the mouth aspect ratio
        mar = (A + B) / (2.0 * C)

        # return the mouth aspect ratio
        return mar

    def get_eye_ratio(self, landmark):
        # left eye
        left_eye_points = landmark[36:42]
        center_top = self._midpoint(left_eye_points[1], left_eye_points[2])
        center_bottom = self._midpoint(left_eye_points[5], left_eye_points[4])

        hor_line_length = math.hypot((left_eye_points[0][0] - left_eye_points[3][0]),
                                     (left_eye_points[0][1] - left_eye_points[3][1]))
        ver_line_length = math.hypot((center_top[0] - center_bottom[0]), (center_top[1] - center_bottom[1]))

        left_ratio = hor_line_length / ver_line_length if ver_line_length != 0 else 100

        # right eye
        right_eye_points = landmark[42:48]
        center_top = self._midpoint(right_eye_points[1], right_eye_points[2])
        center_bottom = self._midpoint(right_eye_points[5], right_eye_points[4])

        hor_line_length = math.hypot((right_eye_points[0][0] - right_eye_points[3][0]),
                                     (right_eye_points[0][1] - left_eye_points[3][1]))
        ver_line_length = math.hypot((center_top[0] - center_bottom[0]), (center_top[1] - center_bottom[1]))

        right_ratio = hor_line_length / ver_line_length if ver_line_length != 0 else 100

        return (left_ratio + right_ratio) / 2

    @staticmethod
    def _midpoint(p1, p2):
        return int((p1[0] + p2[0]) / 2), int((p1[1] + p2[1]) / 2)

    def get_landmark_2(self, img):
        bboxes, landmarks, confidences, images = self.face_detector.predict(img, {"max_faces": 1,
                                                                                  "receive_mode": "meta",
                                                                                  "fast": True})
        if not bboxes or len(bboxes) == 0:
            return None
        box = bboxes[0]
        # bw = box[2] - box[0]
        # bh = box[3] - box[1]
        # box[1] += int(0.2*bh)
        offset_y = int(abs((box[3] - box[1]) * 0.1))
        box = self.move_box(box, [0, offset_y])
        # box = self.get_square_box(box)
        marks = self.landmark_model.get_dlib(img, box)
        if self.debug:
            cv2.rectangle(img, (box[0], box[1]), (box[2], box[3]), (0, 255, 0), thickness=2)
            for i, p in enumerate(marks, start=1):
                cv2.circle(img, (int(p[0]), int(p[1])), radius=1, color=(255, 100, 0), thickness=2)
        return box, marks

    def get_landmark(self, img):
        bboxes, landmarks, confidences, images = self.face_detector.predict(img, {"max_faces": 1,
                                                                                  "receive_mode": "meta",
                                                                                  "fast": True})
        if not bboxes or len(bboxes) == 0:
            return None
        box = bboxes[0]
        # Move box down.
        # diff_height_width = (box[3] - box[1]) - (box[2] - box[0])
        offset_y = int(abs((box[3] - box[1]) * 0.1))
        box = self.move_box(box, [0, offset_y])
        box = self.get_square_box(box)
        if self.box_in_image(box, img):
            face_img = img[box[1]:box[3], box[0]:box[2]]
            marks = self.landmark_model.predict(face_img)
            marks *= (box[2] - box[0])
            marks[:, 0] += box[0]
            marks[:, 1] += box[1]
            # marks = np.vstack([np.array([[0, 0]]), marks])
            if self.debug:
                cv2.rectangle(img, (box[0], box[1]), (box[2], box[3]), (0, 255, 0), thickness=2)
                for i, p in enumerate(marks, start=1):
                    cv2.circle(img, (int(p[0]), int(p[1])), radius=1, color=(255, 100, 0), thickness=1)
            return box, marks

    @staticmethod
    def move_box(box, offset):
        """Move the box to direction specified by vector offset"""
        left_x = box[0] + offset[0]
        top_y = box[1] + offset[1]
        right_x = box[2] + offset[0]
        bottom_y = box[3] + offset[1]
        return [left_x, top_y, right_x, bottom_y]

    @staticmethod
    def box_in_image(box, image):
        """Check if the box is in image"""
        rows = image.shape[0]
        cols = image.shape[1]
        return box[0] >= 0 and box[1] >= 0 and box[2] <= cols and box[3] <= rows

    @staticmethod
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


class PoseEstimator(object):
    """Estimate head pose according to the facial landmarks"""

    def __init__(self, img_size=(480, 640)):
        self.size = img_size

        # 3D model points.
        self.model_points = np.array([
            (0.0, 0.0, 0.0),  # Nose tip
            (0.0, -330.0, -65.0),  # Chin
            (-225.0, 170.0, -135.0),  # Left eye left corner
            (225.0, 170.0, -135.0),  # Right eye right corner
            (-150.0, -150.0, -125.0),  # Mouth left corner
            (150.0, -150.0, -125.0)  # Mouth right corner
        ]) / 4.5

        self.model_points_68 = self._get_full_model_points()

        # Camera internals
        self.focal_length = self.size[1]
        self.camera_center = (self.size[1] / 2, self.size[0] / 2)
        self.camera_matrix = np.array(
            [[self.focal_length, 0, self.camera_center[0]],
             [0, self.focal_length, self.camera_center[1]],
             [0, 0, 1]], dtype="double")

        # Assuming no lens distortion
        self.dist_coeefs = np.zeros((4, 1))

        # Rotation vector and translation vector
        self.r_vec = np.array([[0.01891013], [0.08560084], [-3.14392813]])
        self.t_vec = np.array(
            [[-14.97821226], [-10.62040383], [-2053.03596872]])
        # self.r_vec = None
        # self.t_vec = None

    def get_angles(self, landmark):
        r_vec, t_vec = self._solve_pose_by_68_points(landmark)
        rx, ry, rz = self._get_angles(r_vec, t_vec)
        return rx, ry, rz

    def _solve_pose_by_68_points(self, image_points):
        """
        Solve pose from all the 68 image points
        Return (rotation_vector, translation_vector) as pose.
        """

        if self.r_vec is None:
            (_, rotation_vector, translation_vector) = cv2.solvePnP(
                self.model_points_68, image_points, self.camera_matrix, self.dist_coeefs)
            self.r_vec = rotation_vector
            self.t_vec = translation_vector

        (_, rotation_vector, translation_vector) = cv2.solvePnP(
            self.model_points_68,
            image_points,
            self.camera_matrix,
            self.dist_coeefs,
            rvec=self.r_vec,
            tvec=self.t_vec,
            useExtrinsicGuess=True)

        return rotation_vector, translation_vector

    # rotation vector to euler angles
    @staticmethod
    def _get_angles(rvec, tvec):
        rmat = cv2.Rodrigues(rvec)[0]
        P = np.hstack((rmat, tvec))  # projection matrix [R | t]
        degrees = -cv2.decomposeProjectionMatrix(P)[6]
        rx, ry, rz = degrees[:, 0]
        return rx, ry, rz

    def _get_full_model_points(self):
        model_points = np.array(
            [-73.393523, -72.775014, -70.533638, -66.850058, -59.790187, -48.368973, -34.121101, -17.875411, 0.098749,
             17.477031, 32.648966, 46.372358, 57.343480, 64.388482, 68.212038, 70.486405, 71.375822, -61.119406,
             -51.287588, -37.804800, -24.022754, -11.635713, 12.056636, 25.106256, 38.338588, 51.191007, 60.053851,
             0.653940, 0.804809, 0.992204, 1.226783, -14.772472, -7.180239, 0.555920, 8.272499, 15.214351, -46.047290,
             -37.674688, -27.883856, -19.648268, -28.272965, -38.082418, 19.265868, 27.894191, 37.437529, 45.170805,
             38.196454, 28.764989, -28.916267, -17.533194, -6.684590, 0.381001, 8.375443, 18.876618, 28.794412,
             19.057574, 8.956375, 0.381549, -7.428895, -18.160634, -24.377490, -6.897633, 0.340663, 8.444722, 24.474473,
             8.449166, 0.205322, -7.198266, -29.801432, -10.949766, 7.929818, 26.074280, 42.564390, 56.481080,
             67.246992, 75.056892, 77.061286, 74.758448, 66.929021, 56.311389, 42.419126, 25.455880, 6.990805,
             -11.666193, -30.365191, -49.361602, -58.769795, -61.996155, -61.033399, -56.686759, -57.391033, -61.902186,
             -62.777713, -59.302347, -50.190255, -42.193790, -30.993721, -19.944596, -8.414541, 2.598255, 4.751589,
             6.562900, 4.661005, 2.643046, -37.471411, -42.730510, -42.711517, -36.754742, -35.134493, -34.919043,
             -37.032306, -43.342445, -43.110822, -38.086515, -35.532024, -35.484289, 28.612716, 22.172187, 19.029051,
             20.721118, 19.035460, 22.394109, 28.079924, 36.298248, 39.634575, 40.395647, 39.836405, 36.677899,
             28.677771, 25.475976, 26.014269, 25.326198, 28.323008, 30.596216, 31.408738, 30.844876, 47.667532,
             45.909403, 44.842580, 43.141114, 38.635298, 30.750622, 18.456453, 3.609035, -0.881698, 5.181201, 19.176563,
             30.770570, 37.628629, 40.886309, 42.281449, 44.142567, 47.140426, 14.254422, 7.268147, 0.442051, -6.606501,
             -11.967398, -12.051204, -7.315098, -1.022953, 5.349435, 11.615746, -13.380835, -21.150853, -29.284036,
             -36.948060, -20.132003, -23.536684, -25.944448, -23.695741, -20.858157, 7.037989, 3.021217, 1.353629,
             -0.111088, -0.147273, 1.476612, -0.665746, 0.247660, 1.696435, 4.894163, 0.282961, -1.172675, -2.240310,
             -15.934335, -22.611355, -23.748437, -22.721995, -15.610679, -3.217393, -14.987997, -22.554245, -23.591626,
             -22.406106, -15.121907, -4.785684, -20.893742, -22.220479, -21.025520, -5.712776, -20.671489, -21.903670,
             -20.328022])
        model_points = np.reshape(model_points, (3, -1)).T

        # Transform the model into a front view.
        model_points[:, 2] *= -1

        return model_points

    def draw_annotation_box(self, image, rotation_vector, translation_vector, color=(255, 255, 255), line_width=2):
        """Draw a 3D box as annotation of pose"""
        point_3d = []
        rear_size = 75
        rear_depth = 0
        point_3d.append((-rear_size, -rear_size, rear_depth))
        point_3d.append((-rear_size, rear_size, rear_depth))
        point_3d.append((rear_size, rear_size, rear_depth))
        point_3d.append((rear_size, -rear_size, rear_depth))
        point_3d.append((-rear_size, -rear_size, rear_depth))

        front_size = 100
        front_depth = 100
        point_3d.append((-front_size, -front_size, front_depth))
        point_3d.append((-front_size, front_size, front_depth))
        point_3d.append((front_size, front_size, front_depth))
        point_3d.append((front_size, -front_size, front_depth))
        point_3d.append((-front_size, -front_size, front_depth))
        point_3d = np.array(point_3d, dtype=np.float).reshape(-1, 3)

        # Map to 2d image points
        (point_2d, _) = cv2.projectPoints(point_3d,
                                          rotation_vector,
                                          translation_vector,
                                          self.camera_matrix,
                                          self.dist_coeefs)
        point_2d = np.int32(point_2d.reshape(-1, 2))

        # Draw all the lines
        cv2.polylines(image, [point_2d], True, color, line_width, cv2.LINE_AA)
        cv2.line(image, tuple(point_2d[1]), tuple(
            point_2d[6]), color, line_width, cv2.LINE_AA)
        cv2.line(image, tuple(point_2d[2]), tuple(
            point_2d[7]), color, line_width, cv2.LINE_AA)
        cv2.line(image, tuple(point_2d[3]), tuple(
            point_2d[8]), color, line_width, cv2.LINE_AA)

    def draw_axis(self, img, R, t):
        points = np.float32(
            [[30, 0, 0], [0, 30, 0], [0, 0, 30], [0, 0, 0]]).reshape(-1, 3)

        axisPoints, _ = cv2.projectPoints(
            points, R, t, self.camera_matrix, self.dist_coeefs)

        img = cv2.line(img, tuple(axisPoints[3].ravel()), tuple(
            axisPoints[0].ravel()), (255, 0, 0), 3)
        img = cv2.line(img, tuple(axisPoints[3].ravel()), tuple(
            axisPoints[1].ravel()), (0, 255, 0), 3)
        img = cv2.line(img, tuple(axisPoints[3].ravel()), tuple(
            axisPoints[2].ravel()), (0, 0, 255), 3)

    def draw_axes(self, img, R, t):
        img = cv2.drawFrameAxes(img, self.camera_matrix, self.dist_coeefs, R, t, 30)

    def get_pose_marks(self, marks):
        """Get marks ready for pose estimation from 68 marks"""
        pose_marks = []
        pose_marks.append(marks[30])  # Nose tip
        pose_marks.append(marks[8])  # Chin
        pose_marks.append(marks[36])  # Left eye left corner
        pose_marks.append(marks[45])  # Right eye right corner
        pose_marks.append(marks[48])  # Mouth left corner
        pose_marks.append(marks[54])  # Mouth right corner
        return pose_marks


if __name__ == '__main__':

    recog = ActionRecognition("localhost", 8500, thresh_config=Config(), debug=True)
    cap = cv2.VideoCapture(0)
    queue = []

    n_imgs = 72
    rxs = np.zeros(n_imgs)
    rys = np.zeros(n_imgs)
    rzs = np.zeros(n_imgs)
    res = np.zeros(n_imgs)
    rms = np.zeros(n_imgs)
    i = 0
    while cap.isOpened():
        ret, frame = cap.read()

        if not ret:
            break
        frame = cv2.flip(frame, 1)
        # cv2.imwrite(f"/media/thiennt/projects/remote_lvt/ekyc-lvt/application/test/data/liveness/head_main/{i}.jpg",
        #             frame)
        # if i == 50:
        #     break
        values = recog.get_values(frame)
        if not values:
            continue

        rx, ry, rz, eye_ratio, mouth_ratio = values

        # print(rz)
        rxs[i % n_imgs] = rx
        rys[i % n_imgs] = ry
        rzs[i % n_imgs] = rz
        res[i % n_imgs] = eye_ratio
        rms[i % n_imgs] = mouth_ratio

        if i % n_imgs == 0:
            cv2.putText(frame, "NEXT ACTION", (50, 50), cv2.FONT_HERSHEY_COMPLEX, 2, (255, 255, 0), 3)
            recog.check_action_discrete(rxs, rys, rzs, res, rms, Command.HEAD_LEFT)
        i += 1
        cv2.imshow("frame", frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
