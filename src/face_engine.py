import cv2
import numpy as np

import config
import utils


class Detection:
    def __init__(self, box, landmarks, score):
        self.box = box
        self.landmarks = landmarks
        self.score = score


class ReferenceStore:
    def __init__(self):
        self.embeddings = []
        self.mirrors = []

    def count(self):
        return len(self.embeddings)

    def empty(self):
        return not self.embeddings

    def add(self, embedding, mirror):
        self.embeddings.append(embedding)
        self.mirrors.append(mirror)

    def remove(self, index):
        if 0 <= index < len(self.embeddings):
            del self.embeddings[index]
            del self.mirrors[index]

    def clear(self):
        self.embeddings = []
        self.mirrors = []

    def match(self, vector):
        best_score = config.MIN_COSINE
        best_index = config.NO_MATCH_INDEX
        for index, (original, mirror) in enumerate(zip(self.embeddings, self.mirrors)):
            score = max(utils.cosine_similarity(vector, original), utils.cosine_similarity(vector, mirror))
            if score > best_score:
                best_score = score
                best_index = index
        return best_score, best_index


class FaceEngine:
    def __init__(self, detector_path, recognition_path):
        size = (config.DETECT_DEFAULT_WIDTH, config.DETECT_DEFAULT_HEIGHT)
        self.detector = cv2.FaceDetectorYN.create(
            detector_path,
            "",
            size,
            config.DETECT_SCORE_THRESHOLD,
            config.DETECT_NMS_THRESHOLD,
            config.DETECT_TOP_K,
            config.DETECT_BACKEND_DEFAULT,
            config.DETECT_TARGET_CPU,
        )
        self.detector_size = size
        self.session = None
        self.input_name = ""
        self.device = config.DEVICE_LABEL_NONE
        self.open_session(recognition_path)

    def open_session(self, recognition_path):
        try:
            import onnxruntime
        except ImportError:
            return
        try:
            onnxruntime.set_default_logger_severity(config.ONNX_LOG_SEVERITY_ERROR)
        except Exception:
            pass
        try:
            options = onnxruntime.SessionOptions()
            options.intra_op_num_threads = config.ONNX_INTRA_THREADS
            self.session = onnxruntime.InferenceSession(recognition_path, sess_options=options, providers=list(config.ONNX_PROVIDERS))
            self.input_name = self.session.get_inputs()[0].name
            used = self.session.get_providers()[0]
            if used == config.ONNX_PROVIDERS[0]:
                self.device = config.DEVICE_LABEL_GPU
            else:
                self.device = config.DEVICE_LABEL_CPU
        except Exception:
            self.session = None
            self.device = config.DEVICE_LABEL_NONE

    def sync_detector_size(self, width, height):
        if (width, height) != self.detector_size:
            self.detector.setInputSize((width, height))
            self.detector_size = (width, height)

    def detect(self, bgr):
        height, width = bgr.shape[0], bgr.shape[1]
        self.sync_detector_size(width, height)
        _, faces = self.detector.detect(bgr)
        found = []
        if faces is None:
            return found
        for row in faces:
            x, y, w, h, x1, y1, x2, y2, x3, y3, x4, y4, x5, y5, score = row
            box = utils.clip_box(int(x), int(y), int(w), int(h), width, height)
            if box.width < config.MIN_FACE_WIDTH or box.height < config.MIN_FACE_HEIGHT:
                continue
            raw = np.array([[x1, y1], [x2, y2], [x3, y3], [x4, y4], [x5, y5]], dtype=np.float32)
            found.append(Detection(box, utils.order_yunet_landmarks(raw), float(score)))
        return found

    def align(self, bgr, landmarks):
        return utils.align_face(bgr, landmarks)

    def embed_crop(self, crop):
        if self.session is None:
            return None
        try:
            rgb = cv2.cvtColor(crop, cv2.COLOR_BGR2RGB).astype(np.float32)
            normed = (rgb - config.PIXEL_NORM_MEAN) / config.PIXEL_NORM_DIV
            blob = np.expand_dims(np.transpose(normed, (2, 0, 1)), axis=0)
            raw = self.session.run(None, {self.input_name: blob})[0][0]
            return utils.l2_normalize(raw.astype(np.float32))
        except Exception:
            return None

    def embed_face(self, bgr, landmarks):
        crop = self.align(bgr, landmarks)
        if crop is None:
            return None, None
        return self.embed_crop(crop), crop

    def largest(self, detections):
        best = None
        best_area = 0
        for det in detections:
            area = det.box.area()
            if area > best_area:
                best_area = area
                best = det
        return best
