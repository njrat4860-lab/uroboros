import os

import cv2
import numpy as np

import config


class Box:
    def __init__(self, x, y, width, height):
        self.x = x
        self.y = y
        self.width = width
        self.height = height

    def area(self):
        return self.width * self.height

    def corners(self):
        return (self.x, self.y, self.x + self.width, self.y + self.height)


def parse_drop_data(text):
    paths = []
    current = []
    in_braces = False
    for char in text.strip():
        if char == "{" and not in_braces:
            in_braces = True
            continue
        if char == "}" and in_braces:
            in_braces = False
            paths.append("".join(current).strip())
            current = []
            continue
        if char.isspace() and not in_braces:
            if current:
                paths.append("".join(current).strip())
                current = []
            continue
        current.append(char)
    if current:
        paths.append("".join(current).strip())
    return [path for path in paths if path]


def is_image_file(path):
    return os.path.splitext(path)[1].lower() in config.IMAGE_EXTENSIONS


def read_image(path):
    try:
        raw = np.fromfile(path, dtype=np.uint8)
        if raw.size == 0:
            return None
        return cv2.imdecode(raw, cv2.IMREAD_COLOR)
    except (OSError, ValueError):
        return None


def downscale_max_width(bgr, max_width):
    height, width = bgr.shape[0], bgr.shape[1]
    if width <= max_width:
        return bgr
    scale = max_width / width
    return cv2.resize(bgr, (max_width, int(height * scale)))


def clamp01(value):
    return max(0.0, min(1.0, value))


def clamp(value, low, high):
    return max(low, min(high, value))


def lerp(low, high, ratio):
    return low + (high - low) * clamp01(ratio)


def l2_normalize(vector):
    norm = float(np.linalg.norm(vector))
    if norm < config.EPS:
        return vector
    return vector / norm


def cosine_similarity(first, second):
    denom = float(np.linalg.norm(first) * np.linalg.norm(second))
    if denom < config.EPS:
        return 0.0
    return float(np.dot(first, second) / denom)


def order_yunet_landmarks(raw):
    eyes = sorted([raw[0], raw[1]], key=lambda point: point[0])
    mouth = sorted([raw[3], raw[4]], key=lambda point: point[0])
    return np.array([eyes[0], eyes[1], raw[2], mouth[0], mouth[1]], dtype=np.float32)


def align_face(bgr, ordered):
    template = np.array(config.ARCFACE_TEMPLATE, dtype=np.float32)
    matrix, _ = cv2.estimateAffinePartial2D(ordered, template)
    if matrix is None:
        return None
    size = (config.FACE_CROP_SIZE, config.FACE_CROP_SIZE)
    return cv2.warpAffine(bgr, matrix, size, borderValue=config.WARP_BORDER)


def mirror_crop(crop):
    return cv2.flip(crop, config.FLIP_HORIZONTAL)


def estimate_yaw(ordered):
    left_eye = ordered[0]
    right_eye = ordered[1]
    nose = ordered[2]
    eye_mid_x = (left_eye[0] + right_eye[0]) / 2.0
    eye_dist = float(np.linalg.norm(left_eye - right_eye))
    if eye_dist < config.MIN_EYE_DIST:
        return config.YAW_MAX
    offset = abs(float(nose[0]) - eye_mid_x) / eye_dist
    return min(offset * config.YAW_GAIN, config.YAW_MAX)


def skin_ratio(crop):
    region = crop[config.OCCLUDE_REGION_TOP:config.OCCLUDE_REGION_BOTTOM, config.OCCLUDE_REGION_LEFT:config.OCCLUDE_REGION_RIGHT]
    ycrcb = cv2.cvtColor(region, cv2.COLOR_BGR2YCrCb)
    lower = np.array([config.CHANNEL_MIN, config.SKIN_CR_MIN, config.SKIN_CB_MIN], dtype=np.uint8)
    upper = np.array([config.CHANNEL_MAX, config.SKIN_CR_MAX, config.SKIN_CB_MAX], dtype=np.uint8)
    mask = cv2.inRange(ycrcb, lower, upper)
    return float(np.mean(mask > 0))


def is_occluded(crop):
    return skin_ratio(crop) < config.OCCLUDE_SKIN_RATIO


def iou(first, second):
    first_corners = first.corners()
    second_corners = second.corners()
    x_left = max(first_corners[0], second_corners[0])
    y_top = max(first_corners[1], second_corners[1])
    x_right = min(first_corners[2], second_corners[2])
    y_bottom = min(first_corners[3], second_corners[3])
    inter_width = max(0, x_right - x_left)
    inter_height = max(0, y_bottom - y_top)
    inter = inter_width * inter_height
    union = first.area() + second.area() - inter
    if union <= 0:
        return 0.0
    return inter / union


def clip_box(x, y, width, height, frame_width, frame_height):
    fixed_x = max(0, min(x, frame_width - 1))
    fixed_y = max(0, min(y, frame_height - 1))
    fixed_w = max(0, min(width, frame_width - fixed_x))
    fixed_h = max(0, min(height, frame_height - fixed_y))
    return Box(fixed_x, fixed_y, fixed_w, fixed_h)


def box_center_size(box):
    return (box.x + box.width / 2.0, box.y + box.height / 2.0, float(box.width))


def format_score(value):
    return f"{value:.2f}"
