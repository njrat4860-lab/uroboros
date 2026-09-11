from collections import deque

import cv2
import numpy as np

import config
import utils


class LivenessState:
    def __init__(self):
        self.centers = deque(maxlen=config.MOTION_HISTORY)
        self.eyes = deque(maxlen=config.EYE_HISTORY)
        self.score = config.LIVENESS_START
        self.frames = 0
        self.texture = config.NEUTRAL_CUE
        self.moire = config.NEUTRAL_CUE
        self.specular = config.NEUTRAL_CUE


def expand_box(box, factor, frame_width, frame_height):
    wide = box.width * factor
    tall = box.height * factor
    x = int(box.x + (box.width - wide) / 2.0)
    y = int(box.y + (box.height - tall) / 2.0)
    return utils.clip_box(x, y, int(wide), int(tall), frame_width, frame_height)


def crop_around(gray, point, wide, tall):
    height, width = gray.shape[0], gray.shape[1]
    x = int(point[0] - wide / 2.0)
    y = int(point[1] - tall / 2.0)
    box = utils.clip_box(x, y, wide, tall, width, height)
    return gray[box.y:box.y + box.height, box.x:box.x + box.width]


def edge_energy(patch):
    if patch.size == 0:
        return 0.0
    grad = cv2.Sobel(patch, cv2.CV_32F, config.SOBEL_X_ORDER, config.SOBEL_Y_ORDER, ksize=config.SOBEL_KERNEL)
    return float(np.mean(np.abs(grad)))


def eye_energy_now(gray, landmarks):
    left = crop_around(gray, landmarks[0], config.EYE_CROP_WIDTH, config.EYE_CROP_HEIGHT)
    right = crop_around(gray, landmarks[1], config.EYE_CROP_WIDTH, config.EYE_CROP_HEIGHT)
    return (edge_energy(left) + edge_energy(right)) / 2.0


def eye_cue(state):
    if len(state.eyes) < config.EYE_MIN_FRAMES:
        return config.NEUTRAL_CUE
    values = np.array(state.eyes, dtype=np.float64)
    spread = float(np.max(values) - np.min(values))
    mean = float(np.mean(values))
    if mean < config.EPS:
        return config.NEUTRAL_CUE
    return utils.clamp01((spread / mean) / config.EYE_RANGE_NORM)


def motion_cue(state):
    if len(state.centers) < config.MOTION_MIN_FRAMES:
        return config.NEUTRAL_CUE
    points = np.array(state.centers, dtype=np.float64)
    widths = points[:, 2]
    if np.any(widths < config.EPS):
        return config.NEUTRAL_CUE
    norm_x = points[:, 0] / widths
    norm_y = points[:, 1] / widths
    var = (float(np.var(norm_x)) + float(np.var(norm_y))) / 2.0
    level = float(np.log10(var + config.EPS))
    ratio = (level - config.MOTION_LOG_LOW) / (config.MOTION_LOG_HIGH - config.MOTION_LOG_LOW)
    return utils.lerp(config.MOTION_SCORE_LOW, config.MOTION_SCORE_HIGH, ratio)


def texture_cue(crop):
    gray = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY)
    center = gray[config.TEXTURE_CENTER_TOP:config.TEXTURE_CENTER_BOTTOM, config.TEXTURE_CENTER_LEFT:config.TEXTURE_CENTER_RIGHT]
    sharp = float(cv2.Laplacian(center, cv2.CV_64F).var())
    if sharp < config.TEXTURE_SOFT:
        return config.TEXTURE_SCORE_MIN
    if sharp < config.TEXTURE_GOOD:
        return utils.lerp(config.TEXTURE_SCORE_MIN, config.TEXTURE_SCORE_MID, (sharp - config.TEXTURE_SOFT) / (config.TEXTURE_GOOD - config.TEXTURE_SOFT))
    if sharp < config.TEXTURE_GREAT:
        return utils.lerp(config.TEXTURE_SCORE_MID, config.TEXTURE_SCORE_MAX, (sharp - config.TEXTURE_GOOD) / (config.TEXTURE_GREAT - config.TEXTURE_GOOD))
    if sharp < config.TEXTURE_NOISY:
        return config.TEXTURE_SCORE_MAX
    return config.TEXTURE_SCORE_NOISY


def moire_ok(crop):
    gray = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY)
    small = cv2.resize(gray, (config.MOIRE_SIZE, config.MOIRE_SIZE))
    spectrum = np.fft.fftshift(np.fft.fft2(small))
    mag = np.log1p(np.abs(spectrum))
    half = config.MOIRE_SIZE / 2.0
    rows, cols = np.mgrid[0:config.MOIRE_SIZE, 0:config.MOIRE_SIZE]
    dist = np.sqrt((rows - half) ** 2 + (cols - half) ** 2)
    low = mag[dist <= config.MOIRE_LOW_RADIUS]
    ring = mag[(dist >= config.MOIRE_RING_INNER) & (dist <= config.MOIRE_RING_OUTER)]
    low_mean = float(np.mean(low))
    ring_mean = float(np.mean(ring))
    ring_std = float(np.std(ring))
    peaks = int(np.sum(ring > ring_mean + config.MOIRE_PEAK_SIGMA * ring_std))
    peak_part = min(peaks / config.MOIRE_PEAK_NORM, 1.0)
    ratio_part = utils.clamp01((ring_mean / (low_mean + config.EPS) - config.MOIRE_RATIO_BASE) / config.MOIRE_RATIO_SPAN)
    screen = config.MOIRE_PEAK_WEIGHT * peak_part + config.MOIRE_RATIO_WEIGHT * ratio_part
    if screen < config.MOIRE_STRONG_AT:
        return 1.0
    return utils.clamp01(1.0 - screen)


def band_of(mid_x, mid_y, wide, tall, horizontal):
    top_edge = tall * config.BAND_EDGE
    bottom_edge = tall * (1.0 - config.BAND_EDGE)
    left_edge = wide * config.BAND_EDGE
    right_edge = wide * (1.0 - config.BAND_EDGE)
    if horizontal:
        if mid_y < top_edge:
            return config.BORDER_BAND_TOP
        if mid_y > bottom_edge:
            return config.BORDER_BAND_BOTTOM
        return None
    if mid_x < left_edge:
        return config.BORDER_BAND_LEFT
    if mid_x > right_edge:
        return config.BORDER_BAND_RIGHT
    return None


def border_score(gray, box):
    height, width = gray.shape[0], gray.shape[1]
    outer = expand_box(box, config.BORDER_EXPAND, width, height)
    patch = gray[outer.y:outer.y + outer.height, outer.x:outer.x + outer.width]
    edges = cv2.Canny(patch, config.BORDER_CANNY_LOW, config.BORDER_CANNY_HIGH)
    longest = box.width * config.BORDER_LINE_RATIO
    lines = cv2.HoughLinesP(edges, config.HOUGH_RHO, np.pi / 180.0, config.BORDER_HOUGH_THRESHOLD, minLineLength=longest, maxLineGap=config.BORDER_MAX_LINE_GAP)
    if lines is None:
        return 0.0
    bands = set()
    for line in lines.reshape(-1, config.HOUGH_LINE_DIM):
        x1, y1, x2, y2 = line
        horizontal = abs(x2 - x1) >= abs(y2 - y1)
        mid_x = (x1 + x2) / 2.0
        mid_y = (y1 + y2) / 2.0
        band = band_of(mid_x, mid_y, patch.shape[1], patch.shape[0], horizontal)
        if band is not None:
            bands.add(band)
    return len(bands) / config.BORDER_BANDS


def specular_ok(crop):
    hsv = cv2.cvtColor(crop, cv2.COLOR_BGR2HSV)
    hot = (hsv[:, :, 1] <= config.SPECULAR_SAT_MAX) & (hsv[:, :, 2] >= config.SPECULAR_VALUE_MIN)
    frac = float(np.mean(hot))
    if frac <= config.SPECULAR_MILD:
        return 1.0
    if frac >= config.SPECULAR_BAD:
        return config.SPECULAR_SCORE_BAD
    return utils.lerp(1.0, config.SPECULAR_SCORE_BAD, (frac - config.SPECULAR_MILD) / (config.SPECULAR_BAD - config.SPECULAR_MILD))


def combine(motion, texture, moire, border_raw, specular, eye):
    border_ok = 1.0 - border_raw
    score = (
        config.WEIGHT_MOTION * motion
        + config.WEIGHT_TEXTURE * texture
        + config.WEIGHT_MOIRE * moire
        + config.WEIGHT_BORDER * border_ok
        + config.WEIGHT_SPECULAR * specular
        + config.WEIGHT_EYE * eye
    )
    if border_raw > config.LIVENESS_BORDER_STRONG:
        score = min(score, config.LIVENESS_BORDER_CAP)
    if motion < config.LIVENESS_STATIC_MOTION and texture < config.LIVENESS_FLAT_TEXTURE:
        score = min(score, config.LIVENESS_STATIC_FLAT_CAP)
    return score


def verdict_of(score):
    if score < config.LIVENESS_SPOOF_BELOW:
        return config.VERDICT_SPOOF
    if score < config.LIVENESS_LIVE_ABOVE:
        return config.VERDICT_UNCERTAIN
    return config.VERDICT_LIVE


class LivenessAnalyzer:
    def new_state(self):
        return LivenessState()

    def update(self, state, gray, box, landmarks, crop):
        state.frames += 1
        cx, cy, wide = utils.box_center_size(box)
        state.centers.append((cx, cy, wide))
        state.eyes.append(eye_energy_now(gray, landmarks))
        if crop is not None:
            state.texture = texture_cue(crop)
            state.moire = moire_ok(crop)
            state.specular = specular_ok(crop)
        motion = motion_cue(state)
        eye = eye_cue(state)
        raw_border = border_score(gray, box)
        raw = combine(motion, state.texture, state.moire, raw_border, state.specular, eye)
        if state.frames == 1:
            state.score = raw
        else:
            alpha = config.LIVENESS_SMOOTH_ALPHA
            state.score = alpha * raw + (1.0 - alpha) * state.score
        return state.score, verdict_of(state.score)
