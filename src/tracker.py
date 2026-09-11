import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFont

import config
import utils


COLORS_BGR = {
    config.COLOR_NAME_GREEN: config.COLOR_GREEN,
    config.COLOR_NAME_YELLOW: config.COLOR_YELLOW,
    config.COLOR_NAME_RED: config.COLOR_RED,
}


class Track:
    def __init__(self, box, landmarks, live_state):
        self.box = box
        self.landmarks = landmarks
        self.yaw = 0.0
        self.occluded = False
        self.score_raw = config.MIN_COSINE
        self.score_smooth = config.MIN_COSINE
        self.has_score = False
        self.live_score = config.LIVENESS_START
        self.verdict = config.VERDICT_UNCERTAIN
        self.color = config.COLOR_NAME_RED
        self.label = ""
        self.missing = 0
        self.since_embed = 0
        self.live_state = live_state
        self.last_crop = None


def associate(tracks, detections):
    scored = []
    for track_index, track in enumerate(tracks):
        for det_index, det in enumerate(detections):
            overlap = utils.iou(track.box, det.box)
            if overlap >= config.TRACK_IOU_MATCH:
                scored.append((overlap, track_index, det_index))
    scored.sort(key=lambda item: item[0], reverse=True)
    pairs = []
    used_tracks = set()
    used_dets = set()
    for _, track_index, det_index in scored:
        if track_index in used_tracks or det_index in used_dets:
            continue
        used_tracks.add(track_index)
        used_dets.add(det_index)
        pairs.append((track_index, det_index))
    return pairs, used_tracks, used_dets


def center_shift(old, new):
    old_cx, old_cy, old_w = utils.box_center_size(old)
    new_cx, new_cy, _ = utils.box_center_size(new)
    if old_w < config.EPS:
        return 0.0
    dist = float(np.sqrt((new_cx - old_cx) ** 2 + (new_cy - old_cy) ** 2))
    return dist / old_w


def decide_color(score, bonus, green_thr, yellow_thr, verdict):
    if verdict == config.VERDICT_SPOOF:
        return config.COLOR_NAME_RED
    if score >= green_thr:
        if verdict == config.VERDICT_LIVE:
            return config.COLOR_NAME_GREEN
        return config.COLOR_NAME_YELLOW
    if score + bonus >= yellow_thr:
        return config.COLOR_NAME_YELLOW
    return config.COLOR_NAME_RED


def build_label(track, refs_empty):
    if refs_empty:
        return ""
    parts = [utils.format_score(track.score_smooth)]
    if track.occluded:
        parts.append(config.TAG_MASK)
    if track.yaw > config.YAW_PROFILE_DEGREES:
        parts.append(config.TAG_PROFILE)
    if track.verdict == config.VERDICT_SPOOF:
        parts.append(config.TAG_SPOOF)
    return " ".join(parts)


class TrackManager:
    def __init__(self, engine, analyzer, store):
        self.engine = engine
        self.analyzer = analyzer
        self.store = store
        self.tracks = []

    def reset(self):
        self.tracks = []

    def drop_stale_scores(self):
        if self.store.empty():
            for track in self.tracks:
                track.has_score = False
                track.score_smooth = config.MIN_COSINE

    def update(self, detections, bgr, gray, green_thr, yellow_thr):
        self.drop_stale_scores()
        pairs, used_tracks, used_dets = associate(self.tracks, detections)
        for track_index, det_index in pairs:
            self.refresh(self.tracks[track_index], detections[det_index], bgr, gray, green_thr, yellow_thr)
        old_count = len(self.tracks)
        for track_index in range(old_count):
            if track_index not in used_tracks:
                self.tracks[track_index].missing += 1
        for det_index, det in enumerate(detections):
            if det_index not in used_dets:
                self.tracks.append(self.spawn(det, bgr, gray, green_thr, yellow_thr))
        self.tracks = [track for track in self.tracks if track.missing <= config.TRACK_MAX_MISSING]
        return [track for track in self.tracks if track.missing == 0]

    def needs_embed(self, track, moved):
        if self.store.empty():
            return False
        if not track.has_score:
            return True
        if track.since_embed >= config.RECOGNIZE_EVERY_FRAMES:
            return True
        return moved > config.REFRESH_MOVE_RATIO

    def apply_match(self, track, vector):
        raw, _ = self.store.match(vector)
        track.score_raw = raw
        if track.has_score:
            alpha = config.SCORE_SMOOTH_ALPHA
            track.score_smooth = alpha * raw + (1.0 - alpha) * track.score_smooth
        else:
            track.score_smooth = raw
            track.has_score = True
        track.since_embed = 0

    def apply_crop(self, track, crop):
        if crop is None:
            return None
        track.last_crop = crop
        track.occluded = utils.is_occluded(crop)
        return crop

    def refresh(self, track, det, bgr, gray, green_thr, yellow_thr):
        moved = center_shift(track.box, det.box)
        track.box = det.box
        track.landmarks = det.landmarks
        track.missing = 0
        track.yaw = utils.estimate_yaw(det.landmarks)
        track.since_embed += 1
        fresh_crop = None
        if self.needs_embed(track, moved):
            vector, crop = self.engine.embed_face(bgr, det.landmarks)
            if vector is not None:
                self.apply_match(track, vector)
            fresh_crop = self.apply_crop(track, crop)
        track.live_score, track.verdict = self.analyzer.update(track.live_state, gray, det.box, det.landmarks, fresh_crop)
        self.decide(track, green_thr, yellow_thr)

    def spawn(self, det, bgr, gray, green_thr, yellow_thr):
        track = Track(det.box, det.landmarks, self.analyzer.new_state())
        track.yaw = utils.estimate_yaw(det.landmarks)
        fresh_crop = None
        if not self.store.empty():
            vector, crop = self.engine.embed_face(bgr, det.landmarks)
            if vector is not None:
                self.apply_match(track, vector)
            fresh_crop = self.apply_crop(track, crop)
        track.live_score, track.verdict = self.analyzer.update(track.live_state, gray, det.box, det.landmarks, fresh_crop)
        self.decide(track, green_thr, yellow_thr)
        return track

    def decide(self, track, green_thr, yellow_thr):
        track.color = decide_color(track.score_smooth, self.bonus(track), green_thr, yellow_thr, track.verdict)
        track.label = build_label(track, self.store.empty())

    def bonus(self, track):
        if track.occluded or track.yaw > config.YAW_PROFILE_DEGREES:
            return config.YELLOW_BONUS_MASKED
        return 0.0


class LabelFonts:
    def __init__(self):
        self.label = load_font(config.LABEL_FONT_SIZE)
        self.hint = load_font(config.HINT_FONT_SIZE)


def load_font(size):
    for path in (config.WINDOWS_ARIAL, config.LINUX_DEJAVU):
        try:
            return ImageFont.truetype(path, size)
        except OSError:
            continue
    return ImageFont.load_default()


def draw_tracks(bgr, tracks, fonts, hint):
    for track in tracks:
        color = COLORS_BGR.get(track.color, config.COLOR_RED)
        box = track.box
        cv2.rectangle(bgr, (box.x, box.y), (box.x + box.width, box.y + box.height), color, config.BOX_THICKNESS)
    draw_texts(bgr, tracks, fonts, hint)


def draw_texts(bgr, tracks, fonts, hint):
    wants_labels = any(track.label for track in tracks)
    if not wants_labels and not hint:
        return
    rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
    canvas = Image.fromarray(rgb)
    pen = ImageDraw.Draw(canvas)
    for track in tracks:
        if track.label:
            place_label(pen, track, fonts.label)
    if hint:
        place_hint(pen, canvas.size[0], canvas.size[1], hint, fonts.hint)
    painted = np.array(canvas)
    cv2.cvtColor(painted, cv2.COLOR_RGB2BGR, dst=bgr)


def place_label(pen, track, font):
    box = track.box
    bounds = pen.textbbox((0, 0), track.label, font=font)
    text_w = bounds[2] - bounds[0]
    text_h = bounds[3] - bounds[1]
    x = box.x
    y = box.y - text_h - config.LABEL_PAD_Y * 2
    if y < 0:
        y = box.y + box.height
    pen.rectangle([x, y, x + text_w + config.LABEL_PAD_X * 2, y + text_h + config.LABEL_PAD_Y * 2], fill=config.LABEL_BG)
    pen.text((x + config.LABEL_PAD_X, y + config.LABEL_PAD_Y), track.label, font=font, fill=config.LABEL_FG)


def place_hint(pen, wide, tall, hint, font):
    bounds = pen.textbbox((0, 0), hint, font=font)
    text_w = bounds[2] - bounds[0]
    text_h = bounds[3] - bounds[1]
    x = int((wide - text_w) / 2.0)
    y = int((tall - text_h) / 2.0)
    pen.rectangle(
        [x - config.LABEL_PAD_X, y - config.LABEL_PAD_Y, x + text_w + config.LABEL_PAD_X, y + text_h + config.LABEL_PAD_Y],
        fill=config.HINT_BG,
    )
    pen.text((x, y), hint, font=font, fill=config.HINT_FG)
