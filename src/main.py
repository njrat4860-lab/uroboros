import os
import queue
import sys
import threading
import time

CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, CURRENT_DIR)

import cv2
import numpy as np

import config
import face_engine
import liveness
import models
import tracker
import utils

try:
    import tkinter as tk
    from tkinter import filedialog
    TK_READY = True
except ImportError:
    TK_READY = False

try:
    from tkinterdnd2 import DND_FILES, TkinterDnD
    DND_READY = True
except ImportError:
    DND_READY = False

try:
    from PIL import Image, ImageTk
    PIL_READY = True
except ImportError:
    PIL_READY = False


class Worker:
    def __init__(self, inbox, events, frames):
        self.inbox = inbox
        self.events = events
        self.frames = frames
        self.store = face_engine.ReferenceStore()
        self.analyzer = liveness.LivenessAnalyzer()
        self.engine = None
        self.manager = None
        self.fonts = None
        self.green = config.GREEN_THRESHOLD_DEFAULT
        self.yellow = config.YELLOW_THRESHOLD_DEFAULT
        self.camera_index = config.CAMERA_INDEXES[0]
        self.capture = None
        self.models_ok = False
        self.models_failed = False
        self.running = True
        self.fail_count = 0
        self.fps = 0.0
        self.last_moment = 0.0
        self.last_stats = 0.0
        self.last_camera_try = 0.0
        self.last_progress = 0.0

    def send_event(self, kind, data):
        self.events.put((kind, data))

    def send_status(self, text):
        self.send_event(config.EV_STATUS, text)

    def progress(self, label, done, total):
        now = time.monotonic()
        if now - self.last_progress < config.PROGRESS_THROTTLE:
            return
        self.last_progress = now
        done_mb = str(int(done / config.BYTES_IN_MB))
        if total and total > 0:
            total_mb = str(int(total / config.BYTES_IN_MB))
            self.send_status(label + config.SEP_COLON + done_mb + config.SEP_SLASH + total_mb + config.MB_SUFFIX)
        else:
            self.send_status(label + config.SEP_COLON + done_mb + config.MB_SUFFIX)

    def run(self):
        self.fonts = tracker.LabelFonts()
        self.send_status(config.STATUS_START)
        self.load_models()
        while self.running:
            self.drain_inbox()
            if not self.running:
                break
            frame = self.grab_frame()
            if frame is None:
                time.sleep(config.IDLE_SLEEP)
                continue
            self.tick_fps()
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            tracks = []
            if self.models_ok:
                detections = self.engine.detect(frame)
                tracks = self.manager.update(detections, frame, gray, self.green, self.yellow)
            tracker.draw_tracks(frame, tracks, self.fonts, self.hint_text())
            self.push_frame(frame)
            self.push_stats(len(tracks))
        self.close_camera()

    def fail_models(self, message):
        self.models_ok = False
        self.models_failed = True
        self.send_event(config.EV_MODELS, config.MODELS_FAIL)
        self.send_status(message)

    def load_models(self):
        self.send_status(config.STATUS_DOWNLOAD)
        try:
            detector_path, recognition_path = models.ensure_models(self.progress)
        except models.ModelError as exc:
            self.fail_models(str(exc))
            return
        try:
            self.engine = face_engine.FaceEngine(detector_path, recognition_path)
        except Exception:
            self.engine = None
            self.fail_models(config.STATUS_ENGINE_FAIL)
            return
        self.manager = tracker.TrackManager(self.engine, self.analyzer, self.store)
        self.models_ok = True
        self.models_failed = False
        self.send_event(config.EV_MODELS, config.MODELS_OK)
        if self.engine.session is None:
            self.send_status(config.STATUS_NO_ORT)
        else:
            self.send_status(config.STATUS_READY)

    def drain_inbox(self):
        while self.running:
            try:
                kind, data = self.inbox.get_nowait()
            except queue.Empty:
                return
            self.handle_op(kind, data)

    def handle_op(self, kind, data):
        if kind == config.OP_ADD_REF:
            self.add_reference(data)
        elif kind == config.OP_REMOVE_REF:
            self.store.remove(data)
            self.send_event(config.EV_REF_REMOVED, data)
        elif kind == config.OP_CLEAR_REFS:
            self.store.clear()
            self.send_event(config.EV_REFS_CLEARED, None)
        elif kind == config.OP_THRESHOLDS:
            green, yellow = data
            if green < yellow + config.THRESHOLD_MIN_GAP:
                green = yellow + config.THRESHOLD_MIN_GAP
            self.green = green
            self.yellow = yellow
        elif kind == config.OP_CAMERA:
            self.camera_index = data
            if self.manager is not None:
                self.manager.reset()
            self.open_camera()
        elif kind == config.OP_RETRY:
            self.load_models()
        elif kind == config.OP_STOP:
            self.running = False

    def add_reference(self, path):
        if self.engine is None:
            self.send_event(config.EV_REF_ERROR, config.STATUS_NO_MODELS_YET)
            return
        if self.engine.session is None:
            self.send_event(config.EV_REF_ERROR, config.STATUS_NO_ORT)
            return
        if self.store.count() >= config.MAX_REFERENCES:
            self.send_event(config.EV_REF_ERROR, config.STATUS_REF_LIMIT)
            return
        image = utils.read_image(path)
        if image is None:
            self.send_event(config.EV_REF_ERROR, config.STATUS_BAD_FILE + path)
            return
        prepared = utils.downscale_max_width(image, config.REFERENCE_DETECT_WIDTH)
        best = self.engine.largest(self.engine.detect(prepared))
        if best is None:
            self.send_event(config.EV_REF_ERROR, config.STATUS_NO_FACE + path)
            return
        vector, crop = self.engine.embed_face(prepared, best.landmarks)
        if vector is None or crop is None:
            self.send_event(config.EV_REF_ERROR, config.STATUS_BAD_FILE + path)
            return
        mirror = self.engine.embed_crop(utils.mirror_crop(crop))
        if mirror is None:
            mirror = vector
        self.store.add(vector, mirror)
        thumb = cv2.resize(crop, (config.THUMB_SIZE, config.THUMB_SIZE))
        self.send_event(config.EV_REF_ADDED, thumb)
        self.send_status(config.STATUS_REF_ADDED + config.SEP_COLON + str(self.store.count()))

    def open_camera(self):
        self.close_camera()
        if config.USE_WINDOWS_BACKEND:
            self.capture = cv2.VideoCapture(self.camera_index, cv2.CAP_DSHOW)
        else:
            self.capture = cv2.VideoCapture(self.camera_index, cv2.CAP_ANY)
        if self.capture is not None and self.capture.isOpened():
            self.capture.set(cv2.CAP_PROP_FRAME_WIDTH, config.CAMERA_WIDTH)
            self.capture.set(cv2.CAP_PROP_FRAME_HEIGHT, config.CAMERA_HEIGHT)
            self.capture.set(cv2.CAP_PROP_FPS, config.CAMERA_FPS)
            self.fail_count = 0
            self.send_status(config.STATUS_CAMERA + str(self.camera_index + 1))
        else:
            self.capture = None
            self.send_status(config.STATUS_NO_CAMERA)

    def close_camera(self):
        if self.capture is not None:
            try:
                self.capture.release()
            except Exception:
                pass
            self.capture = None

    def grab_frame(self):
        if self.capture is None:
            now = time.monotonic()
            if now - self.last_camera_try >= config.CAMERA_RETRY_PAUSE:
                self.last_camera_try = now
                self.open_camera()
            return None
        ok, frame = self.capture.read()
        if not ok or frame is None:
            self.fail_count += 1
            if self.fail_count >= config.CAMERA_FAILS_BEFORE_REOPEN:
                self.close_camera()
                self.send_status(config.STATUS_NO_CAMERA)
            return None
        self.fail_count = 0
        return frame

    def tick_fps(self):
        now = time.monotonic()
        if self.last_moment > 0.0:
            delta = now - self.last_moment
            if delta < config.MIN_DT:
                delta = config.MIN_DT
            instant = 1.0 / delta
            self.fps = config.FPS_SMOOTH * self.fps + (1.0 - config.FPS_SMOOTH) * instant
        self.last_moment = now

    def hint_text(self):
        if self.models_failed:
            return None
        if not self.models_ok:
            return config.STATUS_DOWNLOAD
        if self.store.empty():
            return config.HINT_NO_REFS
        return None

    def push_frame(self, frame):
        try:
            while True:
                self.frames.get_nowait()
        except queue.Empty:
            pass
        self.frames.put(frame)

    def push_stats(self, faces):
        now = time.monotonic()
        if now - self.last_stats < config.STATS_EVERY_SECONDS:
            return
        self.last_stats = now
        if self.engine is not None:
            device = self.engine.device
        else:
            device = config.DEVICE_LABEL_NONE
        self.send_event(config.EV_STATS, (self.fps, device, faces, self.store.count()))


class App:
    def __init__(self, root, inbox, events, frames):
        self.root = root
        self.inbox = inbox
        self.events = events
        self.frames = frames
        self.thumbs = []
        self.photo = None
        self.refs = 0
        self.camera_pos = 0
        self.sliders_locked = False
        self.green_var = tk.DoubleVar(value=config.GREEN_THRESHOLD_DEFAULT)
        self.yellow_var = tk.DoubleVar(value=config.YELLOW_THRESHOLD_DEFAULT)
        self.build()
        self.poll()

    def send_op(self, kind, data):
        self.inbox.put((kind, data))

    def build(self):
        self.root.title(config.WINDOW_TITLE)
        self.root.geometry(str(config.WINDOW_WIDTH) + config.GEOMETRY_SEPARATOR + str(config.WINDOW_HEIGHT))
        self.root.minsize(config.WINDOW_MIN_WIDTH, config.WINDOW_MIN_HEIGHT)
        self.root.protocol(config.WM_CLOSE_PROTOCOL, self.close)
        top = tk.Frame(self.root)
        top.pack(fill=tk.X, padx=config.PAD_X, pady=config.PAD_Y)
        self.clear_button = tk.Button(top, text=config.BUTTON_CLEAR, command=self.clear_refs)
        self.clear_button.pack(side=tk.LEFT)
        self.camera_button = tk.Button(top, text=config.BUTTON_CAMERA, command=self.next_camera)
        self.camera_button.pack(side=tk.LEFT, padx=config.PAD_X)
        self.retry_button = tk.Button(top, text=config.BUTTON_RETRY, command=self.retry_models)
        self.add_button = tk.Button(top, text=config.BUTTON_ADD_PHOTO, command=self.pick_photos)
        if not DND_READY:
            self.add_button.pack(side=tk.LEFT, padx=config.PAD_X)
        self.green_scale = tk.Scale(
            top,
            label=config.SLIDER_GREEN,
            orient=tk.HORIZONTAL,
            from_=config.THRESHOLD_SLIDER_MIN,
            to=config.THRESHOLD_SLIDER_MAX,
            resolution=config.THRESHOLD_SLIDER_STEP,
            variable=self.green_var,
            command=lambda _: self.thresholds_changed(),
            length=config.SLIDER_LENGTH,
        )
        self.green_scale.pack(side=tk.LEFT, padx=config.PAD_X)
        self.yellow_scale = tk.Scale(
            top,
            label=config.SLIDER_YELLOW,
            orient=tk.HORIZONTAL,
            from_=config.THRESHOLD_SLIDER_MIN,
            to=config.THRESHOLD_SLIDER_MAX,
            resolution=config.THRESHOLD_SLIDER_STEP,
            variable=self.yellow_var,
            command=lambda _: self.thresholds_changed(),
            length=config.SLIDER_LENGTH,
        )
        self.yellow_scale.pack(side=tk.LEFT, padx=config.PAD_X)
        self.stats_label = tk.Label(top, text="")
        self.stats_label.pack(side=tk.RIGHT)
        legend = tk.Frame(self.root)
        legend.pack(fill=tk.X, padx=config.PAD_X)
        tk.Label(legend, text=config.LABEL_GREEN, bg=config.COLOR_NAME_GREEN, fg=config.FG_WHITE, width=config.LEGEND_WIDTH).pack(side=tk.LEFT)
        tk.Label(legend, text=config.LABEL_YELLOW, bg=config.COLOR_NAME_YELLOW, fg=config.FG_BLACK, width=config.LEGEND_WIDTH).pack(side=tk.LEFT, padx=config.PAD_X)
        tk.Label(legend, text=config.LABEL_RED, bg=config.COLOR_NAME_RED, fg=config.FG_WHITE, width=config.LEGEND_WIDTH).pack(side=tk.LEFT)
        refs_row = tk.Frame(self.root)
        refs_row.pack(fill=tk.X, padx=config.PAD_X, pady=config.PAD_Y)
        tk.Label(refs_row, text=config.REFS_TITLE).pack(side=tk.LEFT)
        self.thumbs_row = tk.Frame(refs_row)
        self.thumbs_row.pack(side=tk.LEFT, padx=config.PAD_X)
        self.video_label = tk.Label(self.root)
        self.video_label.pack(padx=config.PAD_X, pady=config.PAD_Y)
        self.show_placeholder()
        self.status_label = tk.Label(self.root, text=config.STATUS_START, anchor=tk.W, justify=tk.LEFT, wraplength=config.STATUS_WRAP)
        self.status_label.pack(fill=tk.X, padx=config.PAD_X, pady=config.PAD_Y)
        if DND_READY:
            for target in (self.root, self.video_label, self.thumbs_row):
                if hasattr(target, "drop_target_register"):
                    target.drop_target_register(DND_FILES)
                    target.dnd_bind(config.DROP_EVENT, self.on_drop)

    def show_placeholder(self):
        blank = np.zeros((config.DISPLAY_HEIGHT, config.DISPLAY_WIDTH, 3), dtype=np.uint8)
        blank[:, :] = config.PLACEHOLDER_GRAY
        rgb = cv2.cvtColor(blank, cv2.COLOR_BGR2RGB)
        self.photo = ImageTk.PhotoImage(Image.fromarray(rgb))
        self.video_label.configure(image=self.photo)

    def poll(self):
        self.poll_frame()
        self.poll_events()
        self.root.after(config.GUI_POLL_MS, self.poll)

    def poll_frame(self):
        latest = None
        while True:
            try:
                latest = self.frames.get_nowait()
            except queue.Empty:
                break
        if latest is None:
            return
        rgb = cv2.cvtColor(latest, cv2.COLOR_BGR2RGB)
        small = Image.fromarray(rgb).resize((config.DISPLAY_WIDTH, config.DISPLAY_HEIGHT), Image.BILINEAR)
        self.photo = ImageTk.PhotoImage(small)
        self.video_label.configure(image=self.photo)

    def poll_events(self):
        while True:
            try:
                kind, data = self.events.get_nowait()
            except queue.Empty:
                return
            self.handle_event(kind, data)

    def handle_event(self, kind, data):
        if kind == config.EV_STATUS:
            self.status_label.configure(text=data)
        elif kind == config.EV_STATS:
            fps, device, faces, refs = data
            self.refs = refs
            self.stats_label.configure(text=self.stats_text(fps, device, faces, refs))
        elif kind == config.EV_MODELS:
            self.models_state(data)
        elif kind == config.EV_REF_ADDED:
            self.thumb_added(data)
        elif kind == config.EV_REF_ERROR:
            self.status_label.configure(text=data)
        elif kind == config.EV_REF_REMOVED:
            self.thumb_removed(data)
        elif kind == config.EV_REFS_CLEARED:
            self.thumbs = []
            self.refs = 0
            self.refresh_thumbs()
            self.status_label.configure(text=config.STATUS_REFS_CLEARED)

    def stats_text(self, fps, device, faces, refs):
        return (
            str(int(fps))
            + config.STATS_FPS_SUFFIX
            + config.SEP_BAR
            + device
            + config.SEP_BAR
            + config.STATS_FACES
            + str(faces)
            + config.SEP_BAR
            + config.STATS_REFS
            + str(refs)
        )

    def models_state(self, state):
        if state == config.MODELS_FAIL:
            self.retry_button.pack(side=tk.LEFT, padx=config.PAD_X)
        else:
            self.retry_button.pack_forget()

    def thumb_added(self, thumb_bgr):
        rgb = cv2.cvtColor(thumb_bgr, cv2.COLOR_BGR2RGB)
        photo = ImageTk.PhotoImage(Image.fromarray(rgb))
        self.thumbs.append(photo)
        self.refresh_thumbs()

    def thumb_removed(self, index):
        if 0 <= index < len(self.thumbs):
            del self.thumbs[index]
            self.refresh_thumbs()

    def refresh_thumbs(self):
        for child in self.thumbs_row.winfo_children():
            child.destroy()
        for index, photo in enumerate(self.thumbs):
            label = tk.Label(self.thumbs_row, image=photo)
            label.pack(side=tk.LEFT, padx=config.THUMB_PAD)
            label.bind(config.CLICK_EVENT, lambda event, pos=index: self.remove_ref(pos))

    def on_drop(self, event):
        for path in utils.parse_drop_data(event.data):
            self.offer_path(path)

    def offer_path(self, path):
        if not utils.is_image_file(path):
            self.status_label.configure(text=config.STATUS_NOT_IMAGE + path)
            return
        if self.refs >= config.MAX_REFERENCES:
            self.status_label.configure(text=config.STATUS_REF_LIMIT)
            return
        self.send_op(config.OP_ADD_REF, path)

    def pick_photos(self):
        paths = filedialog.askopenfilenames(title=config.BUTTON_ADD_PHOTO, filetypes=[(config.FILES_TITLE, config.FILES_MASK)])
        for path in paths:
            self.offer_path(path)

    def clear_refs(self):
        self.send_op(config.OP_CLEAR_REFS, None)

    def remove_ref(self, index):
        self.send_op(config.OP_REMOVE_REF, index)

    def next_camera(self):
        self.camera_pos = (self.camera_pos + 1) % len(config.CAMERA_INDEXES)
        self.send_op(config.OP_CAMERA, config.CAMERA_INDEXES[self.camera_pos])

    def retry_models(self):
        self.send_op(config.OP_RETRY, None)

    def thresholds_changed(self):
        if self.sliders_locked:
            return
        self.sliders_locked = True
        try:
            green = self.green_var.get()
            yellow = self.yellow_var.get()
            if green < yellow + config.THRESHOLD_MIN_GAP:
                green = yellow + config.THRESHOLD_MIN_GAP
                if green > config.THRESHOLD_SLIDER_MAX:
                    green = config.THRESHOLD_SLIDER_MAX
                    yellow = green - config.THRESHOLD_MIN_GAP
                    self.yellow_var.set(yellow)
                self.green_var.set(green)
            self.send_op(config.OP_THRESHOLDS, (green, yellow))
        finally:
            self.sliders_locked = False

    def close(self):
        self.send_op(config.OP_STOP, None)
        self.root.destroy()


def main():
    if not TK_READY:
        raise SystemExit(config.TK_MISSING_MESSAGE)
    if not PIL_READY:
        raise SystemExit(config.PIL_MISSING_MESSAGE)
    inbox = queue.Queue()
    events = queue.Queue()
    frames = queue.Queue(maxsize=config.FRAME_QUEUE_SIZE)
    worker = Worker(inbox, events, frames)
    thread = threading.Thread(target=worker.run, daemon=True)
    if DND_READY:
        root = TkinterDnD.Tk()
    else:
        root = tk.Tk()
    App(root, inbox, events, frames)
    thread.start()
    root.mainloop()


if __name__ == "__main__":
    main()
