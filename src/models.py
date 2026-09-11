import os
import shutil
import tempfile
import urllib.request
import zipfile

import config


class ModelError(Exception):
    pass


def ascii_only(text):
    return all(ord(char) < config.ASCII_LIMIT for char in text)


def ascii_safe_path(original, cached_name):
    if ascii_only(original):
        return original
    for folder in (config.ASCII_MODEL_CACHE_DIR, tempfile.gettempdir()):
        try:
            os.makedirs(folder, exist_ok=True)
        except OSError:
            continue
        cached = os.path.join(folder, cached_name)
        if not ascii_only(cached):
            continue
        try:
            if not os.path.exists(cached) or os.path.getsize(cached) != os.path.getsize(original):
                shutil.copyfile(original, cached)
            return cached
        except OSError:
            continue
    return original


def pick_model(paths):
    lowered = [(path, path.lower()) for path in paths]
    for path, low in lowered:
        if low.endswith(config.ONNX_EXTENSION) and config.RECOGNITION_PREFERRED_SUBSTRING in os.path.basename(low):
            return path
    for path, low in lowered:
        if low.endswith(config.ONNX_EXTENSION) and config.RECOGNITION_FALLBACK_SUBSTRING in os.path.basename(low):
            return path
    for path, low in lowered:
        if low.endswith(config.ONNX_EXTENSION):
            return path
    return None


def find_recognition_model():
    if not os.path.isdir(config.MODELS_DIR):
        return None
    paths = []
    for name in os.listdir(config.MODELS_DIR):
        if name == config.YUNET_MODEL_NAME:
            continue
        full = os.path.join(config.MODELS_DIR, name)
        if os.path.isfile(full):
            paths.append(full)
    return pick_model(paths)


def download(url, dest, label, progress):
    part = dest + ".part"
    try:
        if os.path.exists(part):
            os.remove(part)
    except OSError:
        pass

    def hook(blocks, block_size, total):
        progress(label, blocks * block_size, total)

    try:
        urllib.request.urlretrieve(url, part, reporthook=hook)
    except Exception as exc:
        try:
            os.remove(part)
        except OSError:
            pass
        raise ModelError(config.DOWNLOAD_FAIL_HEAD + label + config.DOWNLOAD_FAIL_TAIL) from exc
    os.replace(part, dest)


def extract_recognition(archive):
    try:
        bundle = zipfile.ZipFile(archive)
    except zipfile.BadZipFile as exc:
        raise ModelError(config.STATUS_BAD_ARCHIVE) from exc
    with bundle:
        chosen = pick_model(bundle.namelist())
        if chosen is None:
            raise ModelError(config.STATUS_NO_MODEL_IN_ARCHIVE)
        target = os.path.join(config.MODELS_DIR, os.path.basename(chosen))
        with open(target, "wb") as out:
            out.write(bundle.read(chosen))
    try:
        os.remove(archive)
    except OSError:
        pass
    return target


def ensure_models(progress):
    os.makedirs(config.MODELS_DIR, exist_ok=True)
    detector = os.path.join(config.MODELS_DIR, config.YUNET_MODEL_NAME)
    if not os.path.exists(detector):
        download(config.YUNET_MODEL_URL, detector, config.YUNET_DOWNLOAD_LABEL, progress)
    recognition = find_recognition_model()
    if recognition is None:
        archive = os.path.join(config.MODELS_DIR, config.BUFFALO_ARCHIVE_NAME)
        if not os.path.exists(archive):
            download(config.BUFFALO_ARCHIVE_URL, archive, config.RECOGNITION_DOWNLOAD_LABEL, progress)
        recognition = extract_recognition(archive)
    detector = ascii_safe_path(detector, config.ASCII_DETECTOR_NAME)
    recognition = ascii_safe_path(recognition, config.ASCII_RECOGNITION_NAME)
    return detector, recognition
