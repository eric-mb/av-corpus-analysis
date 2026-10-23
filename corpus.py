"""Loading TIB-AV-A exports and running patterns on them.

Shared by the pages of the pattern app (pattern_app.py).
"""

import base64
import glob
import io
import os
import zipfile
from dataclasses import dataclass, field

import numpy as np
import streamlit as st
import yaml
from PIL import Image

DEFAULT_DATA_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "data", "exports")
PREVIEW_HEIGHT = 90

# Lower value = closer to the subject. Covers both the predicted "Shot Sizes" and
# the "GT: Distance" labels (see notebooks/stat_corpus_pattern_fix.py).
SHOT_SCALE = {
    "Extreme Close-Up": 0,
    "Close-Up": 1,
    "Medium Shot": 2,
    "Full Shot": 3,
    "Long Shot": 4,
    "XCU": 0,
    "BCU": 1,
    "CU": 2,
    "MCU": 3,
    "MS": 4,
    "MLS": 5,
    "LS": 6,
    "VLS": 7,
    "EST": 8,
}


#################
# Data model
#################


@dataclass
class Annotation:
    start: float
    end: float
    labels: list


@dataclass
class Face:
    """A face detected by TIB-AV-A, assigned to a face cluster ("Face 0", "Face 1", ...)."""

    time: float
    cluster: str
    x: float
    y: float
    w: float
    h: float

    @property
    def side(self):
        return "left" if self.x + self.w / 2 < 0.5 else "right"


@dataclass
class Shot:
    index: int
    start: float
    end: float
    # timeline name -> {label: fraction of the shot covered by that label}
    coverage: dict = field(default_factory=dict)
    # GT actors, or face clusters when only automatic data is used
    actors: list = field(default_factory=list)
    faces: list = field(default_factory=list)  # Face detections within the shot
    # scalar timeline name -> mean value within the shot, e.g. "Close-Up" -> 0.7
    scalars: dict = field(default_factory=dict)

    @property
    def duration(self):
        return self.end - self.start

    def get(self, timeline, min_overlap=0.5):
        """Labels of `timeline` covering at least `min_overlap` of this shot."""
        labels = self.coverage.get(timeline, {})
        return [l for l, f in sorted(labels.items(), key=lambda x: -x[1]) if f >= min_overlap]

    def dominant(self, timeline):
        """Label of `timeline` covering the largest part of this shot (or None)."""
        labels = self.coverage.get(timeline, {})
        return max(labels, key=labels.get) if labels else None

    def center(self, actor):
        """Mean horizontal face centre (0 = left, 1 = right) of a face cluster (predictions only)."""
        xs = [f.x + f.w / 2 for f in self.faces if f.cluster == actor]
        return sum(xs) / len(xs) if xs else None

    def side(self, actor):
        """Screen side ("left"/"right") of a face cluster in this shot (predictions only)."""
        center = self.center(actor)
        if center is None:
            return None
        return "left" if center < 0.5 else "right"

    @property
    def size(self):
        """Shot size: "GT: Distance" for ground truth, "Shot Sizes" for predictions."""
        label = self.dominant("GT: Distance") or self.dominant("Shot Sizes")
        return label.split("::")[-1] if label else None

    @property
    def angle(self):
        return self.dominant("GT: Angle")

    @property
    def emotion(self):
        label = self.dominant("Face Emotions")
        return label.split("::")[-1] if label else None


@dataclass
class Video:
    name: str
    duration: float
    fps: float
    timelines: dict  # timeline name -> list of Annotation
    thumbnails: list  # (time, base64 jpeg), sorted by time
    faces: list = field(default_factory=list)  # clustered Face detections, sorted by time
    # scalar timeline name -> (parent timeline name, times, values), e.g. the shot size
    # probabilities "Close-Up", ... below "Shot Sizes"
    scalars: dict = field(default_factory=dict)
    shots: list = field(default_factory=list)


@dataclass
class Match:
    start: float
    end: float
    label: str = ""
    first_shot: int = None
    last_shot: int = None


#################
# Loading
#################


def find_thumbnail_data_id(zip_file):
    plugin_runs = yaml.safe_load(zip_file.read("plugin_runs.yml")) or []
    results = yaml.safe_load(zip_file.read("plugin_run_results.yml")) or []
    thumbnail_runs = {x["id"] for x in plugin_runs if x["type"] == "thumbnail"}
    for result in results:
        if result["plugin_run_id"] in thumbnail_runs:
            return result["data_id"]
    return None


def load_thumbnails(inner):
    images = yaml.safe_load(inner.read("images_data.yml"))["images"]
    thumbnails = []
    for image in images:
        with Image.open(io.BytesIO(inner.read(f"{image['id']}.{image['ext']}"))) as img:
            img.draft("RGB", (PREVIEW_HEIGHT * 4, PREVIEW_HEIGHT * 2))
            img = img.convert("RGB")
            img.thumbnail((PREVIEW_HEIGHT * 4, PREVIEW_HEIGHT))
            buffer = io.BytesIO()
            img.save(buffer, format="JPEG", quality=70)
        thumbnails.append((image["time"], base64.b64encode(buffer.getvalue()).decode()))
    return sorted(thumbnails)


def load_faces(face_data):
    """Join faces, embeddings, bounding boxes and clusters (see notebooks/stat_single.ipynb)."""
    embedding_time = {}  # embedding id -> (face id, time)
    for x in face_data.get("ImageEmbeddings", []):
        embedding_time[x["id"]] = (x["ref_id"], x["time"])
    bboxes = {x["ref_id"]: x for x in face_data.get("BboxesData", [])}

    faces = []
    for i, cluster in enumerate(face_data.get("ClusterData", [])):
        for embedding_id in cluster["embedding_ids"]:
            if embedding_id not in embedding_time:
                continue
            face_id, time = embedding_time[embedding_id]
            bbox = bboxes.get(face_id, {})
            faces.append(
                Face(
                    time=time,
                    cluster=f"Face {i}",
                    x=bbox.get("x", 0.0),
                    y=bbox.get("y", 0.0),
                    w=bbox.get("w", 0.0),
                    h=bbox.get("h", 0.0),
                )
            )
    return sorted(faces, key=lambda f: f.time)


FACE_FILES = {
    "ImageEmbeddings": ("image_embeddings_data.yml", "embeddings"),
    "BboxesData": ("bboxes_data.yml", "bboxes"),
    "ClusterData": ("cluster_data.yml", "cluster"),
}


@st.cache_data(show_spinner=False)
def load_video(path, mtime):
    with zipfile.ZipFile(path) as zip_file:
        meta = yaml.safe_load(zip_file.read("video.yml"))
        timeline_list = yaml.safe_load(zip_file.read("timelines.yml"))
        timeline_names = {x["id"]: x["name"] for x in timeline_list}
        thumbnail_id = find_thumbnail_data_id(zip_file)
        result_data = {
            x["id"]: x["data_id"] for x in yaml.safe_load(zip_file.read("plugin_run_results.yml")) or []
        }
        # scalar data id -> (timeline name, parent timeline name)
        scalar_timelines = {
            result_data[x["plugin_run_result_id"]]: (x["name"], timeline_names.get(x["parent_id"]))
            for x in timeline_list
            if x.get("plugin_run_result_id") in result_data
        }

        timelines = {}
        thumbnails = []
        face_data = {}
        scalars = {}
        for name in zip_file.namelist():
            if not (name.startswith("data/") and name.endswith(".zip")):
                continue
            data_id = os.path.splitext(os.path.basename(name))[0]
            with zipfile.ZipFile(io.BytesIO(zip_file.read(name))) as inner:
                data_type = yaml.safe_load(inner.read("meta.yml"))["type"]
                if data_type == "AnnotationData" and data_id in timeline_names:
                    annotations = yaml.safe_load(inner.read("annotation_data.yml"))
                    timelines[timeline_names[data_id]] = [
                        Annotation(x["start"], x["end"], x["labels"])
                        for x in annotations.get("annotations") or []
                    ]
                elif data_type == "ImagesData" and data_id == thumbnail_id:
                    thumbnails = load_thumbnails(inner)
                elif data_type == "ScalarData" and data_id in scalar_timelines:
                    timeline, parent = scalar_timelines[data_id]
                    times = np.load(io.BytesIO(inner.read("time.npz")))
                    values = np.load(io.BytesIO(inner.read("y.npz")))
                    scalars[timeline] = (parent, times, values)
                elif data_type in FACE_FILES:
                    filename, key = FACE_FILES[data_type]
                    face_data[data_type] = yaml.safe_load(inner.read(filename)).get(key) or []

    return Video(
        name=(meta.get("name") or "").strip() or os.path.splitext(os.path.basename(path))[0],
        duration=meta["duration"],
        fps=meta["fps"],
        timelines=timelines,
        thumbnails=thumbnails,
        faces=load_faces(face_data),
        scalars=scalars,
    )


def build_shots(video, use_gt):
    shot_timeline = "GT: Shots" if use_gt else "Shots"
    shots = []
    for index, a in enumerate(sorted(video.timelines.get(shot_timeline, []), key=lambda x: x.start)):
        shot = Shot(index=index, start=a.start, end=a.end)
        length = max(shot.duration, 1e-6)
        for name, annotations in video.timelines.items():
            if name == shot_timeline:
                continue
            coverage = {}
            for x in annotations:
                overlap = min(x.end, shot.end) - max(x.start, shot.start)
                if overlap <= 0:
                    continue
                for label in x.labels:
                    coverage[label] = coverage.get(label, 0.0) + overlap / length
            if coverage:
                shot.coverage[name] = coverage
        shot.faces = [f for f in video.faces if shot.start < f.time < shot.end]
        for name, (_, times, values) in video.scalars.items():
            inside = (times >= shot.start) & (times < shot.end)
            if inside.any():
                shot.scalars[name] = float(values[inside].mean())
        if use_gt:
            shot.actors = sorted(
                label for name in shot.coverage if name.startswith("GT: Actor ") for label in shot.get(name)
            )
        else:
            shot.actors = sorted({f.cluster for f in shot.faces})
        shots.append(shot)
    return shots


def load_videos(data_dir):
    paths = sorted(glob.glob(os.path.join(data_dir, "*.zip")))
    videos = []
    progress = st.progress(0.0, text="Loading videos …")
    for i, path in enumerate(paths):
        progress.progress(i / len(paths), text=f"Loading {os.path.basename(path)} …")
        try:
            videos.append(load_video(path, os.path.getmtime(path)))
        except Exception as e:
            st.warning(f"Could not load {path}: {e}")
    progress.empty()
    return videos


#################
# Pattern helpers (available inside the pattern code)
#################


def alternating(values, min_length=3, link=None):
    """Runs where values alternate between two different non-empty values A, B, A, B, ...

    `link(i, j)` is an optional extra condition for every pair of consecutive
    indices i, j = i + 1 within a run.
    Returns a list of (first_index, last_index, (A, B)).
    """
    results = []
    n = len(values)
    i = 0
    while i < n - 1:
        a, b = values[i], values[i + 1]
        if a is not None and b is not None and a != b:
            j = i + 1
            while (
                j < n
                and values[j] == (a if (j - i) % 2 == 0 else b)
                and (link is None or link(j - 1, j))
            ):
                j += 1
            if j - i >= min_length:
                results.append((i, j - 1, (a, b)))
                # the last shot may open the next alternation (A B A | A C A ...)
                i = j - 1
                continue
        i += 1
    return results


def runs(values, min_length=2):
    """Runs of identical non-empty values. Returns a list of (first_index, last_index, value)."""
    results = []
    i = 0
    while i < len(values):
        j = i
        while j < len(values) and values[j] == values[i]:
            j += 1
        if values[i] is not None and j - i >= min_length:
            results.append((i, j - 1, values[i]))
        i = j
    return results


def monotonic(values, decreasing=True, min_length=3, min_step=0, plateau=None):
    """Runs of strictly decreasing (or increasing) non-empty values.

    Consecutive values have to differ by more than `min_step`. If `plateau` is
    set, consecutive values differing by at most `plateau` (e.g. 0 for equal
    labels) continue the run as well; they are trimmed at the start and end of
    a run, and `min_length` then counts the different levels instead of values
    (FS, MS, MS, CU has 3 levels).
    Returns a list of (first_index, last_index, [values]).
    """
    def follows(prev, cur):
        return prev - cur > min_step if decreasing else cur - prev > min_step

    def same(prev, cur):
        return plateau is not None and abs(cur - prev) <= plateau

    results = []
    i = 0
    while i < len(values):
        j = i + 1
        if values[i] is not None:
            while (
                j < len(values)
                and values[j] is not None
                and (follows(values[j - 1], values[j]) or same(values[j - 1], values[j]))
            ):
                j += 1
            first, last = i, j - 1
            while first < last and same(values[first], values[first + 1]):
                first += 1
            while last > first and same(values[last - 1], values[last]):
                last -= 1
            levels = 1 + sum(follows(values[k - 1], values[k]) for k in range(first + 1, last + 1))
            if levels >= min_length:
                results.append((first, last, values[first : last + 1]))
                i = j - 1 if j - 1 > i else j
                continue
        i += 1
    return results


def steady(values, max_step=0.05, min_length=3):
    """Runs of non-empty values where consecutive values differ by at most `max_step`.

    Returns a list of (first_index, last_index, [values]).
    """
    results = []
    i = 0
    while i < len(values):
        j = i + 1
        if values[i] is not None:
            while j < len(values) and values[j] is not None and abs(values[j] - values[j - 1]) <= max_step:
                j += 1
            if j - i >= min_length:
                results.append((i, j - 1, values[i:j]))
        i = j
    return results


def scale(label):
    """Rank of a shot size / distance label (0 = closest), or None."""
    return SHOT_SCALE.get(label)


PATTERN_GLOBALS = {
    "alternating": alternating,
    "runs": runs,
    "monotonic": monotonic,
    "steady": steady,
    "scale": scale,
    "SHOT_SCALE": SHOT_SCALE,
    "Shot": Shot,
    "Face": Face,
    "Video": Video,
    "Annotation": Annotation,
}

EXAMPLES = {
    "Shot-reverse-shot": '''\
# Consecutive shots showing exactly one character each,
# alternating between two characters: A, B, A, B, ...
def find(video):
    characters = [s.actors[0] if len(s.actors) == 1 else None for s in video.shots]
    return [
        (first, last, f"{a} ↔ {b}")
        for first, last, (a, b) in alternating(characters, min_length=3)
    ]
''',
    "Opposition": '''\
# Shot-reverse-shot where A is always on one side of the frame and B on the
# other side, and the face centres of consecutive shots are at least MIN_DELTA
# apart. Needs face positions, so it only finds matches with predictions.
MIN_DELTA = 0.1  # fraction of the frame width


def character(shot):
    if len(shot.actors) != 1 or shot.side(shot.actors[0]) is None:
        return None
    return (shot.actors[0], shot.side(shot.actors[0]))


def find(video):
    shots = video.shots

    def far_apart(i, j):
        return abs(shots[i].center(shots[i].actors[0]) - shots[j].center(shots[j].actors[0])) >= MIN_DELTA

    characters = [character(s) for s in shots]
    return [
        (first, last, f"{a} ({side_a}) ↔ {b} ({side_b})")
        for first, last, ((a, side_a), (b, side_b)) in alternating(characters, min_length=4, link=far_apart)
        if a != b and side_a != side_b
    ]
''',
    "Frameshare": '''\
# Shot-reverse-shot where A and B occupy the same position in the frame: the
# face centres of consecutive shots are at most MAX_DELTA apart.
# Needs face positions, so it only finds matches with predictions.
MAX_DELTA = 0.1  # fraction of the frame width


def character(shot):
    if len(shot.actors) != 1 or shot.center(shot.actors[0]) is None:
        return None
    return shot.actors[0]


def find(video):
    shots = video.shots

    def close(i, j):
        return abs(shots[i].center(shots[i].actors[0]) - shots[j].center(shots[j].actors[0])) <= MAX_DELTA

    characters = [character(s) for s in shots]
    return [
        (first, last, f"{a} ↔ {b}")
        for first, last, (a, b) in alternating(characters, min_length=2, link=close)
    ]
''',
    "Intensification": '''\
# The camera moves closer from shot to shot (e.g. LS -> MS -> MCU -> CU).
# Shots repeating the previous size continue the pattern (FS -> MS -> MS -> CU);
# min_length is the number of different sizes.
def find(video):
    sizes = [s.size for s in video.shots]
    ranks = [scale(x) for x in sizes]
    return [
        (first, last, " → ".join(sizes[first : last + 1]))
        for first, last, _ in monotonic(ranks, decreasing=True, min_length=3, plateau=0)
    ]
''',
    "Intensification (probabilities)": '''\
# The camera moves closer from shot to shot, measured with the predicted shot
# size probabilities instead of the labels (predictions only):
#   scale = 1 * p(Extreme Close-Up) + 2 * p(Close-Up) + 3 * p(Medium Shot)
#         + 4 * p(Full Shot) + 5 * p(Long Shot)
# with the probabilities averaged over each shot.
# Shots whose scale changes by at most PLATEAU count as the same size and
# continue the pattern; min_length is the number of different sizes.
WEIGHTS = {"Extreme Close-Up": 1, "Close-Up": 2, "Medium Shot": 3, "Full Shot": 4, "Long Shot": 5}
MIN_STEP = 0.5  # how much closer a shot has to be to count as the next size
PLATEAU = 0.05  # how much the scale may change between shots of the same size


def shot_scale(shot):
    if not all(name in shot.scalars for name in WEIGHTS):
        return None
    return sum(weight * shot.scalars[name] for name, weight in WEIGHTS.items())


def find(video):
    scales = [shot_scale(s) for s in video.shots]
    return [
        (first, last, " → ".join(f"{x:.2f}" for x in values))
        for first, last, values in monotonic(scales, decreasing=True, min_length=3, min_step=MIN_STEP, plateau=PLATEAU)
    ]
''',
    "Same size": '''\
# At least three consecutive shots with the same shot size.
def find(video):
    sizes = [s.size for s in video.shots]
    return [(first, last, size) for first, last, size in runs(sizes, min_length=3)]
''',
    "Same size (probabilities)": '''\
# Consecutive shots with (nearly) the same shot size, measured with the predicted
# shot size probabilities instead of the labels (predictions only):
#   scale = 1 * p(Extreme Close-Up) + 2 * p(Close-Up) + 3 * p(Medium Shot)
#         + 4 * p(Full Shot) + 5 * p(Long Shot)
# with the probabilities averaged over each shot.
WEIGHTS = {"Extreme Close-Up": 1, "Close-Up": 2, "Medium Shot": 3, "Full Shot": 4, "Long Shot": 5}
MAX_STEP = 0.05  # how much the scale may change from one shot to the next


def shot_scale(shot):
    if not all(name in shot.scalars for name in WEIGHTS):
        return None
    return sum(weight * shot.scalars[name] for name, weight in WEIGHTS.items())


def find(video):
    scales = [shot_scale(s) for s in video.shots]
    return [
        (first, last, " → ".join(f"{x:.2f}" for x in values))
        for first, last, values in steady(scales, max_step=MAX_STEP, min_length=3)
    ]
''',
}


def to_match(m, video):
    if isinstance(m, Match):
        return m
    if isinstance(m, dict):
        return Match(float(m["start"]), float(m["end"]), str(m.get("label", "")))
    first, last, *rest = m
    first, last = int(first), int(last)
    return Match(
        start=video.shots[first].start,
        end=video.shots[last].end,
        label=str(rest[0]) if rest else "",
        first_shot=first,
        last_shot=last,
    )


def run_pattern(code, video):
    namespace = dict(PATTERN_GLOBALS)
    exec(compile(code, "<pattern>", "exec"), namespace)
    find = namespace.get("find")
    if not callable(find):
        raise ValueError("The pattern code must define a function `find(video)`.")
    return [to_match(m, video) for m in find(video) or []]


#################
# Data sources
#################


GROUND_TRUTH = "Ground truth (GT: timelines)"
DATA_SOURCES = {
    GROUND_TRUTH: "Only the imported `GT:` timelines: shots, actors, scenes, distance "
    "(→ `shot.size`) and angle.",
    "Predictions (TIB-AV-A)": "Only automatic TIB-AV-A results: shot detection, face clustering "
    "(→ `shot.actors` = `Face 0`, `Face 1`, …), face emotions and `Shot Sizes` (→ `shot.size`).",
}
PREDICTION_TIMELINES = {"Shots", "Shot Sizes", "Face Emotions"}


def prepare_videos(videos, use_gt):
    """Keep only the timelines of the chosen data source and split the videos into shots."""
    for video in videos:
        if use_gt:
            video.timelines = {k: v for k, v in video.timelines.items() if k.startswith("GT:")}
            video.faces = []
            video.scalars = {}
        else:
            video.timelines = {k: v for k, v in video.timelines.items() if k in PREDICTION_TIMELINES}
            video.scalars = {k: v for k, v in video.scalars.items() if v[0] in PREDICTION_TIMELINES}
        video.shots = build_shots(video, use_gt)
    return videos
