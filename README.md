# AV Corpus Analysis

Search for editing patterns, such as shot / reverse shot, intensification or runs of the same shot size, across a corpus of films annotated with [TIB AV-Analytics (TIB-AV-A)](https://service.tib.eu/tibava).

The core of this repository is the **pattern app** ([pattern_app.py](pattern_app.py)), a Streamlit app that loads every TIB-AV-A export in a folder. It has two pages:

- **Pattern search** runs a user-defined pattern on every video and shows the matches on a timeline with keyframes for each video.
- **Corpus statistics** compares the videos: the distribution of shot sizes and how often each pattern occurs.

## Repository layout

```
.
├── pattern_app.py      # the app (Streamlit): entry point and pattern search page
├── corpus_stats.py     # corpus statistics page
├── corpus.py           # loading TIB-AV-A exports, shots, pattern helpers and examples
├── pyproject.toml      # dependencies of the app (uv)
├── uv.lock
├── data/
│   └── exports/        # TIB-AV-A exports (*.zip) of the corpus, loaded by the app
└── notebooks/          # Jupyter notebooks and scripts for corpus statistics (separate uv project)
```

## Running the app

Requires [uv](https://docs.astral.sh/uv/).

```sh
uv run streamlit run pattern_app.py
```

The app opens in the browser. By default, it reads all `*.zip` files in `data/exports/`. You can choose another folder in the sidebar. The data folder and the data source apply to both pages.

### Data source

In the sidebar, you choose which annotations a pattern sees:

- **Predictions (TIB-AV-A)**: only the automatic results: shot detection, face clustering (`shot.actors` are `Face 0`, `Face 1`, …), face positions, face emotions, and shot sizes (labels and probabilities).
- **Ground truth**: only the imported `GT:` timelines: shots, actors, scenes, distance (`shot.size`) and angle.

### Writing a pattern

A pattern is Python code that defines `find(video)`. The app runs it for every video. `find` returns a list of matches, either as

- shot ranges `(first, last)` or `(first, last, label)`, where the shot indices are 0-based and inclusive, or
- time ranges `{"start": seconds, "end": seconds, "label": ...}`.

```python
# Consecutive shots showing exactly one character each,
# alternating between two characters: A, B, A, B, ...
def find(video):
    characters = [s.actors[0] if len(s.actors) == 1 else None for s in video.shots]
    return [
        (first, last, f"{a} ↔ {b}")
        for first, last, (a, b) in alternating(characters, min_length=3)
    ]
```

The pattern code can use:

| Name | Description |
| --- | --- |
| `video.shots` | `Shot` objects with `index`, `start`, `end`, `duration`, `actors`, `size`, `angle` (GT only), `emotion`, `faces`, `scalars` (predictions only), `center(actor)`, `side(actor)`, `get(timeline)`, `dominant(timeline)` |
| `video.faces`, `video.timelines` | the raw face detections and annotation timelines |
| `alternating(values, min_length, link)` | runs alternating between two values A, B, A, B, … |
| `runs(values, min_length)` | runs of identical values |
| `monotonic(values, decreasing, min_length, min_step, plateau)` | strictly decreasing or increasing runs |
| `steady(values, max_step, min_length)` | runs whose values change by at most `max_step` from one value to the next |
| `scale(label)`, `SHOT_SCALE` | rank of a shot size label (0 = closest) |

The app comes with example patterns that you can open from the **Load example** menu: shot-reverse-shot, opposition, frameshare, intensification and same size. Some of them have variants that use the shot size probabilities. The examples are defined in [corpus.py](corpus.py).

### Corpus statistics

The **Corpus statistics** page shows the number of videos and shots, the total runtime and the average shot length of the corpus, and per video:

- **Shot sizes**: a stacked bar per video, ordered from the closest to the farthest shot size. Switch between the share of shots, the share of runtime and the number of shots.
- **Patterns**: one bar chart per pattern with the matches per video. It covers all example patterns, plus the last searched pattern when it isn't one of the examples. Switch between the number of matches, matches per 100 shots and the share of runtime covered by matches.

Each chart has a table with the numbers below it.

## Data

`data/exports/` contains the TIB-AV-A exports of the corpus (17 exports, some films are split into two parts). The exports include the automatic analyses and the imported ground truth (`GT:`) timelines. For some films, these are corrected versions of the original exports (`*_fix.zip`).

To add a film, export the video from TIB-AV-A and put the `.zip` file in `data/exports/`.

## Notebooks

`notebooks/` contains the Jupyter notebooks and scripts for the statistical film analysis:

- `stat_single.ipynb`: analysis of a single TIB-AV-A export
- `stat_corpus.ipynb`, `stat_corpus_fix.ipynb`, `stat_corpus_pattern_fix.ipynb`: corpus statistics, e.g. the correlation between TIB-AV-A predictions and the ground truth (on the original and the corrected exports)
- `stat_corpus_pattern_fix.py`, `est_start_end.py`, `parse.py`: helper scripts for the corpus statistics, start / end alignment and parsing ELAN annotations

The folder is a separate uv project:

```sh
cd notebooks
uv run jupyter lab
```

## Background

This repository started from the materials of the DHd 2026 workshop *Film- und Videoanalyse mit VIAN & TIB-AV-A – Grundlagen, Anwendungen und Schnittstellen*.
