"""Search editing patterns (e.g. shot / reverse shot) across TIB-AV-A exports.

Run from the repository root:

    uv run streamlit run pattern_app.py

Every *.zip TIB-AV-A export in the data folder (default: data/exports) is loaded once. A pattern is a
piece of Python code defining ``find(video)``, which is executed for every
video. ``find`` returns its matches either as shot ranges ``(first, last)`` /
``(first, last, label)`` (0-based shot indices, inclusive) or as time ranges
``{"start": s, "end": s, "label": ...}`` in seconds.
"""

import bisect
import html
import os
import traceback

import streamlit as st

import corpus_stats
from corpus import (
    DATA_SOURCES,
    DEFAULT_DATA_DIR,
    EXAMPLES,
    GROUND_TRUTH,
    PREVIEW_HEIGHT,
    load_videos,
    prepare_videos,
    run_pattern,
)

THUMB_HEIGHT = 60
THUMB_WIDTH = 107
MATCH_COLORS = ["#e8590c", "#1c7ed6", "#2f9e44", "#ae3ec9", "#f08c00", "#0c8599"]


#################
# Rendering
#################


def fmt_time(seconds):
    return f"{int(seconds // 60)}:{seconds % 60:04.1f}"


def tick_step(duration):
    for step in [1, 2, 5, 10, 15, 30, 60, 120, 300, 600]:
        if duration / step <= 12:
            return step
    return 1200


def keyframe(video, t):
    times = [x for x, _ in video.thumbnails]
    i = min(bisect.bisect_left(times, t), len(times) - 1)
    return video.thumbnails[i][1]


def match_shots(video, m):
    if m.first_shot is not None:
        return video.shots[m.first_shot : m.last_shot + 1]
    return [s for s in video.shots if s.end > m.start and s.start < m.end]


def shot_card(video, shot, color=None):
    caption = " · ".join(x for x in [f"#{shot.index}", ", ".join(shot.actors), shot.size] if x)
    border = f"border-top:3px solid {color};" if color else ""
    image = (
        f'<img src="data:image/jpeg;base64,{keyframe(video, (shot.start + shot.end) / 2)}" '
        f'style="height:{PREVIEW_HEIGHT}px;display:block">'
        if video.thumbnails
        else ""
    )
    return (
        f'<div style="{border}flex:0 0 auto">{image}'
        f'<div style="font-size:11px;padding:2px 0;max-width:{PREVIEW_HEIGHT * 16 // 9}px;'
        f'overflow:hidden;text-overflow:ellipsis;white-space:nowrap">{html.escape(caption)}</div></div>'
    )


def shot_strip(video, shots, color=None):
    cards = "".join(shot_card(video, s, color) for s in shots)
    return f'<div style="display:flex;flex-wrap:wrap;gap:4px">{cards}</div>'


POPUP_STYLE = (
    "display:none;position:absolute;top:100%;left:0;z-index:1000;max-width:100%;"
    "padding:6px;border-radius:6px;background:rgba(25,25,30,0.96);color:#fff;"
    "box-shadow:0 4px 16px rgba(0,0,0,0.35);"
)


def render_timeline(video, matches, px_per_second):
    duration = max(video.duration, 1e-6)

    def pct(t):
        return f"{100 * min(max(t, 0), duration) / duration:.4f}%"

    if px_per_second:
        width = f"{duration * px_per_second:.0f}px"
        n_thumbs = max(1, round(duration * px_per_second / THUMB_WIDTH))
    else:
        width = "100%"
        n_thumbs = 12

    thumbs = []
    for k in range(n_thumbs if video.thumbnails else 0):
        t = (k + 0.5) * duration / n_thumbs
        thumbs.append(
            f'<img src="data:image/jpeg;base64,{keyframe(video, t)}" title="{fmt_time(t)}" '
            f'style="flex:1 1 0;min-width:0;height:{THUMB_HEIGHT}px;object-fit:cover;display:block">'
        )

    def shot_popup_left(shot):
        # Only in fit mode do timeline positions match the (non-scrolling) popup area.
        if px_per_second:
            return ""
        return f"left:clamp(0px, {pct(shot.start)}, calc(100% - 260px));"

    # Hovering a shot or a match shows a popup below the timeline. The popups live
    # outside the horizontally scrolling area so they are not clipped; :has() links
    # the hovered element to its popup.
    rules, popups = [], []

    shots = []
    for s in video.shots:
        details = [f"Shot {s.index}: {fmt_time(s.start)} – {fmt_time(s.end)}"] + [
            f"{k}: {v}"
            for k, v in [("Size", s.size), ("Angle", s.angle), ("Emotion", s.emotion)]
            if v
        ]
        shade = "0.22" if s.index % 2 == 0 else "0.38"
        shots.append(
            f'<div class="pt-s{s.index}" style="position:absolute;top:0;bottom:0;'
            f"left:{pct(s.start)};width:calc({pct(s.end)} - {pct(s.start)});"
            f"background:rgba(128,128,128,{shade});border-right:1px solid rgba(128,128,128,0.7);"
            f'overflow:hidden;font-size:10px;line-height:20px;padding-left:2px;box-sizing:border-box">'
            f"{s.index}</div>"
        )
        rules.append(f".pt-wrap:has(.pt-s{s.index}:hover) .pt-ps{s.index}{{display:block !important}}")
        popups.append(
            f'<div class="pt-ps{s.index}" style="{POPUP_STYLE}{shot_popup_left(s)}">{shot_card(video, s)}'
            f'<div style="font-size:11px;opacity:0.8">{html.escape(" · ".join(details[1:]))}</div></div>'
        )

    colors = {}
    overlays, lines = [], []
    for k, m in enumerate(matches):
        color = colors.setdefault(m.label, MATCH_COLORS[len(colors) % len(MATCH_COLORS)])
        shot_range = f" · shots {m.first_shot}–{m.last_shot}" if m.first_shot is not None else ""
        geometry = f"left:{pct(m.start)};width:max(2px, calc({pct(m.end)} - {pct(m.start)}))"
        overlays.append(
            f'<div style="position:absolute;top:0;bottom:0;{geometry};background:{color}22;'
            f'border-left:2px solid {color};border-right:2px solid {color};box-sizing:border-box;pointer-events:none"></div>'
        )
        lines.append(
            f'<div class="pt-m{k}" style="position:absolute;top:0;height:12px;{geometry};cursor:pointer">'
            f'<div style="margin-top:2px;height:8px;background:{color};border-radius:4px"></div></div>'
        )
        rules.append(f".pt-wrap:has(.pt-m{k}:hover) .pt-pm{k}{{display:block !important}}")
        popups.append(
            f'<div class="pt-pm{k}" style="{POPUP_STYLE}">'
            f'<div style="font-size:12px;margin-bottom:4px"><b style="color:{color}">{html.escape(m.label)}</b>'
            f"{shot_range} · {fmt_time(m.start)} – {fmt_time(m.end)}</div>"
            f"{shot_strip(video, match_shots(video, m), color)}</div>"
        )

    step = tick_step(duration)
    ticks = "".join(
        f'<div style="position:absolute;left:{pct(t)};top:0;font-size:10px;opacity:0.7;'
        f'border-left:1px solid rgba(128,128,128,0.6);padding-left:2px;height:14px">{fmt_time(t)[:-2]}</div>'
        for t in range(0, int(duration) + 1, step)
    )

    return f"""
<style>{''.join(rules)}</style>
<div class="pt-wrap" style="position:relative">
<div style="overflow-x:auto;padding-bottom:4px">
  <div style="position:relative;width:{width};min-width:100%">
    <div style="display:flex;height:{THUMB_HEIGHT}px;overflow:hidden;border-radius:4px 4px 0 0">{''.join(thumbs)}</div>
    <div style="position:relative;height:20px;margin-top:2px">{''.join(shots)}</div>
    <div style="position:relative;height:12px;margin-top:2px">{''.join(lines)}</div>
    <div style="position:relative;height:16px;margin-top:2px">{ticks}</div>
    <div style="position:absolute;top:0;left:0;right:0;height:{THUMB_HEIGHT + 22}px;pointer-events:none">{''.join(overlays)}</div>
  </div>
</div>
{''.join(popups)}
</div>
"""


def render_match_strips(video, matches):
    colors = {}
    rows = []
    for m in matches:
        color = colors.setdefault(m.label, MATCH_COLORS[len(colors) % len(MATCH_COLORS)])
        shots = match_shots(video, m)
        if m.first_shot is not None:
            shot_range = f"shots {m.first_shot}–{m.last_shot} ({len(shots)} shots)"
        else:
            shot_range = f"{len(shots)} shots"
        rows.append(
            f'<div style="margin-bottom:12px"><div style="font-size:13px;margin-bottom:4px">'
            f'<b style="color:{color}">{html.escape(m.label)}</b> · {shot_range} · '
            f"{fmt_time(m.start)} – {fmt_time(m.end)} ({m.end - m.start:.1f} s)</div>"
            f"{shot_strip(video, shots, color)}</div>"
        )
    return "".join(rows)


#################
# App
#################


def load_example():
    st.session_state.code = EXAMPLES[st.session_state.example]


def search_page(videos):
    with st.sidebar:
        st.header("Display")
        zoom = st.select_slider(
            "Zoom (px per second)", options=["Fit", 2, 5, 10, 20, 40], value="Fit"
        )
        show_empty = st.checkbox("Show videos without matches", value=False)

    st.title("Pattern search")

    # The text area's state is dropped while another page is shown, so it is
    # restored from the last searched pattern.
    st.session_state.setdefault("active_code", EXAMPLES["Shot-reverse-shot"])
    if "code" not in st.session_state:
        st.session_state.code = st.session_state.active_code

    st.selectbox("Load example", list(EXAMPLES), key="example", on_change=load_example)
    with st.form("pattern"):
        st.text_area(
            "Pattern code",
            key="code",
            height=260,
            help="Define `find(video)`. Available: video.shots (shot.index/start/end/duration/"
            "actors/size/angle/emotion/faces/scalars/center(actor)/side(actor)/get(timeline)/"
            "dominant(timeline)), video.faces "
            "(face.time/cluster/x/y/w/h/side), video.timelines, alternating(), runs(), "
            "monotonic(), steady(), scale(). angle is ground truth only; emotion, faces and scalars "
            "(shot size / emotion probabilities) are predictions only.",
        )
        if st.form_submit_button("Search", type="primary"):
            st.session_state.active_code = st.session_state.code

    code = st.session_state.active_code
    try:
        compile(code, "<pattern>", "exec")
    except SyntaxError:
        st.error("Syntax error in pattern code")
        st.code(traceback.format_exc(limit=0))
        return

    results = []
    for video in videos:
        try:
            results.append((video, run_pattern(code, video), None))
        except Exception:
            results.append((video, [], traceback.format_exc()))

    total = sum(len(m) for _, m, _ in results)
    with_matches = sum(1 for _, m, _ in results if m)
    c1, c2, c3 = st.columns(3)
    c1.metric("Matches", total)
    c2.metric("Videos with matches", f"{with_matches} / {len(videos)}")
    c3.metric("Shots searched", sum(len(v.shots) for v in videos))

    px_per_second = None if zoom == "Fit" else zoom
    for video, matches, error in results:
        if not matches and not error and not show_empty:
            continue
        st.subheader(video.name)
        st.caption(
            f"{len(matches)} matches · {len(video.shots)} shots · {fmt_time(video.duration)}"
        )
        if error:
            st.error("Pattern failed for this video")
            st.code(error)
            continue
        st.html(render_timeline(video, matches, px_per_second))
        if matches:
            with st.expander(f"Matches ({len(matches)})"):
                st.html(render_match_strips(video, matches))


def main():
    st.set_page_config(page_title="Pattern search", layout="wide")

    with st.sidebar:
        st.header("Data")
        data_dir = st.text_input("Data folder (TIB-AV-A *.zip exports)", value=os.path.normpath(DEFAULT_DATA_DIR))
        source = st.selectbox("Data source", list(DATA_SOURCES), index=1)
        st.caption(DATA_SOURCES[source])

    videos = load_videos(data_dir)
    if not videos:
        st.info(f"No TIB-AV-A exports (*.zip) found in `{os.path.abspath(data_dir)}`.")
        return
    prepare_videos(videos, use_gt=source == GROUND_TRUTH)

    pages = [
        st.Page(lambda: search_page(videos), title="Pattern search", icon=":material/search:", url_path="search", default=True),
        st.Page(
            lambda: corpus_stats.page(videos, st.session_state.get("active_code")),
            title="Corpus statistics",
            icon=":material/bar_chart:",
            url_path="statistics",
        ),
    ]
    st.navigation(pages).run()


if __name__ == "__main__":
    main()
