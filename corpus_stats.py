"""Corpus statistics page of the pattern app: shot sizes and pattern frequencies per video."""

import altair as alt
import pandas as pd
import streamlit as st

from corpus import EXAMPLES, SHOT_SCALE, run_pattern

# Ordinal blue ramp (light -> dark); closer shot sizes get darker steps.
SIZE_RAMP = ["#9ec5f4", "#86b6ef", "#6da7ec", "#5598e7", "#3987e5", "#2a78d6", "#256abf", "#1c5cab", "#184f95"]
NO_SIZE = "No size"
THEMES = {
    "light": {"surface": "#ffffff", "bar": "#2a78d6", "no_size": "#c3c2b7"},
    "dark": {"surface": "#0e1117", "bar": "#3987e5", "no_size": "#52514e"},
}
ROW_HEIGHT = 22
CURRENT_PATTERN = "Current search pattern"


def theme():
    return THEMES.get(st.context.theme.type, THEMES["light"])


def fmt_duration(seconds):
    return f"{int(seconds // 3600)}:{int(seconds % 3600 // 60):02d}:{int(seconds % 60):02d}"


def covered(intervals):
    """Total length of the union of (start, end) intervals."""
    total, end = 0.0, float("-inf")
    for s, e in sorted(intervals):
        if e > end:
            total += e - max(s, end)
            end = e
    return total


#################
# Shot sizes
#################


def size_order(sizes):
    """Shot sizes from closest to farthest; unknown labels after, "No size" last."""
    known = sorted((s for s in sizes if s in SHOT_SCALE), key=SHOT_SCALE.get)
    unknown = sorted(s for s in sizes if s not in SHOT_SCALE and s != NO_SIZE)
    return known + unknown + ([NO_SIZE] if NO_SIZE in sizes else [])


def size_colors(order):
    sized = [s for s in order if s != NO_SIZE]
    n = len(sized)
    # Evenly spaced steps of the ramp, the closest size on the darkest step.
    steps = [SIZE_RAMP[round(i * (len(SIZE_RAMP) - 1) / max(n - 1, 1))] for i in range(n)][::-1]
    return steps + ([theme()["no_size"]] if NO_SIZE in order else [])


def size_table(videos):
    rows = []
    for video in videos:
        n_shots = len(video.shots)
        runtime = sum(s.duration for s in video.shots)
        counts = {}
        for shot in video.shots:
            size = shot.size or NO_SIZE
            count, duration = counts.get(size, (0, 0.0))
            counts[size] = (count + 1, duration + shot.duration)
        for size, (count, duration) in counts.items():
            rows.append(
                {
                    "Video": video.name,
                    "Shot size": size,
                    "Number of shots": count,
                    "Share of shots (%)": 100 * count / n_shots,
                    "Share of runtime (%)": 100 * duration / runtime if runtime else 0.0,
                }
            )
    return pd.DataFrame(rows)


def size_chart(df, measure, video_order):
    order = size_order(set(df["Shot size"]))
    df = df.assign(order=df["Shot size"].map(order.index))
    return (
        alt.Chart(df)
        .mark_bar(stroke=theme()["surface"], strokeWidth=2)
        .encode(
            y=alt.Y("Video:N", sort=video_order, title=None),
            x=alt.X(
                f"{measure}:Q",
                stack="zero",
                title=measure,
                scale=alt.Scale(domain=[0, 100]) if measure.startswith("Share") else alt.Undefined,
            ),
            color=alt.Color(
                "Shot size:N",
                scale=alt.Scale(domain=order, range=size_colors(order)),
                legend=alt.Legend(orient="top", title=None, direction="horizontal", columns=len(order)),
            ),
            order=alt.Order("order:Q"),
            tooltip=[
                "Video",
                "Shot size",
                "Number of shots",
                alt.Tooltip("Share of shots (%):Q", format=".1f"),
                alt.Tooltip("Share of runtime (%):Q", format=".1f"),
            ],
        )
        .properties(height=ROW_HEIGHT * len(video_order))
    )


#################
# Patterns
#################


def pattern_table(videos, patterns):
    rows, errors = [], []
    for name, code in patterns.items():
        for video in videos:
            try:
                matches = run_pattern(code, video)
            except Exception as e:
                errors.append(f"{name} · {video.name}: {type(e).__name__}: {e}")
                continue
            rows.append(
                {
                    "Pattern": name,
                    "Video": video.name,
                    "Matches": len(matches),
                    "Matches per 100 shots": 100 * len(matches) / len(video.shots) if video.shots else 0.0,
                    "Share of runtime (%)": 100 * covered((m.start, m.end) for m in matches) / video.duration
                    if video.duration
                    else 0.0,
                }
            )
    return pd.DataFrame(rows), errors


def pattern_chart(df, measure, video_order, pattern_order):
    return (
        alt.Chart(df)
        .mark_bar(color=theme()["bar"], cornerRadiusEnd=4)
        .encode(
            y=alt.Y("Video:N", sort=video_order, title=None),
            x=alt.X(f"{measure}:Q", title=None),
            tooltip=[
                "Pattern",
                "Video",
                "Matches",
                alt.Tooltip("Matches per 100 shots:Q", format=".1f"),
                alt.Tooltip("Share of runtime (%):Q", format=".1f"),
            ],
        )
        .properties(width=200, height=ROW_HEIGHT * len(video_order))
        .facet(facet=alt.Facet("Pattern:N", sort=pattern_order, title=None), columns=3)
        .resolve_axis(y="independent")
    )


#################
# Page
#################


def page(videos, active_code=None):
    st.title("Corpus statistics")

    shots = [s for v in videos for s in v.shots]
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Videos", len(videos))
    c2.metric("Shots", len(shots))
    c3.metric("Runtime", fmt_duration(sum(v.duration for v in videos)))
    c4.metric("Average shot length", f"{sum(s.duration for s in shots) / len(shots):.1f} s" if shots else "–")

    video_order = sorted(v.name for v in videos)

    st.header("Shot sizes")
    sizes = size_table(videos)
    if sizes.empty:
        st.info("No shots in the selected data source.")
    else:
        measure = st.segmented_control(
            "Measure",
            ["Share of shots (%)", "Share of runtime (%)", "Number of shots"],
            default="Share of shots (%)",
            key="size_measure",
        ) or "Share of shots (%)"
        st.altair_chart(size_chart(sizes, measure, video_order), width="stretch")
        with st.expander("Table"):
            st.dataframe(
                sizes.pivot_table(index="Video", columns="Shot size", values=measure, fill_value=0)[
                    size_order(set(sizes["Shot size"]))
                ],
                column_config={c: st.column_config.NumberColumn(format="%.1f") for c in sizes["Shot size"].unique()}
                if measure != "Number of shots"
                else None,
            )

    st.header("Patterns")
    patterns = dict(EXAMPLES)
    if active_code and active_code not in EXAMPLES.values():
        patterns = {CURRENT_PATTERN: active_code, **patterns}
    selected = st.multiselect("Patterns", list(patterns), default=list(patterns), key="stat_patterns")
    measure = st.segmented_control(
        "Measure",
        ["Matches", "Matches per 100 shots", "Share of runtime (%)"],
        default="Matches",
        key="pattern_measure",
    ) or "Matches"
    if not selected:
        return
    with st.spinner("Running patterns …"):
        df, errors = pattern_table(videos, {name: patterns[name] for name in selected})
    if errors:
        with st.expander(f"Pattern errors ({len(errors)})"):
            st.code("\n".join(errors))
    if df.empty:
        return
    st.caption("One chart per pattern. Patterns that need face positions or shot size probabilities only find matches with predictions.")
    st.altair_chart(pattern_chart(df, measure, video_order, selected))
    with st.expander("Table"):
        table = df.pivot_table(index="Video", columns="Pattern", values=measure, fill_value=0)
        table = table[[name for name in selected if name in table.columns]]
        st.dataframe(
            table,
            column_config={c: st.column_config.NumberColumn(format="%.1f") for c in selected}
            if measure != "Matches"
            else None,
        )
