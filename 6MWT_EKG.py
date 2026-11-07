#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Streamlit app for processing a Delsys 6MWT EKG using a recorded Analog 1
trigger or a manually recorded clock time.

The app automatically creates the 6MWT END trigger exactly 6 minutes (360 s) later.

@author: beier

Oct 2026 - v2
"""

import csv
import hashlib
from itertools import islice
from urllib.parse import quote
# from pathlib import Path
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

TEST_DURATION_SECONDS = 360
HR_EPOCH_SECONDS = 15
DEFAULT_ANALOG_THRESHOLD_V = 2.5

# #change save folder
# SAVE_FOLDER = Path("/Users/beier/Library/CloudStorage/OneDrive-UBC/Courtney's team - Mobility and Balance Rehab Lab - STUDY 3 - STROKE training/Training DATA/training - EKG raw (trigger fixed)/overground")
# SAVE_FOLDER.mkdir(parents=True,exist_ok=True)

st.set_page_config(page_title="6MWT EKG Trigger Reconstruction",layout="wide")
st.title("6MWT EKG Trigger Reconstruction")

st.write(
    """
    Upload a raw Delsys overground EKG CSV file. Choose a recorded **Analog 1**
    trigger or enter the manually recorded **6MWT start time**.
    The app places the end marker **6 minutes after the selected start**.
    """)

#SESSION STATE

if "export_data" not in st.session_state:
    st.session_state.export_data = None
if "output_filename" not in st.session_state:
    st.session_state.output_filename = None
if "processed_source" not in st.session_state:
    st.session_state.processed_source = None

#FUNCTIONS

def get_csv_rows(uploaded_file,nrows=7):
    uploaded_file.seek(0)
    lines = (line.decode("utf-8-sig",errors="replace") for line in uploaded_file)
    return list(islice(csv.reader(lines),nrows))

def extract_recording_metadata(uploaded_file):
    sampling_row = 6
    data_start = 7
    rows = get_csv_rows(uploaded_file,nrows=data_start)

    if len(rows) < data_start:
        raise ValueError("The uploaded file does not contain the expected 7-row Delsys header.")

    if len(rows[1]) < 2:
        raise ValueError("Could not locate recording Date/Time.")

    recording_start_string = str(rows[1][1]).strip()

    try:
        recording_start = pd.to_datetime(recording_start_string)
    except Exception as e:
        raise ValueError(f"Could not parse recording Date/Time: {recording_start_string}") from e

    collection_length = np.nan
    if len(rows[2]) >= 2:
        try:
            collection_length = float(str(rows[2][1]).strip())
        except Exception:
            collection_length = np.nan

    channel_labels = [label.strip().casefold() for label in rows[5]]
    channel_pairs = list(zip(channel_labels,channel_labels[1:]))
    ekg_columns = next(
        ((i,i+1) for i,(time_label,signal_label) in enumerate(channel_pairs)
         if time_label.startswith("ekg") and "time series" in time_label
         and signal_label.startswith("ekg") and "time series" not in signal_label),
        None,
    )
    analog_columns = next(
        ((i,i+1) for i,(time_label,signal_label) in enumerate(channel_pairs)
         if time_label == "analog 1 time series (s)" and signal_label == "analog 1 (v)"),
        None,
    )
    if ekg_columns is None:
        if analog_columns is not None or any(label.startswith("analog ") for label in channel_labels[:2]):
            raise ValueError("Could not locate the EKG time and signal columns in the Delsys header.")
        # Legacy EKG-only exports use the first two columns.
        ekg_columns = (0,1)

    ekg_signal_column = ekg_columns[1]
    if len(rows[sampling_row]) <= ekg_signal_column:
        raise ValueError("Could not locate the EKG sampling frequency.")
    sampling_text = rows[sampling_row][ekg_signal_column].strip().lower().removesuffix("hz").strip()

    try:
        ekg_fs = float(sampling_text)
    except Exception as e:
        raise ValueError(f"Could not parse EKG sampling frequency from '{sampling_text}'.") from e

    return {
        "recording_start":recording_start,
        "collection_length":collection_length,
        "ekg_fs":ekg_fs,
        "ekg_columns":ekg_columns,
        "analog_columns":analog_columns,
        "data_start":data_start
    }

def load_channel(uploaded_file,data_start,columns):
    uploaded_file.seek(0)

    channel = pd.read_csv(
        uploaded_file,
        skiprows=data_start,
        header=None,
        usecols=list(columns),
        skipinitialspace=True,
        low_memory=False,
    )
    channel = channel.apply(pd.to_numeric,errors="coerce")
    samples = channel.to_numpy(dtype=float)
    return samples[np.isfinite(samples).all(axis=1)].T

def load_ekg(uploaded_file,metadata_info):
    times,signal = load_channel(uploaded_file,metadata_info["data_start"],metadata_info["ekg_columns"])
    if times.size == 0:
        raise ValueError("No valid numeric EKG data found.")
    return times,signal

def find_analog_trigger_times(uploaded_file,metadata_info,threshold_v):
    times,voltage = load_channel(
        uploaded_file,metadata_info["data_start"],metadata_info["analog_columns"])
    if times.size < 2:
        raise ValueError("Analog 1 has too few numeric samples to detect a falling edge.")
    if np.any(np.diff(times) <= 0):
        raise ValueError("Analog 1 timestamps must be strictly increasing.")

    # The sample file uses an active-low pulse. A recording that starts low
    # contains no observed falling edge until the voltage goes high and drops again.
    low = voltage <= threshold_v
    falling_edges = np.flatnonzero(~low[:-1] & low[1:])+1
    return times[falling_edges]

def parse_manual_clock_time(clock_time_string,recording_start):
    clock_time_string = str(clock_time_string).strip()

    if clock_time_string == "":
        raise ValueError("6MWT start time is blank.")

    recording_date_string = recording_start.strftime("%Y-%m-%d")

    try:
        event_datetime = pd.to_datetime(f"{recording_date_string} {clock_time_string}")
    except Exception as e:
        raise ValueError(f"Could not interpret '{clock_time_string}' as a clock time.") from e

    if event_datetime < recording_start:
        event_datetime += pd.Timedelta(days=1)

    seconds_from_start = (event_datetime-recording_start).total_seconds()
    return event_datetime,seconds_from_start

def make_ekg_plot(ekg_data,time_vector,trigger_times=(),trigger_labels=None,title="EKG Signal"):
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=time_vector,y=ekg_data,mode="lines",name="EKG",line=dict(width=1)))

    for i,trigger_time in enumerate(trigger_times):
        fig.add_vline(x=trigger_time,line_width=2,line_dash="dash")

        if trigger_labels is not None:
            fig.add_annotation(x=trigger_time,y=1,yref="paper",text=trigger_labels[i],showarrow=False,textangle=-90,xanchor="left",yanchor="top")

    fig.update_layout(title=title,xaxis_title="Time (s)",yaxis_title="EKG Signal (mV)",height=600,hovermode="x unified")
    return fig

def detect_r_peaks(signal,fs,invert_signal):
    import neurokit2 as nk

    clean = nk.ecg_clean(signal, sampling_rate=fs, method="neurokit")
    if invert_signal:
        clean = -clean
    _, info = nk.ecg_peaks(clean, sampling_rate=fs, method="neurokit")
    detected = info["ECG_R_Peaks"]
    if len(detected) >= 3:
        _, detected = nk.signal_fixpeaks(detected, sampling_rate=fs, method="Kubios")
    detected = np.unique(np.asarray(detected, dtype=int))
    detected = detected[(detected >= 0) & (detected < len(clean))]

    # Match the reference dashboard's positive amplitude and spacing filters.
    candidates = detected[clean[detected] > 0]
    if candidates.size:
        candidates = candidates[clean[candidates] >= 0.20 * np.median(clean[candidates])]

    kept = []
    min_spacing = int(0.30 * fs)
    for sample in candidates:
        if not kept or sample - kept[-1] >= min_spacing:
            kept.append(sample)
        elif clean[sample] > clean[kept[-1]]:
            kept[-1] = sample
    return clean, detected, np.array(kept, dtype=int)


def summarize_walk_hr(time, peaks):
    epoch_starts = np.arange(0, TEST_DURATION_SECONDS, HR_EPOCH_SECONDS)
    peaks = peaks[(time[peaks] >= 0) & (time[peaks] < TEST_DURATION_SECONDS)]
    peak_times = time[peaks]
    rr = np.diff(peak_times)
    bpm = np.divide(60, rr, out=np.full(rr.shape, np.nan), where=rr > 0)
    valid = (bpm >= 40) & (bpm <= 180)
    beat_hr = pd.DataFrame({
        "walk_sample_idx": peaks[1:][valid],
        "time_s": peak_times[1:][valid],
        "rr_s": rr[valid],
        "hr_bpm": bpm[valid],
    })
    beat_hr["hr_smooth_bpm"] = beat_hr["hr_bpm"].rolling(5, center=True, min_periods=1).mean()
    epochs = pd.DataFrame({
        "epoch": np.arange(1, len(epoch_starts) + 1),
        "start_s": epoch_starts,
        "end_s": epoch_starts + HR_EPOCH_SECONDS,
        "r_peak_count": np.histogram(peak_times, bins=np.r_[epoch_starts, TEST_DURATION_SECONDS])[0],
    })
    # Assign each valid interval to the epoch containing its later R-peak.
    grouped = beat_hr.groupby(beat_hr["time_s"] // HR_EPOCH_SECONDS)["hr_bpm"]
    epochs["valid_rr_count"] = grouped.count().reindex(epochs.index, fill_value=0)
    epochs["mean_hr_bpm"] = grouped.mean().reindex(epochs.index)
    return beat_hr, epochs


def peak_near_time(time, clean, target, lo, hi):
    first = max(lo, min(hi - 1, int(time.searchsorted(target - 0.03))))
    last = max(first + 1, min(hi, int(time.searchsorted(target + 0.03, side="right"))))
    return first + int(np.argmax(clean[first:last]))


def show_walk_hr(walk, fs, source, output_filename):
    st.divider()
    st.subheader("6MWT R-peak review and heart rate")
    if walk is None:
        st.info("Process the walk above to prepare the trimmed EKG for R-peak detection.")
        return

    time = walk["time"]
    st.write("Detect R-peaks in the trimmed EKG, then review the peaks by 15-second epoch.")
    st.caption(f"Analysis input: {len(time):,} trimmed EKG samples spanning {time[-1]:.3f} s from the selected start.")
    invert_signal = st.checkbox("Invert EKG polarity for R-peak detection")
    analysis_key = (source, invert_signal)
    if st.button("Detect R-peaks"):
        st.session_state.pop("walk_hr_review", None)
        if not np.isfinite(fs) or fs <= 0:
            st.error("Heart-rate analysis needs a valid EKG sampling rate.")
            return

        try:
            with st.spinner("Detecting R-peaks in the trimmed EKG..."):
                clean, detected, auto = detect_r_peaks(walk["signal"], fs, invert_signal)
        except ModuleNotFoundError as exc:
            if exc.name != "neurokit2":
                raise
            st.error("Install NeuroKit2 in the environment running Streamlit with `python -m pip install neurokit2`.")
            return
        except (ValueError, IndexError) as exc:
            st.error(f"Could not detect R-peaks in this walk. {exc}")
            return

        st.session_state.walk_hr_review = {
            "key": analysis_key,
            "clean": clean,
            "detected": detected[time[detected] < TEST_DURATION_SECONDS],
            "auto": auto[time[auto] < TEST_DURATION_SECONDS],
            "peaks": auto[time[auto] < TEST_DURATION_SECONDS],
            "history": [],
        }

    review = st.session_state.get("walk_hr_review")
    if review is None or review["key"] != analysis_key:
        return

    clean, peaks = review["clean"], review["peaks"]
    if not review["auto"].size:
        st.warning("No automatic R-peaks passed the filters. Try inverting the signal or add peaks during review.")

    epoch = st.selectbox(
        "Review epoch",
        range(1, TEST_DURATION_SECONDS // HR_EPOCH_SECONDS + 1),
        format_func=lambda n: f"Epoch {n} ({(n - 1) * HR_EPOCH_SECONDS}–{n * HR_EPOCH_SECONDS} s)",
    )
    left, right = (epoch - 1) * HR_EPOCH_SECONDS, epoch * HR_EPOCH_SECONDS
    lo, hi = time.searchsorted([left, right])
    visible_peaks = peaks[(peaks >= lo) & (peaks < hi)]
    remove_col, add_col, undo_col = st.columns(3)
    if visible_peaks.size:
        peak_choices = {f"{time[i]:.4f} s (sample {i})": i for i in visible_peaks}
        chosen = remove_col.selectbox("Peak to remove", list(peak_choices), index=None)
        if remove_col.button("Remove peak", disabled=chosen is None):
            review["history"].append(peaks)
            peaks = peaks[peaks != peak_choices[chosen]]

    if hi > lo:
        add_at = add_col.number_input(
            "Add peak near walk time (s)", min_value=float(left), max_value=float(right),
            value=(left + right) / 2, step=0.001, format="%.3f", key=f"add_r_peak_time_{epoch}",
        )
        if add_col.button("Add peak"):
            sample = peak_near_time(time, clean, add_at, lo, hi)
            if sample not in peaks:
                review["history"].append(peaks)
                peaks = np.union1d(peaks, [sample])
    if undo_col.button("Undo last peak edit", disabled=not review["history"]):
        peaks = review["history"].pop()
    review["peaks"] = peaks

    chart_key = f"walk_r_peak_chart_{epoch}"

    def edit_clicked_peak():
        points = st.session_state[chart_key]["selection"]["points"]
        if len(points) != 1:
            return

        point = points[0]
        current = review["peaks"]
        sample = point.get("customdata")
        if sample is None:
            sample = peak_near_time(time, clean, float(point["x"]), lo, hi)
            if sample in current:
                return

        review["history"].append(current)
        review["peaks"] = (
            current[current != sample] if sample in current else np.union1d(current, [sample])
        )

    st.caption(
        "Click a diamond or star to remove that peak. Click the EKG trace near a missed peak to add it, "
        "or click a gray × to restore it. Use Undo last peak edit to reverse a change."
    )
    # Markers make the waveform samples selectable. WebGL keeps dense epochs responsive.
    fig = go.Figure(go.Scattergl(
        x=time[lo:hi], y=clean[lo:hi], name="EKG", mode="lines+markers",
        line=dict(width=1), marker=dict(size=3, opacity=0.2),
    ))
    for label, indices, color, symbol in (
        ("Kept automatic peaks", np.intersect1d(peaks, review["auto"]), "#E8790C", "diamond"),
        ("Removed peaks", np.setdiff1d(review["detected"], peaks), "#737373", "x"),
        ("Manually added peaks", np.setdiff1d(peaks, review["auto"]), "#1677BB", "star"),
    ):
        visible = indices[(indices >= lo) & (indices < hi)]
        fig.add_scatter(x=time[visible], y=clean[visible], customdata=visible.tolist(), mode="markers", name=label,
                        marker=dict(size=9, color=color, symbol=symbol))
    fig.update_xaxes(range=[left, right], title="Time since walk start (s)")
    fig.update_layout(title=f"Cleaned EKG · Epoch {epoch}", yaxis_title="EKG Signal (mV)",
                      height=425, clickmode="event+select", hovermode="closest")
    cursor_svg = (
        '<svg xmlns="http://www.w3.org/2000/svg" width="19" height="19">'
        '<path d="M9.5 1v5m0 7v5M1 9.5h5m7 0h5" fill="none" '
        'stroke="#757575" stroke-width="1" shape-rendering="crispEdges"/>'
        '</svg>'
    )
    st.html(f"""
        <style>
        .st-key-walk_peak_edit_plot .draglayer .nsewdrag {{
            cursor: url("data:image/svg+xml,{quote(cursor_svg, safe='')}") 9 9, crosshair !important;
        }}
        </style>
    """)
    with st.container(key="walk_peak_edit_plot"):
        st.plotly_chart(fig, use_container_width=True, key=chart_key,
                        on_select=edit_clicked_peak, selection_mode="points")

    beat_hr, epochs = summarize_walk_hr(time, peaks)
    beat_hr.insert(1, "export_sample_idx", beat_hr["walk_sample_idx"] + walk["export_start"])
    st.subheader("Heart rate by 15-second epoch")
    st.caption(
        "Epochs start at the walk marker. Each average uses unsmoothed HR values from 40–180 bpm, "
        "assigned by the time of the later R-peak. An epoch with no valid intervals has a blank mean."
    )
    st.dataframe(epochs, hide_index=True, use_container_width=True)

    hr_fig = go.Figure()
    hr_fig.add_scatter(x=beat_hr["time_s"], y=beat_hr["hr_bpm"], mode="markers", name="Beat-to-beat HR")
    hr_fig.add_scatter(x=beat_hr["time_s"], y=beat_hr["hr_smooth_bpm"], mode="lines", name="5-beat average")
    hr_fig.add_scatter(x=(epochs["start_s"] + epochs["end_s"]) / 2, y=epochs["mean_hr_bpm"],
                       mode="markers", name="15-second mean", marker=dict(size=9, symbol="diamond"))
    hr_fig.update_layout(xaxis_title="Time since walk start (s)", yaxis_title="Heart rate (bpm)", height=400)
    st.plotly_chart(hr_fig, use_container_width=True)

    peak_export = pd.DataFrame({
        "walk_sample_idx": peaks,
        "export_sample_idx": peaks + walk["export_start"],
        "time_s": time[peaks],
        "amplitude_mV": clean[peaks],
        "source": np.where(np.isin(peaks, review["auto"]), "automatic", "manual"),
    })
    stem = output_filename.removesuffix("_trigger_fixed.csv")
    for column, (label, table, suffix) in zip(st.columns(3), (
        ("Download R-peaks CSV", peak_export, "R_peaks"),
        ("Download beat-to-beat HR CSV", beat_hr, "HR_beat_to_beat"),
        ("Download 15-second HR CSV", epochs, "HR_15s"),
    )):
        column.download_button(label, table.to_csv(index=False), f"{stem}_{suffix}.csv", mime="text/csv")


#FILE UPLOAD

uploaded_file = st.file_uploader("Upload raw overground Delsys CSV",type=["csv"])

if uploaded_file is not None:
    file_digest = hashlib.sha256(uploaded_file.getbuffer()).hexdigest()

    # LOAD FILE
    try:
        info = extract_recording_metadata(uploaded_file)
        time_vector,ekg_data = load_ekg(uploaded_file,info)
    except Exception as e:
        st.error(f"Could not read the Delsys file:\n\n{e}")
        st.stop()

    delsys_recording_start = info["recording_start"]
    ekg_fs = info["ekg_fs"]
    collection_length = info["collection_length"]

    # Make EKG time relative to first EKG sample
    ekg_time_origin = time_vector[0]
    time_vector = time_vector-ekg_time_origin
    if np.any(np.diff(time_vector) < 0):
        st.error("EKG timestamps must be in chronological order.")
        st.stop()
    calculated_duration = time_vector[-1]

    has_analog_trigger = info["analog_columns"] is not None
    use_analog_trigger = False
    if has_analog_trigger:
        use_analog_trigger = st.radio(
            "6MWT start source",
            ["Recorded Analog 1 trigger","Manual clock time"],
        ) == "Recorded Analog 1 trigger"

    clock_offset = 0.0
    if not use_analog_trigger:
        st.subheader("Clock alignment (optional)")
        clock_offset = st.number_input(
            "How many seconds is the Delsys clock ahead (+) or behind (-)?",
            value=0.0,
            step=1.0,
            help=(
                "Enter a positive value if the Delsys clock is ahead of the clock "
                "used to record the 6MWT start time, or a negative value if it is behind."
            ),
        )

        st.caption(
            "Leave at 0 if the clocks are synchronized. "
            "Example: Delsys 2 min ahead = +120 s; Delsys 1 min behind = -60 s."
        )

    # Shift the entire Delsys recording clock window
    recording_start = delsys_recording_start-pd.to_timedelta(clock_offset,unit="s")
    recording_end = recording_start+pd.to_timedelta(calculated_duration,unit="s")

    # RECORDING INFORMATION

    st.subheader("Recording information")

    c1,c2,c3,c4 = st.columns([1.4,1,1,1])

    clock_label = "Delsys" if use_analog_trigger else "Corrected"
    c1.metric(f"{clock_label} recording start",recording_start.strftime("%Y-%m-%d %I:%M:%S %p"))
    c2.metric(f"{clock_label} recording end",recording_end.strftime("%I:%M:%S %p"))
    c3.metric("EKG sampling rate",f"{ekg_fs:.4f} Hz")
    c4.metric("Calculated duration",f"{calculated_duration:.3f} s")

    if not use_analog_trigger:
        st.caption(
            f"Original Delsys metadata start: {delsys_recording_start.strftime('%Y-%m-%d %I:%M:%S %p')} | "
            f"Clock offset: {clock_offset:+.1f} s"
        )

    if not np.isnan(collection_length):
        duration_difference = calculated_duration-collection_length
        st.caption(
            f"Delsys collection length: {collection_length:.3f} s | "
            f"Difference: {duration_difference:.4f} s")

        if abs(duration_difference) > 1:
            st.warning(
                "Calculated EKG duration differs from the Delsys Collection Length "
                "by more than 1 second.")

    # DETECT ANALOG TRIGGERS FOR THE PREVIEW AND PICKER
    analog_markers = {}
    threshold_v = None
    if use_analog_trigger:
        st.subheader("Recorded Analog 1 triggers")
        threshold_v = st.number_input(
            "Analog 1 falling-edge threshold (V)",
            value=DEFAULT_ANALOG_THRESHOLD_V,
            step=0.1,
            help="A trigger starts when the voltage drops from above the threshold to at or below it.",
        )
        try:
            analog_times = find_analog_trigger_times(uploaded_file,info,threshold_v)
        except (ValueError,pd.errors.ParserError) as e:
            st.error(f"Could not detect Analog 1 triggers. {e}")
        else:
            analog_markers = {
                f"Trigger {i}": float(seconds)
                for i,seconds in enumerate(analog_times-ekg_time_origin,start=1)
            }
            if not analog_markers:
                st.warning("No Analog 1 falling edges were detected at this threshold.")

    # RAW EKG
    with st.expander("Preview raw EKG",expanded=use_analog_trigger):
        if analog_markers:
            st.caption("The numbered Analog 1 markers match the start-trigger choices below.")
        raw_fig = make_ekg_plot(
            ekg_data=ekg_data,
            time_vector=time_vector,
            trigger_times=analog_markers.values(),
            trigger_labels=list(analog_markers),
            title="Raw Overground EKG",
        )
        st.plotly_chart(raw_fig,use_container_width=True)

    # 6MWT INPUT
    st.subheader("6MWT timing")

    selected_trigger = None
    sixmwt_start_clock = None
    if use_analog_trigger:
        st.write(
            "Choose the trigger that marks the start of this walk. "
            "The app sets the end marker 360 seconds later."
        )
        if analog_markers:
            trigger_options = {
                f"{label} at {seconds:.4f} s from EKG start": seconds
                for label,seconds in analog_markers.items()
            }
            selection = st.selectbox(
                "6MWT start trigger",
                options=list(trigger_options),
                index=None,
                placeholder="Choose the trigger for this walk",
                key=f"analog_start_{uploaded_file.name}_{file_digest}_{threshold_v}",
            )
            if selection is not None:
                selected_trigger = trigger_options[selection]
    else:
        st.write(
            "Enter the manually recorded clock time when the **6-minute walk test started**. "
            "The app sets the end marker 360 seconds later."
        )
        sixmwt_start_clock = st.text_input("**6MWT start time**",placeholder="Example: 10:51:13 AM")

    # OUTPUT TRIMMING
    st.subheader("Output trimming")

    trim_choice = st.radio(
        "Choose exported EKG range:",
        [
            "Keep full recording",
            "Start recording at 6MWT start",
            "Keep only the 6MWT"],
        index=2)

    source = (
        uploaded_file.name,file_digest,
        use_analog_trigger,threshold_v,selected_trigger,sixmwt_start_clock,clock_offset,trim_choice,
    )

    # PROCESS THE SELECTED START
    if st.button(
        "Process 6MWT EKG" if use_analog_trigger else "Reconstruct 6MWT Triggers",
        type="primary",
        key="reconstruct_button",
        disabled=use_analog_trigger and selected_trigger is None,
    ):
        if use_analog_trigger:
            start_seconds = selected_trigger
            start_datetime = recording_start+pd.to_timedelta(start_seconds,unit="s")
        else:
            try:
                start_datetime,start_seconds = parse_manual_clock_time(sixmwt_start_clock,recording_start)
            except Exception as e:
                st.error(str(e))
                st.stop()

        end_seconds = start_seconds+TEST_DURATION_SECONDS
        end_datetime = start_datetime+pd.Timedelta(seconds=TEST_DURATION_SECONDS)

        if start_seconds < 0:
            st.error(
                f"The 6MWT start occurs {abs(start_seconds):.2f} s before the EKG recording begins.")
            st.stop()

        if start_seconds > calculated_duration:
            st.error(
                f"The 6MWT start occurs at {start_seconds:.2f} s after the "
                f"recording start, but the recording is only {calculated_duration:.2f} s long.")
            st.stop()

        if end_seconds > calculated_duration+1e-8:
            missing_seconds = end_seconds-calculated_duration
            st.error(
                f"6MWT start: {start_datetime.strftime('%I:%M:%S %p')}\n\n"
                f"6MWT end: {end_datetime.strftime('%I:%M:%S %p')}\n\n"
                f"{clock_label} EKG recording end: {recording_end.strftime('%I:%M:%S %p')}\n\n"
                f"The EKG recording ends {missing_seconds:.2f} s before the 6MWT ends.")
            st.stop()

        # TRIGGER INDICES

        start_idx = int(np.argmin(np.abs(time_vector-start_seconds)))
        end_idx = int(np.argmin(np.abs(time_vector-end_seconds)))
        if end_idx <= start_idx:
            st.error("The selected walk start and end must match two distinct EKG samples.")
            st.stop()

        trigger_indices = np.array([start_idx,end_idx],dtype=int)
        trigger_labels = ["6MWT Start","6MWT End"]

        # TRIGGER SUMMARY
        trigger_info = pd.DataFrame({
            "Trigger Number":[1,2],
            "Trigger Label":["6MWT Start","6MWT End"],
            "6MWT Clock Time":[
                start_datetime.strftime("%I:%M:%S %p"),
                end_datetime.strftime("%I:%M:%S %p")],
            "Delsys Clock Offset (s)":[clock_offset,clock_offset],
            "Seconds From Corrected Recording Start":[start_seconds,end_seconds],
            "Original Sample Index":[start_idx,end_idx]})

        st.subheader("6MWT markers" if use_analog_trigger else "Reconstructed 6MWT triggers")
        st.dataframe(trigger_info,use_container_width=True,hide_index=True)

        # ORIGINAL RECORDING PLOT

        trigger_fig = make_ekg_plot(
            ekg_data=ekg_data,
            time_vector=time_vector,
            trigger_times=time_vector[trigger_indices],
            trigger_labels=trigger_labels,
            title="EKG with 6MWT Start and End")

        st.plotly_chart(trigger_fig,use_container_width=True)

        # TRIM DATA
        walk_slice = slice(start_idx,end_idx+1)
        walk_signal = ekg_data[walk_slice]
        walk_time = time_vector[walk_slice]-time_vector[start_idx]

        if trim_choice == "Keep full recording":
            ekg_output = ekg_data.copy()
            output_time = time_vector.copy()
            output_trigger_indices = trigger_indices.copy()

        elif trim_choice == "Start recording at 6MWT start":
            ekg_output = ekg_data[start_idx:]
            output_time = time_vector[start_idx:]-time_vector[start_idx]
            output_trigger_indices = np.array([0,end_idx-start_idx],dtype=int)

        else:
            ekg_output = walk_signal
            output_time = walk_time
            output_trigger_indices = np.array([0,len(ekg_output)-1],dtype=int)

        # BINARY TRIGGER
        binary_trigger = np.zeros(len(ekg_output),dtype=int)
        binary_trigger[output_trigger_indices] = 1

        # FINAL PLOT
        final_fig = make_ekg_plot(
            ekg_data=ekg_output,
            time_vector=output_time,
            trigger_times=output_time[output_trigger_indices],
            trigger_labels=trigger_labels,
            title="Processed 6MWT EKG")

        st.plotly_chart(final_fig,use_container_width=True)

        # EXPORT DATA
        output_trigger_times = output_time[output_trigger_indices]

        export_data = pd.DataFrame({
            "Time (s)":output_time,
            "EKG Signal":ekg_output,
            "Binary Trigger":binary_trigger})

        trigger_export = pd.DataFrame({
            "Trigger Label":["6MWT Start","6MWT End"],
            "6MWT Clock Time":[
                start_datetime.strftime("%I:%M:%S %p"),
                end_datetime.strftime("%I:%M:%S %p")],
            "Original Delsys Recording Start":[
                delsys_recording_start.strftime("%I:%M:%S %p"),
                delsys_recording_start.strftime("%I:%M:%S %p")],
            "Corrected EKG Recording Start":[
                recording_start.strftime("%I:%M:%S %p"),
                recording_start.strftime("%I:%M:%S %p")],
            "Delsys Clock Offset (s)":[clock_offset,clock_offset],
            "Seconds From Corrected Recording Start":[start_seconds,end_seconds],
            "Trigger Onset Times (s)":[
                output_trigger_times[0],
                output_trigger_times[1]]})

        export_data = pd.concat([export_data,trigger_export],axis=1)

        # STORE PROCESSED DATA
        original_name = uploaded_file.name

        if original_name.lower().endswith(".csv"):
            original_name = original_name[:-4]

        output_filename = f"{original_name}_6MWT_trigger_fixed.csv"

        st.session_state.export_data = export_data
        st.session_state.processed_csv = export_data.to_csv(index=False).encode("utf-8")
        st.session_state.processed_walk = {
            "signal": walk_signal,
            "time": walk_time,
            "export_start": output_trigger_indices[0],
        }
        st.session_state.output_filename = output_filename
        st.session_state.processed_source = source
        st.session_state.pop("walk_hr_review", None)

        st.success("✅ 6MWT EKG processed successfully.")

    # OUTPUT PREVIEW
    if (st.session_state.export_data is not None
        and st.session_state.processed_source == source):
        st.subheader("Output preview")
        st.dataframe(
            st.session_state.export_data.head(100),
            use_container_width=True)

        # # SAVE PROCESSED DATA
        # st.divider()
        # st.subheader("Export processed data")

        # full_save_path = SAVE_FOLDER/st.session_state.output_filename

        # st.write(f"**File name:** `{st.session_state.output_filename}`")
        # st.write(f"**Save folder:** `{SAVE_FOLDER}`")

        # if st.button("💾 Save processed CSV",type="primary",key="save_processed_csv"):
        #     try:
        #         SAVE_FOLDER.mkdir(parents=True,exist_ok=True)

        #         st.session_state.export_data.to_csv(
        #             full_save_path,
        #             index=False)

        #         if full_save_path.exists():
        #             file_size = full_save_path.stat().st_size
        #             st.success(
        #                 f"✅ File saved successfully!\n\n"
        #                 f"{full_save_path}\n\n"
        #                 f"File size: {file_size:,} bytes")
        #         else:
        #             st.error("The save command finished, but the file could not be found.")

        #     except Exception as e:
        #         st.error(f"Could not save file:\n\n{e}")

        # DOWNLOAD PROCESSED DATA
        st.divider()
        st.subheader("Export processed data")

        st.write(f"**File name:** `{st.session_state.output_filename}`")

        st.download_button(
            label="💾 Download processed CSV",
            data=st.session_state.processed_csv,
            file_name=st.session_state.output_filename,
            mime="text/csv",
            type="primary",
            key="download_processed_csv"
        )

        show_walk_hr(st.session_state.get("processed_walk"), ekg_fs, source, st.session_state.output_filename)
