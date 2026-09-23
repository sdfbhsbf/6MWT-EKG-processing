#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Streamlit app for reconstructing 6MWT EKG triggers from manually recorded real-world clock time

Enters the 6MWT START time.

The app automatically creates the 6MWT END trigger exactly 6 minutes (360 s) later.

@author: beier
"""

import csv
# from pathlib import Path
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

# #change save folder
# SAVE_FOLDER = Path("/Users/beier/Library/CloudStorage/OneDrive-UBC/Courtney's team - Mobility and Balance Rehab Lab - STUDY 3 - STROKE training/Training DATA/training - EKG raw (trigger fixed)/overground")
# SAVE_FOLDER.mkdir(parents=True,exist_ok=True)

st.set_page_config(page_title="6MWT EKG Trigger Reconstruction",layout="wide")
st.title("6MWT EKG Trigger Reconstruction")

st.write(
    """
    Upload a raw Delsys overground EKG CSV file and enter the manually recorded
    **6MWT start time**.

    The app reconstructs the 6MWT start trigger from the Delsys recording clock
    and automatically creates a second trigger exactly **6 minutes later**.
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
    raw_bytes = uploaded_file.getvalue()
    text = raw_bytes.decode("utf-8-sig",errors="replace")
    reader = csv.reader(text.splitlines())
    rows = []
    for i,row in enumerate(reader):
        rows.append(row)
        if i >= nrows-1:
            break
    return rows

def extract_recording_metadata(uploaded_file):
    channel_row = 5
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

    channel_names = [str(c).strip() for c in rows[channel_row]]
    ekg_candidates = [i for i,c in enumerate(channel_names) if c.upper().startswith("EKG")]

    if len(ekg_candidates) == 0:
        raise ValueError(f"Could not identify an EKG channel.\n\nDetected channels: {channel_names}")

    ekg_signal_idx = ekg_candidates[0]
    sampling_values = rows[sampling_row]

    if ekg_signal_idx >= len(sampling_values):
        raise ValueError("Could not find EKG sampling frequency.")

    sampling_text = str(sampling_values[ekg_signal_idx]).replace("Hz","").strip()

    try:
        ekg_fs = float(sampling_text)
    except Exception as e:
        raise ValueError(f"Could not parse EKG sampling frequency from '{sampling_text}'.") from e

    return {
        "recording_start":recording_start,
        "collection_length":collection_length,
        "channel_names":channel_names,
        "ekg_signal_idx":ekg_signal_idx,
        "ekg_fs":ekg_fs,
        "data_start":data_start}

def load_ekg(uploaded_file,metadata_info):
    uploaded_file.seek(0)
    ekg_raw = pd.read_csv(
        uploaded_file,
        skiprows=metadata_info["data_start"],
        header=None,
        usecols=[metadata_info["ekg_signal_idx"]]
    )
    ekg_raw.columns = ["EKG"]
    ekg_signal = pd.to_numeric(ekg_raw["EKG"],errors="coerce")
    ekg_signal = ekg_signal.dropna().reset_index(drop=True)

    if len(ekg_signal) == 0:
        raise ValueError("No valid numeric EKG data found.")

    return ekg_signal.to_numpy()

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

def make_ekg_plot(ekg_data,time_vector,trigger_indices=None,trigger_labels=None,title="EKG Signal"):
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=time_vector,y=ekg_data,mode="lines",name="EKG",line=dict(width=1)))

    if trigger_indices is not None:
        for i,idx in enumerate(trigger_indices):
            if idx < 0 or idx >= len(time_vector):
                continue

            trigger_time = time_vector[idx]
            fig.add_vline(x=trigger_time,line_width=2,line_dash="dash")

            if trigger_labels is not None:
                fig.add_annotation(x=trigger_time,y=1,yref="paper",text=trigger_labels[i],showarrow=False,textangle=-90,xanchor="left",yanchor="top")

    fig.update_layout(title=title,xaxis_title="Time (s)",yaxis_title="EKG Signal (mV)",height=600,hovermode="x unified")
    return fig


#FILE UPLOAD

uploaded_file = st.file_uploader("Upload raw overground Delsys CSV",type=["csv"])

if uploaded_file is not None:

    # LOAD FILE
    try:
        info = extract_recording_metadata(uploaded_file)
        ekg_data = load_ekg(uploaded_file,info)
    except Exception as e:
        st.error(f"Could not read the Delsys file:\n\n{e}")
        st.stop()

    recording_start = info["recording_start"]
    ekg_fs = info["ekg_fs"]
    collection_length = info["collection_length"]

    time_vector = np.arange(len(ekg_data))/ekg_fs
    calculated_duration = len(ekg_data)/ekg_fs
    recording_end = recording_start+pd.to_timedelta(calculated_duration,unit="s")

    # RECORDING INFORMATION

    st.subheader("Recording information")

    c1,c2,c3,c4 = st.columns([1.4,1,1,1])

    c1.metric("Recording start",recording_start.strftime("%Y-%m-%d %I:%M:%S %p"))
    c2.metric("Recording end",recording_end.strftime("%I:%M:%S %p"))
    c3.metric("EKG sampling rate",f"{ekg_fs:.4f} Hz")
    c4.metric("Calculated duration",f"{calculated_duration:.3f} s")

    if not np.isnan(collection_length):
        duration_difference = calculated_duration-collection_length
        st.caption(
            f"Delsys collection length: {collection_length:.3f} s | "
            f"Difference: {duration_difference:.4f} s")

        if abs(duration_difference) > 1:
            st.warning(
                "Calculated EKG duration differs from the Delsys Collection Length "
                "by more than 1 second.")

    # RAW EKG
    with st.expander("Preview raw EKG"):
        raw_fig = make_ekg_plot(ekg_data=ekg_data,time_vector=time_vector,title="Raw Overground EKG")
        st.plotly_chart(raw_fig,use_container_width=True)

    # 6MWT INPUT
    st.subheader("6MWT timing")

    st.write(
        """
        Enter the manually recorded clock time when the **6-minute walk test started**.

        The end trigger will automatically be created exactly **360 seconds later**.
        """)

    st.subheader("6MWT start time")
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

    # RECONSTRUCT BUTTON
    if st.button("Reconstruct 6MWT Triggers",type="primary",key="reconstruct_button"):

        try:
            start_datetime,start_seconds = parse_manual_clock_time(sixmwt_start_clock,recording_start)
        except Exception as e:
            st.error(str(e))
            st.stop()

        end_seconds = start_seconds+360
        end_datetime = start_datetime+pd.Timedelta(seconds=360)

        if start_seconds < 0:
            st.error("The reconstructed 6MWT start occurs before the EKG recording begins.")
            st.stop()

        if start_seconds > calculated_duration:
            st.error(
                f"The reconstructed 6MWT start occurs at {start_seconds:.2f} s, "
                f"but the recording is only {calculated_duration:.2f} s long.")
            st.stop()

        if end_seconds > calculated_duration:
            st.error(
                f"The 6MWT would end at {end_seconds:.2f} s after recording start, "
                f"but the EKG recording is only {calculated_duration:.2f} s long.\n\n"
                f"The recording does not contain the full 6-minute walk.")
            st.stop()

        # TRIGGER INDICES

        start_idx = int(round(start_seconds*ekg_fs))
        end_idx = int(round(end_seconds*ekg_fs))

        start_idx = int(np.clip(start_idx,0,len(ekg_data)-1))
        end_idx = int(np.clip(end_idx,0,len(ekg_data)-1))

        trigger_indices = np.array([start_idx,end_idx],dtype=int)
        trigger_labels = ["6MWT Start","6MWT End"]


        # TRIGGER SUMMARY
        trigger_info = pd.DataFrame({
            "Trigger Number":[1,2],
            "Trigger Label":["6MWT Start","6MWT End"],
            "Clock Time":[
                start_datetime.strftime("%I:%M:%S %p"),
                end_datetime.strftime("%I:%M:%S %p")],
            "Seconds From Original Recording Start":[start_seconds,end_seconds],
            "Original Sample Index":[start_idx,end_idx]})

        st.subheader("Reconstructed 6MWT triggers")
        st.dataframe(trigger_info,use_container_width=True,hide_index=True)

        # ORIGINAL RECORDING PLOT

        trigger_fig = make_ekg_plot(
            ekg_data=ekg_data,
            time_vector=time_vector,
            trigger_indices=trigger_indices,
            trigger_labels=trigger_labels,
            title="EKG with Reconstructed 6MWT Start and End")

        st.plotly_chart(trigger_fig,use_container_width=True)

        # TRIM DATA

        if trim_choice == "Keep full recording":
            ekg_output = ekg_data.copy()
            output_time = time_vector.copy()
            output_trigger_indices = trigger_indices.copy()

        elif trim_choice == "Start recording at 6MWT start":
            ekg_output = ekg_data[start_idx:]
            output_time = np.arange(len(ekg_output))/ekg_fs
            output_trigger_indices = np.array([0,end_idx-start_idx],dtype=int)

        else:
            ekg_output = ekg_data[start_idx:end_idx+1]
            output_time = np.arange(len(ekg_output))/ekg_fs
            output_trigger_indices = np.array([0,len(ekg_output)-1],dtype=int)

        # BINARY TRIGGER
        binary_trigger = np.zeros(len(ekg_output),dtype=int)
        binary_trigger[output_trigger_indices] = 1

        # FINAL PLOT
        final_fig = make_ekg_plot(
            ekg_data=ekg_output,
            time_vector=output_time,
            trigger_indices=output_trigger_indices,
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
            "Recorded / Calculated Clock Time":[
                start_datetime.strftime("%I:%M:%S %p"),
                end_datetime.strftime("%I:%M:%S %p")],
            "Seconds From Original Recording Start":[start_seconds,end_seconds],
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
        st.session_state.output_filename = output_filename
        st.session_state.processed_source = uploaded_file.name

        st.success("✅ 6MWT triggers reconstructed successfully.")


    # OUTPUT PREVIEW
    if (st.session_state.export_data is not None
        and st.session_state.processed_source == uploaded_file.name):
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
        
        csv_data = st.session_state.export_data.to_csv(index=False).encode("utf-8")
        
        st.write(f"**File name:** `{st.session_state.output_filename}`")
        
        st.download_button(
            label="💾 Download processed CSV",
            data=csv_data,
            file_name=st.session_state.output_filename,
            mime="text/csv",
            type="primary",
            key="download_processed_csv"
        )          
                    
                    
                    