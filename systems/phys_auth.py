import base64
import io
import numpy as np
import plotly.graph_objs as go
import dash
from dash import html, dcc, Input, Output, State
import dash_bootstrap_components as dbc
import neurokit2 as nk
import matplotlib.pyplot as plt
from scipy.signal import welch

# --------------------
# ECG Analysis function using NeuroKit2 and spectral analysis
# --------------------
def analyze_ecg(ecg_signal, fs=500, verbose=False):
    """
    Process the ECG signal and compute fiducial features, HRV, and RR spectral metrics.
    Returns a dictionary of results and a classification ('Human' or 'Fake').
    """
    try:
        signals, info = nk.ecg_process(ecg_signal, sampling_rate=fs)
    except Exception as e:
        if verbose:
            print(f"[Error] NeuroKit2 failed to process ECG: {e}")
        return {"classification": "Fake", "error": str(e)}
    
    r_peaks = info["ECG_R_Peaks"]
    if len(r_peaks) < 2:
        if verbose:
            print("[Warning] Not enough R-peaks detected => likely Fake.")
        return {"classification": "Fake", "warning": "Not enough R-peaks detected."}
    
    try:
        delineation_signals, wave_peaks = nk.ecg_delineate(
            signals["ECG_Clean"],
            r_peaks,
            sampling_rate=fs,
            method="dwt"
        )
    except Exception as e:
        if verbose:
            print(f"[Error] Could not delineate waves: {e}")
        return {"classification": "Fake", "error": str(e)}
    
    # Initialize lists for fiducial features
    pr_intervals = []
    qrs_durations = []
    qt_intervals = []
    p_durations = []
    r_amplitudes = []
    t_amplitudes = []
    
    for idx, r_idx in enumerate(r_peaks):
        p_onset  = wave_peaks["ECG_P_Onsets"][idx] if idx < len(wave_peaks["ECG_P_Onsets"]) else None
        p_offset = wave_peaks["ECG_P_Offsets"][idx] if idx < len(wave_peaks["ECG_P_Offsets"]) else None
        q_onset  = wave_peaks["ECG_R_Onsets"][idx] if idx < len(wave_peaks["ECG_R_Onsets"]) else None
        s_offset = wave_peaks["ECG_R_Offsets"][idx] if idx < len(wave_peaks["ECG_R_Offsets"]) else None
        t_onset  = wave_peaks["ECG_T_Onsets"][idx] if idx < len(wave_peaks["ECG_T_Onsets"]) else None
        t_offset = wave_peaks["ECG_T_Offsets"][idx] if idx < len(wave_peaks["ECG_T_Offsets"]) else None
        p_peak   = wave_peaks["ECG_P_Peaks"][idx] if idx < len(wave_peaks["ECG_P_Peaks"]) else None
        t_peak   = wave_peaks["ECG_T_Peaks"][idx] if idx < len(wave_peaks["ECG_T_Peaks"]) else None

        # Skip beat if any critical index is missing
        if any(v is None for v in [p_onset, p_offset, q_onset, s_offset, t_onset, t_offset]):
            continue

        # Convert indices (samples) to milliseconds
        p_onset_time   = p_onset   / fs * 1000
        p_offset_time  = p_offset  / fs * 1000
        q_onset_time   = q_onset   / fs * 1000
        s_offset_time  = s_offset  / fs * 1000
        r_time         = r_idx     / fs * 1000
        t_onset_time   = t_onset   / fs * 1000
        t_offset_time  = t_offset  / fs * 1000

        # Compute intervals and durations
        pr_interval_ms = r_time - p_onset_time
        if pr_interval_ms > 0:
            pr_intervals.append(pr_interval_ms)
        qrs_duration_ms = s_offset_time - q_onset_time
        if qrs_duration_ms > 0:
            qrs_durations.append(qrs_duration_ms)
        qt_interval_ms = t_offset_time - q_onset_time
        if qt_interval_ms > 0:
            qt_intervals.append(qt_interval_ms)
        p_duration_ms = p_offset_time - p_onset_time
        if p_duration_ms > 0:
            p_durations.append(p_duration_ms)

        # Record amplitudes from the cleaned signal
        if r_idx < len(signals["ECG_Clean"]):
            r_amplitudes.append(signals["ECG_Clean"][r_idx])
        if t_peak is not None and t_peak < len(signals["ECG_Clean"]):
            t_amplitudes.append(signals["ECG_Clean"][t_peak])
    
    if len(pr_intervals) < 2 or len(qrs_durations) < 2:
        if verbose:
            print("[Warning] Not enough valid beats detected => likely Fake.")
        return {"classification": "Fake", "warning": "Not enough valid beats detected."}
    
    pr_mean  = np.mean(pr_intervals)
    qrs_mean = np.mean(qrs_durations)
    qt_mean  = np.mean(qt_intervals) if len(qt_intervals) > 0 else 0
    p_mean   = np.mean(p_durations) if len(p_durations) > 0 else 0
    r_mean   = np.mean(r_amplitudes) if len(r_amplitudes) > 0 else 0
    t_mean   = np.mean(t_amplitudes) if len(t_amplitudes) > 0 else 0

    # HRV (Time-domain) Analysis
    rr_intervals = np.diff(r_peaks) / fs * 1000  # in milliseconds
    rr_mean = np.mean(rr_intervals) if len(rr_intervals) > 0 else 0
    sdnn = np.std(rr_intervals) if len(rr_intervals) > 1 else 0
    rmssd = np.sqrt(np.mean(np.diff(rr_intervals)**2)) if len(rr_intervals) > 1 else 0

    # RR Spectral Analysis using Welch's method
    rr_intervals_sec = np.diff(r_peaks) / fs
    first_beat_time = r_peaks[0] / fs
    beat_times = np.insert(np.cumsum(rr_intervals_sec), 0, first_beat_time)
    rr_times = (beat_times[:-1] + beat_times[1:]) / 2.0
    resample_rate = 4.0  # Hz
    time_interp = np.arange(beat_times[0], beat_times[-1], 1/resample_rate)
    rr_interp = np.interp(time_interp, rr_times, rr_intervals_sec)
    f, pxx = welch(rr_interp, fs=resample_rate, nperseg=len(rr_interp))
    lf_mask = (f >= 0.04) & (f < 0.15)
    hf_mask = (f >= 0.15) & (f < 0.4)
    lf_power = np.trapz(pxx[lf_mask], f[lf_mask])
    hf_power = np.trapz(pxx[hf_mask], f[hf_mask])
    lf_hf_ratio = lf_power / hf_power if hf_power > 0 else np.nan

    if verbose:
        print("\n==== ECG Analysis Results ====")
        print(f"Detected beats: {len(r_peaks)}")
        print(f"PR interval mean (ms):  {pr_mean:.2f}")
        print(f"RR interval mean (ms):  {rr_mean:.2f}")
        print(f"QRS duration mean (ms): {qrs_mean:.2f}")
        print(f"QT interval mean (ms):  {qt_mean:.2f}")
        print(f"P wave duration mean (ms): {p_mean:.2f}")
        print(f"R amplitude mean (mV): {r_mean:.4f}")
        print(f"T amplitude mean (mV): {t_mean:.4f}")
        print(f"HRV - SDNN (ms): {sdnn:.2f}")
        print(f"HRV - RMSSD (ms): {rmssd:.2f}")
        print(f"LF power: {lf_power:.4f}")
        print(f"HF power: {hf_power:.4f}")
        print(f"LF/HF ratio: {lf_hf_ratio:.4f}")

    # Naive thresholds for classification (example values)
    if not (120 <= pr_mean <= 220):
        classification = "Fake"
    elif not (60 <= qrs_mean <= 130):
        classification = "Fake"
    elif not (250 <= qt_mean <= 550):
        classification = "Fake"
    elif not (50 <= p_mean <= 140):
        classification = "Fake"
    elif not (0.05 <= r_mean <= 3.0):
        classification = "Fake"
    elif not (0 <= t_mean <= 1.5):
        classification = "Fake"
    elif sdnn < 20:
        classification = "Fake"
    elif np.std(qt_intervals) < 10:
        classification = "Fake"
    elif lf_hf_ratio < 0.5 or lf_hf_ratio > 3.0:
        classification = "Fake"
    else:
        classification = "Human"

    return {
        "classification": classification,
        "num_beats": len(r_peaks),
        "PR_mean": pr_mean,
        "QRS_mean": qrs_mean,
        "QT_mean": qt_mean,
        "P_duration_mean": p_mean,
        "R_amplitude_mean": r_mean,
        "T_amplitude_mean": t_mean,
        "RR_mean": rr_mean,
        "SDNN": sdnn,
        "RMSSD": rmssd,
        "LF_power": lf_power,
        "HF_power": hf_power,
        "LF/HF_ratio": lf_hf_ratio
    }

# --------------------
# Dash App Setup
# --------------------
app = dash.Dash(__name__, external_stylesheets=[dbc.themes.CYBORG])
server = app.server

# Layout Components
header = dbc.Row(
    dbc.Col(
        html.H1("ECG Enhanced Analysis", className="text-center text-white mt-3", style={"fontWeight": "bold"}),
        width=12
    ),
    justify="center"
)

upload_card = dbc.Card(
    [
        dbc.CardHeader(html.H4("Upload Your ECG File (.asc)", className="text-white"), className="bg-dark"),
        dbc.CardBody(
            [
                dcc.Upload(
                    id='upload-ecg',
                    children=html.Div(['Drag and Drop or ', html.A('Select ECG File (.asc)')]),
                    style={
                        'width': '100%', 'height': '60px', 'lineHeight': '60px',
                        'borderWidth': '1px', 'borderStyle': 'dashed',
                        'borderRadius': '5px', 'textAlign': 'center', 'margin': '10px'
                    },
                    multiple=False
                ),
                html.Div(id='upload-status', className="text-warning mb-3"),
            ]
        )
    ],
    className="mt-4"
)

graph_card = dbc.Card(
    [
        dbc.CardHeader(html.H4("Raw ECG Signal", className="text-white mb-0"), className="bg-dark"),
        dbc.CardBody(
            dcc.Graph(id='ecg-graph', style={"border": "1px solid #444"})
        )
    ],
    className="mt-4"
)

results_card = dbc.Card(
    [
        dbc.CardHeader(html.H4("ECG Analysis Results", className="text-white mb-0"), className="bg-dark"),
        dbc.CardBody(
            html.Div(id='analysis-results')
        )
    ],
    className="mt-4"
)

app.layout = html.Div(
    style={"minHeight": "100vh", "backgroundColor": "#2c2c2c", "padding": "20px"},
    children=[
        dbc.Container(
            [
                header,
                upload_card,
                graph_card,
                results_card
            ],
            fluid=True
        )
    ]
)

# --------------------
# Callback to process the uploaded file and update visualization and analysis results
# --------------------
@app.callback(
    [
        Output('upload-status', 'children'),
        Output('ecg-graph', 'figure'),
        Output('analysis-results', 'children')
    ],
    Input('upload-ecg', 'contents'),
    State('upload-ecg', 'filename')
)
def process_uploaded_file(contents, filename):
    if contents is not None:
        content_type, content_string = contents.split(',')
        decoded = base64.b64decode(content_string)
        try:
            # Read the uploaded file into a numpy array.
            s = io.StringIO(decoded.decode('utf-8'))
            ecg_data = np.loadtxt(s)
            if ecg_data.ndim > 1:
                ecg_data = ecg_data[:, 0]  # use first column if multiple exist
            
            # Create an ECG plot using Plotly for the raw signal.
            fig = go.Figure(data=go.Scatter(y=ecg_data, mode='lines', line=dict(color='orange')))
            fig.update_layout(
                title='ECG Signal',
                xaxis_title='Samples',
                yaxis_title='Amplitude',
                template='plotly_dark',
                paper_bgcolor='#2c2c2c',
                plot_bgcolor='#2c2c2c',
                font=dict(color='white')
            )
            
            # Analyze the ECG signal.
            result = analyze_ecg(ecg_data, fs=500, verbose=True)
            
            # Build tables for displaying the analysis results.
            table1 = html.Table(
                [
                    html.Tr([html.Th("Metric", style={"padding": "8px", "border": "1px solid #ddd"}),
                             html.Th("Value", style={"padding": "8px", "border": "1px solid #ddd"})]),
                    html.Tr([html.Td("Classification"), html.Td(result.get("classification", "N/A"))]),
                    html.Tr([html.Td("Number of Beats"), html.Td(result.get("num_beats", "N/A"))]),
                    html.Tr([html.Td("PR interval mean (ms)"), html.Td(f"{result.get('PR_mean', 0):.2f}")]),
                    html.Tr([html.Td("QRS duration mean (ms)"), html.Td(f"{result.get('QRS_mean', 0):.2f}")]),
                    html.Tr([html.Td("QT interval mean (ms)"), html.Td(f"{result.get('QT_mean', 0):.2f}")]),
                    html.Tr([html.Td("P wave duration mean (ms)"), html.Td(f"{result.get('P_duration_mean', 0):.2f}")]),
                    html.Tr([html.Td("R amplitude mean (mV)"), html.Td(f"{result.get('R_amplitude_mean', 0):.4f}")]),
                    html.Tr([html.Td("T amplitude mean (mV)"), html.Td(f"{result.get('T_amplitude_mean', 0):.4f}")])
                ],
                style={"width": "100%", "borderCollapse": "collapse", "margin": "20px 0", "color": "white"}
            )

            table2 = html.Table(
                [
                    html.Tr([html.Th("Metric", style={"padding": "8px", "border": "1px solid #ddd"}),
                             html.Th("Value", style={"padding": "8px", "border": "1px solid #ddd"})]),
                    html.Tr([html.Td("RR interval mean (ms)"), html.Td(f"{result.get('RR_mean', 0):.2f}")]),
                    html.Tr([html.Td("SDNN (ms)"), html.Td(f"{result.get('SDNN', 0):.2f}")]),
                    html.Tr([html.Td("RMSSD (ms)"), html.Td(f"{result.get('RMSSD', 0):.2f}")])
                ],
                style={"width": "100%", "borderCollapse": "collapse", "margin": "20px 0", "color": "white"}
            )

            table3 = html.Table(
                [
                    html.Tr([html.Th("Metric", style={"padding": "8px", "border": "1px solid #ddd"}),
                             html.Th("Value", style={"padding": "8px", "border": "1px solid #ddd"})]),
                    html.Tr([html.Td("LF power"), html.Td(f"{result.get('LF_power', 0):.4f}")]),
                    html.Tr([html.Td("HF power"), html.Td(f"{result.get('HF_power', 0):.4f}")]),
                    html.Tr([html.Td("LF/HF ratio"), html.Td(f"{result.get('LF/HF_ratio', 0):.4f}")])
                ],
                style={"width": "100%", "borderCollapse": "collapse", "margin": "20px 0", "color": "white"}
            )
            
            # Generate the NeuroKit2 processed ECG diagram.
            plt.rcParams["figure.figsize"] = (12, 8)
            signals2, info2 = nk.ecg_process(ecg_data, sampling_rate=500)
            nk.ecg_plot(signals2, info2)
            buf = io.BytesIO()
            plt.savefig(buf, format="png", bbox_inches="tight")
            buf.seek(0)
            neurokit_image = base64.b64encode(buf.read()).decode("utf-8")
            plt.close()

            neurokit_img_div = html.Div(
                [
                    html.H4("NeuroKit2 Processed ECG Diagram", className="text-white"),
                    html.Img(src=f"data:image/png;base64,{neurokit_image}", style={"width": "50%", "height": "auto", "marginBottom": "20px"})
                ]
            )
            
            results_div = html.Div(
                [
                    neurokit_img_div,
                    html.H4("Fiducial & Amplitude Metrics", className="text-white"),
                    table1,
                    html.H4("HRV Metrics", className="text-white"),
                    table2,
                    html.H4("RR Spectral Analysis", className="text-white"),
                    table3
                ],
                style={"marginTop": "20px"}
            )
            
            return f"File '{filename}' loaded successfully.", fig, results_div

        except Exception as e:
            return f"Error processing the file: {str(e)}", go.Figure(), html.Div()
    else:
        return "Please upload an ECG file (.asc)", go.Figure(), html.Div()

# --------------------
# Run Server
# --------------------
if __name__ == '__main__':
    app.run_server(debug=True)
