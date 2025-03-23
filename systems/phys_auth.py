import os
import base64
import io
import numpy as np
import pandas as pd
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

        if any(v is None for v in [p_onset, p_offset, q_onset, s_offset, t_onset, t_offset]):
            continue

        p_onset_time   = p_onset   / fs * 1000
        p_offset_time  = p_offset  / fs * 1000
        q_onset_time   = q_onset   / fs * 1000
        s_offset_time  = s_offset  / fs * 1000
        r_time         = r_idx     / fs * 1000
        t_onset_time   = t_onset   / fs * 1000
        t_offset_time  = t_offset  / fs * 1000

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

    rr_intervals = np.diff(r_peaks) / fs * 1000
    rr_mean = np.mean(rr_intervals) if len(rr_intervals) > 0 else 0
    sdnn = np.std(rr_intervals) if len(rr_intervals) > 1 else 0
    rmssd = np.sqrt(np.mean(np.diff(rr_intervals)**2)) if len(rr_intervals) > 1 else 0

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
    }

# --------------------
# Dash App Setup
# --------------------
app = dash.Dash(__name__, external_stylesheets=[dbc.themes.CYBORG])
server = app.server

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
        dbc.CardHeader(html.H4("Raw ECG Signal (Normalized)", className="text-white mb-0"), className="bg-dark"),
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

# ------------------------------------------------------------------------------
# Callback: Process uploaded file, update visualization and analysis results,
# and display additional graphs comparing with the 2000 ECG dataset.
# ------------------------------------------------------------------------------
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
            s = io.StringIO(decoded.decode('utf-8'))
            ecg_data = np.loadtxt(s)
            if ecg_data.ndim > 1:
                ecg_data = ecg_data[:, 0]
            
            # Apply MinMax scaling to normalize the ECG signal to [-1, 1]
            scaled_ecg_data = 2 * (ecg_data - np.min(ecg_data)) / (np.max(ecg_data) - np.min(ecg_data)) - 1
            
            # Plot the normalized raw ECG signal.
            raw_fig = go.Figure(data=go.Scatter(y=scaled_ecg_data, mode='lines', line=dict(color='orange')))
            raw_fig.update_layout(
                title='ECG Signal (Normalized to [-1, 1])',
                xaxis_title='Samples',
                yaxis_title='Normalized Amplitude',
                template='plotly_dark',
                paper_bgcolor='#2c2c2c',
                plot_bgcolor='#2c2c2c',
                font=dict(color='white')
            )
            
            # Analyze the original (unscaled) ECG for physical metrics.
            result = analyze_ecg(scaled_ecg_data, fs=500, verbose=True)
            
            # Build result tables.
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

            # Generate NeuroKit2 processed ECG diagram.
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
            
            # --------------------------
            # Additional Graphs: Compare with 2000 ECG dataset.
            # --------------------------
            csv_filename = "C:/Users/M2-Winterfell/Downloads/GAN-models-for-Bio-Authentication-through-ECG-signals/systems/real_ecgs_metrics.csv"
            if os.path.exists(csv_filename):
                df_metrics = pd.read_csv(csv_filename)
            else:
                df_metrics = pd.DataFrame()
            
            metrics_keys = {
                "PR_mean": "PR interval mean (ms)",
                "QRS_mean": "QRS duration mean (ms)",
                "QT_mean": "QT interval mean (ms)",
                "P_duration_mean": "P wave duration mean (ms)",
                "R_amplitude_mean": "R amplitude mean (mV)",
                "T_amplitude_mean": "T amplitude mean (mV)",
                "RR_mean": "RR interval mean (ms)",
                "SDNN": "SDNN (ms)",
                "RMSSD": "RMSSD (ms)"
            }
            
            graphs = []
            for key, label in metrics_keys.items():
                fig_metric = go.Figure()
                if not df_metrics.empty and key in df_metrics.columns:
                    fig_metric.add_trace(go.Histogram(x=df_metrics[key], nbinsx=50, name="Dataset"))
                    new_value = result.get(key, None)
                    if new_value is not None:
                        fig_metric.add_vline(x=new_value, line_color="red", line_width=3,
                                               annotation_text="New ECG", annotation_position="top right")
                fig_metric.update_layout(
                    title=label,
                    xaxis_title=label,
                    yaxis_title="Count",
                    template="plotly_dark",
                    paper_bgcolor='#2c2c2c',
                    plot_bgcolor='#2c2c2c',
                    font=dict(color='white')
                )
                graphs.append(
                    dbc.Col(dcc.Graph(figure=fig_metric), width=3)
                )
            
            rows = []
            for i in range(0, len(graphs), 4):
                row = dbc.Row(graphs[i:i+4], className="mb-4")
                rows.append(row)
            
            graphs_div = html.Div(
                [html.H4("Comparison with 2000 ECG Dataset", className="text-white mb-3")] + rows,
                style={"marginTop": "20px"}
            )
            
            results_div = html.Div(
                [
                    neurokit_img_div,
                    html.H4("Fiducial & Amplitude Metrics", className="text-white"),
                    table1,
                    html.H4("HRV Metrics", className="text-white"),
                    table2,
                    graphs_div
                ],
                style={"marginTop": "20px"}
            )
            
            return f"File '{filename}' loaded successfully.", raw_fig, results_div

        except Exception as e:
            return f"Error processing the file: {str(e)}", go.Figure(), html.Div()
    else:
        return "Please upload an ECG file (.asc)", go.Figure(), html.Div()

if __name__ == '__main__':
    app.run_server(debug=True)
