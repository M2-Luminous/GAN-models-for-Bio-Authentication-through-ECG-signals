import base64
import io
import os
import numpy as np
import plotly.graph_objs as go
import dash
from dash import html, dcc, Input, Output, State
import dash_bootstrap_components as dbc
from tensorflow.keras.models import load_model, Model
from tqdm import tqdm

# -----------------------------
# Utility Functions
# -----------------------------
def min_max_normalize(data):
    """Normalize ECG data to [-1, 1]."""
    min_val = np.min(data)
    max_val = np.max(data)
    return 2 * (data - min_val) / (max_val - min_val) - 1

def load_lead_I_from_asc(file_path):
    """Load a .asc file and extract Lead I (assumes first column)."""
    try:
        ecg_data = np.loadtxt(file_path)
        return ecg_data[:, 0] if ecg_data.ndim > 1 else ecg_data
    except Exception as e:
        print(f"Error loading {file_path}: {e}")
        return None

def load_exactly_2000_lead_I_from_directory(data_dir, n=2000):
    """Load exactly n ECG files from the directory (using a progress bar)."""
    all_lead_I = []
    i = 0
    with tqdm(total=n, desc=f"Loading from {os.path.basename(data_dir)}", unit='file') as pbar:
        while len(all_lead_I) < n:
            file_name = f"{i}.asc"
            file_path = os.path.join(data_dir, file_name)
            if os.path.exists(file_path):
                lead_I = load_lead_I_from_asc(file_path)
                if lead_I is not None:
                    all_lead_I.append(lead_I)
                    pbar.update(1)
            i += 1
    return np.array(all_lead_I)

# -----------------------------
# Load Registered ECG Database and Compute Embeddings
# -----------------------------
DATA_DIR_REAL = "C:/Users/M2-Winterfell/Downloads/GAN-models-for-Bio-Authentication-through-ECG-signals/datasets/real_ecgs"

# Load 2000 real ECG records (one per registered user)
lead_I_real = load_exactly_2000_lead_I_from_directory(DATA_DIR_REAL, n=2000)
lead_I_real = np.array([min_max_normalize(ecg) for ecg in lead_I_real])

# Create dummy identity labels (0 to 1999)
num_registered_users = 2000
y_identity = np.arange(num_registered_users)

# For CNN input, reshape each ECG to (signal_length, 1)
signal_length = lead_I_real.shape[1]  # e.g., 5000
lead_I_real = lead_I_real.reshape(-1, signal_length, 1)

# -----------------------------
# Load the Identity Model and Build the Embedding Model
# -----------------------------
# (Assuming your identity model was saved with an extra classification head)
loaded_model = load_model("C:/Users/M2-Winterfell/Downloads/my_ecg_identity_model.h5")
# Create an embedding model that outputs the 128-dim normalized feature vector.
# (Assuming that the second-to-last layer is the L2 normalization layer)
embedding_model = Model(inputs=loaded_model.input, outputs=loaded_model.layers[-2].output)

# Pre-compute embeddings for all registered users (from the loaded training ECGs)
registered_embeddings = embedding_model.predict(lead_I_real)

def verify_ecg_signal(ecg_signal, registered_embeddings, threshold=0.8):
    """
    Compare the embedding of the uploaded ECG against registered embeddings.
    Returns a tuple: (authenticated (bool), best_similarity (float))
    """
    new_embedding = embedding_model.predict(ecg_signal)[0]
    # Since embeddings are L2-normalized, dot product equals cosine similarity.
    similarities = np.dot(registered_embeddings, new_embedding)
    best_similarity = np.max(similarities)
    authenticated = best_similarity > threshold
    return authenticated, best_similarity

# -----------------------------
# Dash App Setup and Layout
# -----------------------------
app = dash.Dash(__name__, external_stylesheets=[dbc.themes.CYBORG])
server = app.server

# Header for Second Level Challenge
logo_header = dbc.Row(
    dbc.Col(
        html.H1(
            "Authentication System Level 2",
            className="text-center text-white mt-3",
            style={"fontWeight": "bold"}
        )
    ),
    justify="center"
)

# Login Card: Email/Password (for extra security) + Verify button
login_card = dbc.Card(
    [
        dbc.CardBody(
            [
                html.Label("Email *", className="text-white"),
                dcc.Input(id='email', type='email', placeholder='Enter your email', className="form-control mb-3"),
                html.Label("Password *", className="text-white"),
                dcc.Input(id='password', type='password', placeholder='Enter your password', className="form-control mb-2"),
                html.Div(
                    [html.A("Forgot password?", href="#", className="me-auto")],
                    className="mb-3 text-end"
                ),
                dbc.Button(
                    "Verify",
                    id='submit-button',
                    n_clicks=0,
                    color="warning",
                    className="w-100"
                ),
                html.Div(
                    ["Not registered yet? ", html.A("Create an account", href="#")],
                    className="mt-3 text-center"
                ),
            ]
        )
    ],
    className="mt-4"
)

# ECG Upload Card: Upload file and display result
ecg_card = dbc.Card(
    [
        dbc.CardHeader(
            html.H4("Identity Verification Check", className="text-white mb-0"),
            className="bg-dark"
        ),
        dbc.CardBody(
            [
                html.Div("Upload your ECG file in .asc format to prove your identity.", className="mb-3 text-white"),
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
                dcc.Graph(id='ecg-graph', style={"border": "1px solid #444"}),
                html.Div(id='prediction-result', style={'margin': '10px', 'fontWeight': 'bold', 'color': '#FFD700'})
            ]
        )
    ],
    className="mt-4"
)

# Layout for the Login Page
login_layout = html.Div(
    style={"minHeight": "100vh", "backgroundColor": "#2c2c2c", "padding": "20px"},
    children=[
        dbc.Container(
            [
                logo_header,
                dbc.Row([dbc.Col(login_card, md=4)], className="justify-content-center"),
                dbc.Row([dbc.Col(ecg_card, md=8)], className="justify-content-center")
            ],
            fluid=True
        )
    ]
)

# Success Page Layout
success_layout = html.Div(
    style={"minHeight": "100vh", "backgroundColor": "#2c2c2c"},
    children=[
        dbc.Container(
            [
                html.H2("You passed the second level challenge!", className="text-white mt-5"),
                html.P("Welcome to the next page of your application!", className="text-white")
            ],
            fluid=True,
            className="py-5"
        )
    ]
)

# Main app layout with URL routing
app.layout = html.Div([
    dcc.Location(id='url', refresh=False),
    html.Div(id='page-content')
])

# -----------------------------
# Callbacks
# -----------------------------
# Callback to choose which page to display based on URL.
@app.callback(Output('page-content', 'children'),
              Input('url', 'pathname'))
def display_page(pathname):
    if pathname == "/success":
        return success_layout
    else:
        return login_layout

# Callback to process the uploaded ECG file.
@app.callback(
    [
        Output('upload-status', 'children'),
        Output('ecg-graph', 'figure'),
        Output('prediction-result', 'children'),
        Output('submit-button', 'disabled')
    ],
    Input('upload-ecg', 'contents'),
    State('upload-ecg', 'filename')
)
def process_uploaded_file(contents, filename):
    """
    1. Decodes the uploaded .asc file.
    2. Reads, normalizes, and reshapes the ECG signal.
    3. Verifies the signal using the identity model (cosine similarity against registered embeddings).
    4. Returns a message, visualization, result text, and the status of the Verify button.
    """
    if contents is not None:
        content_type, content_string = contents.split(',')
        decoded = base64.b64decode(content_string)
        try:
            # Read the file into a numpy array.
            s = io.StringIO(decoded.decode('utf-8'))
            ecg_data = np.loadtxt(s)
            if ecg_data.ndim > 1:
                ecg_data = ecg_data[:, 0]
            ecg_data = min_max_normalize(ecg_data)
            # Ensure the signal has the expected length.
            if ecg_data.shape[0] != signal_length:
                return (f"File '{filename}' does not have the expected length of {signal_length} samples.",
                        go.Figure(), "", True)
            # Reshape to (1, signal_length, 1) for the model.
            ecg_input = ecg_data.reshape(1, signal_length, 1)
            
            # Verify the ECG signal against registered embeddings.
            authenticated, best_similarity = verify_ecg_signal(ecg_input, registered_embeddings, threshold=0.8)
            prediction_label = "Real" if authenticated else "Fake"
            result_text = f"Prediction: {prediction_label} (Cosine Similarity: {best_similarity:.2f})"
            
            # Create an ECG visualization.
            fig = go.Figure(data=go.Scatter(y=ecg_data, mode='lines', line=dict(color='orange')))
            fig.update_layout(
                title='ECG Signal',
                xaxis_title='Time',
                yaxis_title='Amplitude',
                template='plotly_dark',
                paper_bgcolor='#2c2c2c',
                plot_bgcolor='#2c2c2c',
                font=dict(color='white')
            )
            
            # Enable the Verify button only if the signal is authenticated.
            submit_disabled = not authenticated
            
            return (f"File '{filename}' loaded successfully.", fig, result_text, submit_disabled)
        except Exception as e:
            return (f"Error processing the file: {str(e)}", go.Figure(), "", True)
    else:
        return ("Please upload an ECG file (.asc)", go.Figure(), "", True)

# Callback for the Verify button to navigate to the success page.
@app.callback(
    Output('url', 'pathname'),
    Input('submit-button', 'n_clicks'),
    [State('email', 'value'),
     State('password', 'value')]
)
def login(n_clicks, email, password):
    if n_clicks > 0 and email and password:
        return "/success"
    return "/"

# -----------------------------
# Run Server
# -----------------------------
if __name__ == '__main__':
    app.run_server(debug=True, port=8051, use_reloader=False)
