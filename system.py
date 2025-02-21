import base64
import io
import numpy as np
import plotly.graph_objs as go
import dash
from dash import html, dcc, Input, Output, State
import dash_bootstrap_components as dbc
from tensorflow.keras.models import load_model

# --------------------
# Load your trained model
# --------------------
loaded_model = load_model("my_ecg_model.h5")
print("Model loaded successfully.")

# --------------------
# Define normalization function
# --------------------
def min_max_normalize(data):
    min_val = np.min(data)
    max_val = np.max(data)
    return 2 * (data - min_val) / (max_val - min_val) - 1

# --------------------
# Dash App Setup
# --------------------
app = dash.Dash(__name__, external_stylesheets=[dbc.themes.CYBORG])
server = app.server

# --------------------
# Layout Components
# --------------------
# Create a logo placeholder for "BLS INTERNATIONAL" 
# (replace the text with an actual image if you have a logo file)
logo_header = dbc.Row(
    [
        dbc.Col(
            html.H1(
                "Authentication System Level 1", 
                className="text-center text-white mt-3",
                style={"fontWeight": "bold"}
            )
        )
    ],
    justify="center"
)

# Login card: email/password, forgot password, verify button
login_card = dbc.Card(
    [
        dbc.CardBody(
            [
                html.Label("Email *", className="text-white"),
                dcc.Input(
                    id='email', 
                    type='email', 
                    placeholder='Enter your email', 
                    className="form-control mb-3"
                ),
                
                html.Label("Password *", className="text-white"),
                dcc.Input(
                    id='password', 
                    type='password', 
                    placeholder='Enter your password', 
                    className="form-control mb-2"
                ),
                
                html.Div(
                    [
                        html.A("Forgot password?", href="#", className="me-auto"),
                    ],
                    className="mb-3 text-end"
                ),

                dbc.Button(
                    "Verify",
                    id='submit-button',
                    n_clicks=0,
                    color="warning",  # matches the gold/amber look
                    className="w-100"
                ),
                
                html.Div(
                    [
                        "Not registered yet? ",
                        html.A("Create an account", href="#")
                    ],
                    className="mt-3 text-center"
                ),
            ]
        )
    ],
    className="mt-4"
)

# ECG upload and visualization card
ecg_card = dbc.Card(
    [
        dbc.CardHeader(
            html.H4("Bio-authentication Check", className="text-white mb-0"),
            className="bg-dark"
        ),
        dbc.CardBody(
            [
                html.Div(
                    "Please upload your ECG file in .asc format to prove you're human.",
                    className="mb-3 text-white"
                ),
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
                
                html.Div(
                    id='prediction-result',
                    style={'margin': '10px', 'fontWeight': 'bold', 'color': '#FFD700'}
                )
            ]
        )
    ],
    className="mt-4"
)

# Define the login (main) page layout
login_layout = html.Div(
    style={
        "minHeight": "100vh",
        "backgroundColor": "#2c2c2c",
        "padding": "20px"
    },
    children=[
        dbc.Container(
            [
                logo_header,
                dbc.Row(
                    [
                        dbc.Col(
                            login_card,
                            md=4
                        )
                    ],
                    className="justify-content-center"
                ),
                dbc.Row(
                    [
                        dbc.Col(
                            ecg_card,
                            md=8
                        )
                    ],
                    className="justify-content-center"
                )
            ],
            fluid=True
        )
    ]
)

# Define the success page layout
success_layout = html.Div(
    style={"minHeight": "100vh", "backgroundColor": "#2c2c2c"},
    children=[
        dbc.Container(
            [
                html.H2("You pass the test", className="text-white mt-5"),
                html.P(
                    "Welcome to the next page of your application!",
                    className="text-white"
                )
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

# --------------------
# Callbacks
# --------------------
# Callback to display the appropriate page based on URL
@app.callback(Output('page-content', 'children'),
              Input('url', 'pathname'))
def display_page(pathname):
    if pathname == "/success":
        return success_layout
    else:
        return login_layout

# Callback to process the uploaded ECG file and update the graph and prediction result
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
    Handles the uploaded .asc file:
    1. Reads it into a numpy array.
    2. Normalizes data.
    3. Reshapes it to (1, 5000, 1) for the model.
    4. Gets the model prediction and updates the graph + result text.
    5. Enables the Verify button only if the ECG is predicted "Real".
    """
    if contents is not None:
        content_type, content_string = contents.split(',')
        decoded = base64.b64decode(content_string)
        try:
            # Read the file into a numpy array
            s = io.StringIO(decoded.decode('utf-8'))
            ecg_data = np.loadtxt(s)
            
            # If multiple columns, extract Lead I (first column)
            if ecg_data.ndim > 1:
                ecg_data = ecg_data[:, 0]
            
            # Normalize and reshape for model input
            ecg_data = min_max_normalize(ecg_data)
            ecg_input = ecg_data.reshape(1, 5000, 1)
            
            # Model prediction
            prediction_prob = loaded_model.predict(ecg_input)
            prediction_label = "Real" if prediction_prob[0][0] > 0.5 else "Fake"
            result_text = f"Prediction: {prediction_label} (Probability: {prediction_prob[0][0]:.2f})"
            
            # Create ECG visualization
            fig = go.Figure(data=go.Scatter(y=ecg_data, mode='lines', line=dict(color='orange')))
            fig.update_layout(
                title='ECG Signal',
                xaxis_title='Time',
                yaxis_title='Amplitude',
                template='plotly_dark',   # keeps it consistent with dark theme
                paper_bgcolor='#2c2c2c',
                plot_bgcolor='#2c2c2c',
                font=dict(color='white')
            )
            
            # Enable "Verify" button only if the ECG is predicted as "Real"
            submit_disabled = (prediction_label != "Real")
            
            return (
                f"File '{filename}' loaded successfully.",
                fig,
                result_text,
                submit_disabled
            )
        except Exception as e:
            return (
                f"Error processing the file: {str(e)}",
                go.Figure(),
                "",
                True
            )
    else:
        return (
            "Please upload an ECG file (.asc)",
            go.Figure(),
            "",
            True
        )

# Callback for the "Verify" (Submit) button to navigate to the success page
@app.callback(
    Output('url', 'pathname'),
    Input('submit-button', 'n_clicks'),
    [State('email', 'value'),
     State('password', 'value')]
)
def login(n_clicks, email, password):
    """
    Checks if the user has clicked "Verify" and provided some email/password.
    If so, navigate to /success. Otherwise, remain on the login page.
    """
    if n_clicks > 0:
        # Additional email/password authentication logic can be added here.
        if email and password:
            return "/success"
    return "/"

# --------------------
# Run Server
# --------------------
if __name__ == '__main__':
    app.run_server(debug=True)
