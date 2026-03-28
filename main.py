import os
import sys
from trame.app import get_server
from trame.ui.vuetify import SinglePageLayout
from trame.widgets import vuetify, html

try:
    import processing
except ImportError as e:
    print(f"Warning: Could not import processing module: {e}")

# --- Workspace Setup ---
UPLOAD_DIR = 'data/uploads'
OUTPUT_DIR = 'data/outputs'
os.makedirs(UPLOAD_DIR, exist_ok=True)
os.makedirs(OUTPUT_DIR, exist_ok=True)

# --- Server ---
server = get_server(client_type="vue2")
state, ctrl = server.state, server.controller

# --- Backend ---
class AppBackend:
    def __init__(self):
        self.cube = None
        self.color_mapper = None
        self.shape = (100, 100, 100)

backend = AppBackend()

# --- State ---
state.active_tab = 0
state.img_src = ""

state.ml_processing = False
state.ml_status_msg = "Awaiting File..."
state.ml_result_ready = False
state.ml_output_path = ""

state.viewer_processing = False
state.viewer_status_msg = "Ready"

state.iline_check = False
state.iline_val = 0
state.iline_max = 100

state.xline_check = False
state.xline_val = 0
state.xline_max = 100

state.time_check = False
state.time_val = 0
state.time_max = 100

# --- Handlers ---
def handle_ml_upload(file_path):
    if not file_path:
        return
    state.ml_processing = True
    state.ml_result_ready = False
    state.ml_status_msg = "Processing file..."
    
    try:
        filename = os.path.basename(file_path)
        output_path = os.path.join(OUTPUT_DIR, f"faults_{filename}")
        state.ml_status_msg = "Process Finished. Success!"
        state.ml_output_path = output_path
        state.ml_result_ready = True
    except Exception as err:
        state.ml_status_msg = f"Error: {str(err)}"
    finally:
        state.ml_processing = False

def handle_viewer_upload(file_path):
    if not file_path:
        return
    
    state.viewer_processing = True
    state.viewer_status_msg = "Processing file..."
    
    try:
        backend.shape = (100, 100, 100)

        state.iline_max = 99
        state.xline_max = 99
        state.time_max = 99

        state.iline_val = 50
        state.xline_val = 50
        state.time_val = 50

        state.viewer_status_msg = f"Ready: {os.path.basename(file_path)}"
    except Exception as err:
        state.viewer_status_msg = f"Viewer Error: {str(err)}"
    finally:
        state.viewer_processing = False


# --- UI ---
with SinglePageLayout(server) as layout:
    layout.title.set_text("Seismic Portal Pro")

    # Content
    with layout.content:

        # ✅ CSS overrides for absolute text visibility
        html.Style("""
        @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&display=swap');

        body { 
            background: radial-gradient(circle at 50% 0%, #1a2035, #0b0f19 80%);
            color: #e2e8f0;
            font-family: 'Inter', sans-serif;
            min-height: 100vh;
        }

        .v-application {
            background: transparent !important;
        }
        .v-application--wrap {
            min-height: 100vh;
        }

        /* Top Navigation Bar */
        .glass-nav {
            background: rgba(11, 15, 25, 0.7) !important;
            backdrop-filter: blur(12px);
            border-bottom: 1px solid rgba(255, 255, 255, 0.08) !important;
        }

        .text-gradient {
            background: linear-gradient(90deg, #00d2ff 0%, #3a7bd5 100%);
            -webkit-background-clip: text;
            -webkit-text-fill-color: transparent;
            font-weight: 700;
            letter-spacing: 0.5px;
        }

        /* Modern Cards */
        .card-modern {
            border-radius: 16px;
            background: linear-gradient(145deg, rgba(22, 27, 34, 0.9), rgba(16, 20, 26, 0.9)) !important;
            border: 1px solid rgba(255, 255, 255, 0.06);
            box-shadow: 0 8px 32px rgba(0, 0, 0, 0.4) !important;
            backdrop-filter: blur(8px);
        }

        .section-title {
            font-weight: 600;
            font-size: 0.95rem;
            color: #8b9bb4 !important; 
            text-transform: uppercase;
            letter-spacing: 1.5px;
            margin-bottom: 16px;
            display: flex;
            align-items: center;
        }
        
        .section-title::after {
            content: "";
            flex: 1;
            height: 1px;
            background: linear-gradient(90deg, rgba(139, 155, 180, 0.3), transparent);
            margin-left: 12px;
        }

        /* Interactive Elements */
        .glow-btn {
            background: linear-gradient(90deg, #00d2ff 0%, #3a7bd5 100%) !important;
            border: none !important;
            color: #fff !important;
            font-weight: 600 !important;
            letter-spacing: 0.5px;
            transition: all 0.3s cubic-bezier(0.25, 0.8, 0.25, 1);
            box-shadow: 0 4px 15px rgba(0, 210, 255, 0.3) !important;
        }

        .glow-btn:hover {
            box-shadow: 0 6px 20px rgba(0, 210, 255, 0.5) !important;
            transform: translateY(-2px);
        }

        .upload-box {
            border: 2px dashed rgba(88, 166, 255, 0.3);
            border-radius: 16px;
            padding: 30px;
            background: rgba(13, 17, 23, 0.5);
            transition: all 0.3s ease;
            cursor: pointer;
        }

        .upload-box:hover {
            border-color: #58a6ff;
            background: rgba(88, 166, 255, 0.05);
            transform: scale(1.01);
        }

        /* 3D Viewer Area */
        .viewer-box {
            border-radius: 16px;
            background: #05070a !important;
            border: 1px solid rgba(255, 255, 255, 0.05);
            box-shadow: inset 0 0 40px rgba(0, 0, 0, 0.8);
            position: relative;
            overflow: hidden;
        }

        .viewer-empty-bg {
            position: absolute;
            inset: 0;
            background-image: 
                radial-gradient(rgba(255,255,255,0.07) 1px, transparent 1px),
                radial-gradient(rgba(255,255,255,0.03) 1px, transparent 1px);
            background-size: 30px 30px, 150px 150px;
            background-position: 0 0, 15px 15px;
            pointer-events: none;
        }

        /* Animations */
        .pulse-icon {
            animation: pulse-glow 2.5s infinite alternate;
        }
        @keyframes pulse-glow {
            0% { transform: scale(0.95); filter: drop-shadow(0 0 10px rgba(88, 166, 255, 0.2)); }
            100% { transform: scale(1.05); filter: drop-shadow(0 0 25px rgba(88, 166, 255, 0.6)); }
        }

        /* 🚨 FIX: Force Tab text colors so they are always visible */
        .v-tabs-bar { border-bottom: 1px solid rgba(255,255,255,0.05); }
        .v-tab { 
            text-transform: none !important; 
            font-weight: 600 !important; 
            letter-spacing: 0.5px; 
            color: #8b9bb4 !important; /* Visible Inactive color */
        }
        .v-tab--active {
            color: #00d2ff !important; /* Bright Active Color */
        }
        .v-input__slider { margin-top: 0px; }
        """)

        # Custom Toolbar Appended to Content
        with vuetify.VAppBar(app=True, flat=True, classes="glass-nav", dark=True):
            vuetify.VIcon("mdi-layers-triple", classes="mr-3", color="#00d2ff", size=28)
            vuetify.VToolbarTitle("Seismic Portal", classes="text-gradient text-h5")
            vuetify.VSpacer()

        with vuetify.VContainer(fluid=True, classes="pa-6 mt-12"):

            # Tabs
            with vuetify.VTabs(
                v_model=("active_tab",),
                background_color="transparent",
                slider_color="#00d2ff", # Added explicit slider color
                centered=True,
                classes="mb-6"
            ):
                with vuetify.VTab():
                    vuetify.VIcon("mdi-brain", classes="mr-2", size=18)
                    html.Span("ML Processing")
                with vuetify.VTab():
                    vuetify.VIcon("mdi-cube-outline", classes="mr-2", size=18)
                    html.Span("3D Workspace")
                with vuetify.VTab():
                    vuetify.VIcon("mdi-chart-timeline-variant", classes="mr-2", size=18)
                    html.Span("Analysis Engine")

            with vuetify.VTabsItems(v_model=("active_tab",), style="background-color: transparent;"):

                # --- ML TAB ---
                with vuetify.VTabItem():
                    with vuetify.VRow(justify="center", classes="mt-6"):
                        with vuetify.VCol(cols="12", sm="10", md="7", lg="6"):
                            with vuetify.VCard(classes="pa-8 card-modern", elevation=0, dark=True):

                                html.Div("Deep Learning Fault Extraction", classes="section-title")
                                html.P("Upload a SEGY file to pass through the neural network for automatic fault detection.", 
                                       classes="text-body-2 grey--text text--lighten-1 mb-6")

                                with html.Div(classes="upload-box text-center"):
                                    vuetify.VIcon("mdi-cloud-upload-outline", size=48, color="#58a6ff", classes="mb-3")
                                    vuetify.VFileInput(
                                        label="Browse or drop SEGY file here",
                                        accept=".segy,.sgy",
                                        outlined=True,
                                        dense=True,
                                        hide_details=True,
                                        color="#00d2ff",
                                        prepend_icon="", 
                                        dark=True
                                    )

                                with vuetify.VRow(v_if=("ml_processing",), justify="center", classes="mt-8 mb-4"):
                                    with vuetify.VCol(classes="text-center"):
                                        vuetify.VProgressCircular(indeterminate=True, color="#00d2ff", size=50, width=4)
                                        html.Div("{{ ml_status_msg }}", classes="mt-3 text-subtitle-2 cyan--text")

                                with vuetify.VRow(v_if=("ml_result_ready",), justify="center", classes="mt-8"):
                                    vuetify.VBtn(
                                        "Download Extracted Faults",
                                        prepend_icon="mdi-download",
                                        class_="glow-btn",
                                        rounded=True,
                                        large=True
                                    )

                # --- VIEWER TAB ---
                with vuetify.VTabItem():
                    with vuetify.VRow(classes="mt-2", spacing=4):

                        # Left panel - Unified Control Center
                        with vuetify.VCol(cols="12", md="4", lg="3"):
                            with vuetify.VCard(classes="pa-6 card-modern", elevation=0, style="height: 100%; min-height: 600px;", dark=True):
                                
                                # Section 1: Data
                                html.Div("Data Source", classes="section-title")
                                vuetify.VFileInput(
                                    label="Load SEGY Volume",
                                    dense=True,
                                    outlined=True,
                                    hide_details=True,
                                    prepend_inner_icon="mdi-database",
                                    prepend_icon="",
                                    color="#00d2ff",
                                    classes="mb-2",
                                    dark=True
                                )
                                html.Div("{{ viewer_status_msg }}", classes="text-caption cyan--text mb-6")

                                vuetify.VDivider(dark=True, classes="mb-6", style="border-color: rgba(255,255,255,0.05);")

                                # Section 2: Controls
                                html.Div("Slicing Controls", classes="section-title")

                                # Inline
                                with vuetify.VRow(align="center", classes="mb-0"):
                                    with vuetify.VCol(cols="auto"):
                                        vuetify.VSwitch(v_model=("iline_check",), color="#00d2ff", hide_details=True, inset=True, dense=True, dark=True)
                                    with vuetify.VCol():
                                        html.Span("Inline", classes="text-body-2 white--text")
                                vuetify.VSlider(v_if=("iline_check",), v_model=("iline_val",), min=0, max=("iline_max",), 
                                                color="#00d2ff", track_color="grey darken-3", thumb_label=True, classes="mt-1 mb-4", dark=True)

                                # Crossline
                                with vuetify.VRow(align="center", classes="mb-0"):
                                    with vuetify.VCol(cols="auto"):
                                        vuetify.VSwitch(v_model=("xline_check",), color="#ff9800", hide_details=True, inset=True, dense=True, dark=True)
                                    with vuetify.VCol():
                                        html.Span("Crossline", classes="text-body-2 white--text")
                                vuetify.VSlider(v_if=("xline_check",), v_model=("xline_val",), min=0, max=("xline_max",), 
                                                color="#ff9800", track_color="grey darken-3", thumb_label=True, classes="mt-1 mb-4", dark=True)

                                # Time Slice
                                with vuetify.VRow(align="center", classes="mb-0"):
                                    with vuetify.VCol(cols="auto"):
                                        vuetify.VSwitch(v_model=("time_check",), color="#b388ff", hide_details=True, inset=True, dense=True, dark=True)
                                    with vuetify.VCol():
                                        html.Span("Time Slice", classes="text-body-2 white--text")
                                vuetify.VSlider(v_if=("time_check",), v_model=("time_val",), min=0, max=("time_max",), 
                                                color="#b388ff", track_color="grey darken-3", thumb_label=True, classes="mt-1", dark=True)

                        # Right panel - Viewer
                        with vuetify.VCol(cols="12", md="8", lg="9"):
                            with vuetify.VCard(
                                classes="viewer-box d-flex align-center justify-center",
                                style="height: 100%; min-height: 600px;",
                                dark=True
                            ):
                                html.Div(classes="viewer-empty-bg")
                                
                                html.Img(v_if=("img_src",),
                                         src=("img_src",),
                                         style="max-width:100%; max-height:100%; border-radius: 8px; z-index: 1; position: relative;")

                                with html.Div(v_if=("!img_src",), classes="text-center", style="z-index: 1;"):
                                    vuetify.VIcon("mdi-axis-arrow", size=72, color="rgba(255,255,255,0.1)", classes="mb-4 pulse-icon")
                                    html.H4("3D Workspace Empty", classes="text-h6 grey--text text--lighten-1 font-weight-regular")
                                    html.P("Load a dataset from the left panel to begin visualization.", classes="text-body-2 grey--text text--darken-1")

                # --- ANALYSIS TAB ---
                with vuetify.VTabItem():
                    with vuetify.VRow(justify="center", align="center", style="height: 600px;"):
                        with vuetify.VCol(classes="text-center"):
                            vuetify.VIcon("mdi-hexagram-outline", size=100, color="rgba(0, 210, 255, 0.4)", classes="pulse-icon")
                            html.H3("ANALYSIS ENGINE", classes="text-h4 mt-6 text-gradient", style="letter-spacing: 3px;")
                            html.P("Advanced morphometric and curvature attributes are currently under development.",
                                   classes="text-body-1 grey--text text--lighten-1 mt-3")


# --- Run ---
if __name__ == "__main__":
    print("Starting Seismic Portal Pro on http://localhost:8081")
    server.start(port=8081)