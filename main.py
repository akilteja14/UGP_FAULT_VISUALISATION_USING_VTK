import os
import sys
import threading
import uuid
import asyncio
from urllib.parse import quote
from aiohttp import web
from trame.app import get_server
from trame.ui.vuetify import SinglePageLayout
from trame.widgets import vuetify, html

processing = None
try:
    import processing
except ImportError as e:
    print(f"Warning: Could not import processing module: {e}")

# --- Workspace Setup ---
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.abspath(os.path.join(BASE_DIR, ".."))
UPLOAD_DIR = os.path.join(PROJECT_ROOT, "data", "uploads")
OUTPUT_DIR = os.path.join(PROJECT_ROOT, "data", "outputs")
os.makedirs(UPLOAD_DIR, exist_ok=True)
os.makedirs(OUTPUT_DIR, exist_ok=True)

# --- Server ---
server = get_server(client_type="vue2")
state, ctrl = server.state, server.controller
server.serve["outputs"] = OUTPUT_DIR
server.serve["mltool"] = BASE_DIR

# --- Backend ---
class AppBackend:
    def __init__(self):
        self.cube = None
        self.color_mapper = None
        self.shape = (100, 100, 100)

backend = AppBackend()

ml_job_status = {
    "stage": "idle",
    "message": "Awaiting File...",
    "percent": 0,
    "running": False,
    "completed_patches": 0,
    "total_patches": 0,
    "elapsed_seconds": 0,
    "eta_seconds": 0,
    "output_path": "",
}

# --- State ---
state.active_tab = 0
state.img_src = ""

state.ml_processing = False
state.ml_uploading = False
state.ml_status_msg = "Awaiting File..."
state.ml_result_ready = False
state.ml_output_path = ""
state.ml_input_path = ""
state.ml_download_url = ""
state.ml_selected_name = ""
state.ml_ready_to_run = False

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
def handle_ml_upload(file_path=None):
    input_path = (file_path or state.ml_input_path or "").strip()
    if not input_path:
        state.ml_status_msg = "Browse and upload a SEG-Y file first."
        return

    input_path = os.path.abspath(input_path)
    if not os.path.exists(input_path):
        state.ml_status_msg = f"File not found: {input_path}"
        return

    if processing is None:
        state.ml_status_msg = "Processing module is unavailable."
        return

    def run_job():
        with state:
            state.ml_processing = True
            state.ml_ready_to_run = False
            state.ml_result_ready = False
            state.ml_input_path = input_path
            state.ml_output_path = ""
            state.ml_download_url = ""
            state.ml_status_msg = "Loading model and processing file..."

        try:
            output_path = processing.build_ml_output_path(input_path, OUTPUT_DIR)
            processing.run_ml_extraction(input_path, output_path)
            download_name = quote(os.path.basename(output_path))
            with state:
                state.ml_status_msg = "Process Finished. Success!"
                state.ml_output_path = output_path
                state.ml_download_url = f"/outputs/{download_name}"
                state.ml_result_ready = True
                state.ml_ready_to_run = True
        except Exception as err:
            with state:
                state.ml_status_msg = f"Error: {str(err)}"
                state.ml_ready_to_run = True
        finally:
            with state:
                state.ml_processing = False

    threading.Thread(target=run_job, daemon=True).start()


def handle_upload_started(filename):
    with state:
        state.ml_uploading = True
        state.ml_selected_name = filename or ""
        state.ml_ready_to_run = False
        state.ml_result_ready = False
        state.ml_output_path = ""
        state.ml_download_url = ""
        state.ml_status_msg = f"Uploading: {state.ml_selected_name}" if state.ml_selected_name else "Uploading file..."


def handle_uploaded_ml_file(file_path):
    if not file_path:
        state.ml_status_msg = "Upload failed. No file path returned."
        return

    with state:
        state.ml_uploading = False
        state.ml_input_path = file_path
        state.ml_selected_name = os.path.basename(file_path)
        state.ml_ready_to_run = True
        state.ml_result_ready = False
        state.ml_output_path = ""
        state.ml_download_url = ""
        state.ml_status_msg = f"Upload complete: {state.ml_selected_name}"


def handle_upload_failed(message):
    with state:
        state.ml_uploading = False
        state.ml_input_path = ""
        state.ml_selected_name = ""
        state.ml_ready_to_run = False
        state.ml_status_msg = f"Upload failed: {message or 'Unknown error'}"


async def upload_ml_file(request):
    reader = await request.multipart()
    field = await reader.next()

    if field is None or field.name != "file":
        return web.json_response({"error": "No file uploaded."}, status=400)

    original_name = os.path.basename(field.filename or "upload.segy")
    ext = os.path.splitext(original_name)[1].lower()
    if ext not in [".segy", ".sgy"]:
        return web.json_response({"error": "Only .segy and .sgy files are supported."}, status=400)

    stem = os.path.splitext(original_name)[0]
    saved_name = f"{stem}_{uuid.uuid4().hex[:8]}{ext}"
    saved_path = os.path.join(UPLOAD_DIR, saved_name)

    with open(saved_path, "wb") as output_file:
        while True:
            chunk = await field.read_chunk()
            if not chunk:
                break
            output_file.write(chunk)

    return web.json_response({"saved_path": saved_path, "filename": saved_name})


async def ml_status(request):
    return web.json_response(ml_job_status)


def bind_server_routes(wslink_server):
    wslink_server.app.router.add_post("/upload_ml", upload_ml_file)
    wslink_server.app.router.add_post("/run_ml", run_ml_file)
    wslink_server.app.router.add_get("/ml_status", ml_status)


ctrl.on_server_bind.add(bind_server_routes)


async def run_ml_file(request):
    try:
        payload = await request.json()
    except Exception:
        return web.json_response({"error": "Invalid request payload."}, status=400)

    input_path = os.path.abspath((payload.get("input_path") or "").strip())
    if not input_path:
        return web.json_response({"error": "Missing input file path."}, status=400)

    if not os.path.exists(input_path):
        return web.json_response({"error": "Uploaded file was not found on the server."}, status=404)

    if processing is None:
        return web.json_response({"error": "Processing module is unavailable."}, status=500)

    output_path = processing.build_ml_output_path(input_path, OUTPUT_DIR)

    ml_job_status.update(
        {
            "stage": "queued",
            "message": "Queued for processing...",
            "percent": 0,
            "running": True,
            "completed_patches": 0,
            "total_patches": 0,
            "elapsed_seconds": 0,
            "eta_seconds": 0,
            "output_path": "",
        }
    )

    def progress_callback(update):
        ml_job_status.update(update)
        ml_job_status["running"] = update.get("stage") != "complete"

    try:
        await asyncio.to_thread(processing.run_ml_extraction, input_path, output_path, progress_callback)
    except Exception as err:
        ml_job_status.update(
            {
                "stage": "error",
                "message": str(err),
                "running": False,
            }
        )
        return web.json_response({"error": str(err)}, status=500)

    ml_job_status.update(
        {
            "stage": "complete",
            "message": "Process Finished. Success!",
            "percent": 100,
            "running": False,
            "output_path": output_path,
        }
    )
    return web.json_response(
        {
            "output_path": output_path,
            "download_url": f"/outputs/{quote(os.path.basename(output_path))}",
        }
    )

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

        .ml-panel {
            border-radius: 16px;
            background: linear-gradient(145deg, rgba(22, 27, 34, 0.9), rgba(16, 20, 26, 0.9));
            border: 1px solid rgba(255, 255, 255, 0.06);
            box-shadow: 0 8px 32px rgba(0, 0, 0, 0.4);
            backdrop-filter: blur(8px);
            padding: 32px;
        }

        .ml-native-input {
            width: 100%;
            color: #e2e8f0;
            background: rgba(255,255,255,0.03);
            border: 1px solid rgba(255,255,255,0.12);
            border-radius: 10px;
            padding: 12px;
        }

        .ml-native-btn {
            margin-top: 16px;
            padding: 12px 20px;
            border-radius: 999px;
            border: none;
            background: linear-gradient(90deg, #00d2ff 0%, #3a7bd5 100%);
            color: white;
            font-weight: 600;
            cursor: pointer;
        }

        .ml-native-btn:disabled {
            opacity: 0.5;
            cursor: not-allowed;
        }

        .ml-download-link {
            display: inline-block;
            margin-top: 16px;
            padding: 12px 20px;
            border-radius: 999px;
            text-decoration: none;
            background: linear-gradient(90deg, #00d2ff 0%, #3a7bd5 100%);
            color: white;
            font-weight: 600;
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
        html.Script("""
        window.__bindMlUploadUi = function() {
            const input = document.getElementById('ml-upload-input');
            const runBtn = document.getElementById('ml-run-btn');
            const statusEl = document.getElementById('ml-status');
            const selectedEl = document.getElementById('ml-selected-name');
            const downloadWrap = document.getElementById('ml-download-wrap');
            const downloadLink = document.getElementById('ml-download-link');
            const outputEl = document.getElementById('ml-output-path');

            if (!input || !runBtn || !statusEl || !selectedEl || !downloadWrap || !downloadLink || !outputEl) return;

            window.mlUploadedPath = window.mlUploadedPath || '';

            input.onchange = async function(event) {
                const files = event.target && event.target.files ? event.target.files : null;
                if (!files || !files.length) return;
                const file = files[0];

                selectedEl.textContent = file.name;
                statusEl.textContent = `Uploading: ${file.name} (0%)`;
                runBtn.disabled = true;
                downloadWrap.style.display = 'none';
                outputEl.textContent = '';

                const formData = new FormData();
                formData.append('file', file);

                try {
                    const payload = await new Promise((resolve, reject) => {
                        const xhr = new XMLHttpRequest();
                        xhr.open('POST', `${window.location.origin}/upload_ml`);
                        xhr.responseType = 'json';

                        xhr.upload.addEventListener('progress', function(progressEvent) {
                            if (progressEvent.lengthComputable) {
                                const percent = Math.round((progressEvent.loaded / progressEvent.total) * 100);
                                statusEl.textContent = `Uploading: ${file.name} (${percent}%)`;
                            } else {
                                const uploadedMb = (progressEvent.loaded / (1024 * 1024)).toFixed(1);
                                statusEl.textContent = `Uploading: ${file.name} (${uploadedMb} MB sent)`;
                            }
                        });

                        xhr.addEventListener('load', function() {
                            const payload = xhr.response || {};
                            if (xhr.status < 200 || xhr.status >= 300) {
                                reject(new Error(payload.error || `HTTP ${xhr.status}`));
                                return;
                            }
                            resolve(payload);
                        });

                        xhr.addEventListener('error', function() {
                            reject(new Error('Network error during upload'));
                        });

                        xhr.send(formData);
                    });

                    window.mlUploadedPath = payload.saved_path;
                    statusEl.textContent = `Upload complete: ${payload.filename}`;
                    runBtn.disabled = false;
                } catch (error) {
                    console.error(error);
                    statusEl.textContent = `Upload failed: ${error && error.message ? error.message : 'Unknown error'}`;
                    window.mlUploadedPath = '';
                } finally {
                    event.target.value = '';
                }
            };

            runBtn.onclick = async function() {
                if (!window.mlUploadedPath) {
                    statusEl.textContent = 'Browse and upload a SEG-Y file first.';
                    return;
                }

                runBtn.disabled = true;
                downloadWrap.style.display = 'none';
                outputEl.textContent = '';
                statusEl.textContent = 'Running fault extraction...';

                try {
                    const response = await fetch(`${window.location.origin}/run_ml`, {
                        method: 'POST',
                        headers: { 'Content-Type': 'application/json' },
                        body: JSON.stringify({ input_path: window.mlUploadedPath })
                    });
                    const payload = await response.json();
                    if (!response.ok) {
                        statusEl.textContent = `Run failed: ${payload.error || 'Unknown error'}`;
                        runBtn.disabled = false;
                        return;
                    }

                    statusEl.textContent = 'Process Finished. Success!';
                    outputEl.textContent = payload.output_path;
                    downloadLink.href = payload.download_url;
                    downloadWrap.style.display = 'block';
                    runBtn.disabled = false;
                } catch (error) {
                    console.error(error);
                    statusEl.textContent = `Run failed: ${error && error.message ? error.message : 'Unknown error'}`;
                    runBtn.disabled = false;
                }
            };
        };

        window.__bindMlUploadUi();
        setInterval(window.__bindMlUploadUi, 1000);
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
                        with vuetify.VCol(cols="12", sm="11", md="9", lg="8"):
                            html.Iframe(
                                src="/mltool/ml_tool.html",
                                style="width: 100%; height: 620px; border: 0; border-radius: 16px; background: transparent;",
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
