import os
import sys
import threading
import uuid
import asyncio
from urllib.parse import quote
import numpy as np
import requests
from aiohttp import web
from trame.app import get_server
from trame.ui.vuetify import SinglePageLayout
from trame.widgets import vuetify, html
from trame.widgets import vtk as vtk_widgets
import vtk

from VTK_PY.segy_viewer import build_vtk_image, build_mapper, create_actors

# --- Workspace Setup ---
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
# Default to localhost if backend is on the same machine, or override via environment variable
BACKEND_URL = os.environ.get("BACKEND_URL", "http://localhost:5000")

# --- Server ---
server = get_server(client_type="vue2")
state, ctrl = server.state, server.controller
server.serve["mltool"] = BASE_DIR

# --- Backend ---
class AppBackend:
    def __init__(self):
        self.cube = None
        self.color_mapper = None
        self.shape = (100, 100, 100)
        self.viewer_ready = False
        self.loaded_path = ""

        self.renderer = vtk.vtkRenderer()
        self.window = vtk.vtkRenderWindow()
        self.window.SetOffScreenRendering(1)
        self.window.AddRenderer(self.renderer)

        self.interactor = vtk.vtkRenderWindowInteractor()
        self.interactor.SetRenderWindow(self.window)
        self.interactor.SetInteractorStyle(vtk.vtkInteractorStyleTrackballCamera())

backend = AppBackend()

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
state.viewer_status_msg = "Waiting for slice server"
state.viewer_loaded_name = ""

state.iline_check = False
state.iline_val = 0
state.iline_max = 100

state.xline_check = False
state.xline_val = 0
state.xline_max = 100

state.time_check = False
state.time_val = 0
state.time_max = 100

# --- proxy API functions ---
def configure_slice_server(file_path):
    response = requests.post(
        f"{BACKEND_URL}/load",
        json={"path": file_path},
        timeout=120,
    )
    response.raise_for_status()
    return response.json()


def fetch_slice(inline_idx, crossline_idx, time_idx, mode):
    response = requests.get(
        f"{BACKEND_URL}/slice",
        params={
            "inline": inline_idx,
            "crossline": crossline_idx,
            "time": time_idx,
            "mode": mode,
        },
        timeout=30,
    )
    response.raise_for_status()
    return np.array(response.json()["data"], dtype=np.float32)


def build_sparse_remote_cube():
    volume = np.zeros(backend.shape, dtype=np.float32)
    fetched_slices = []

    if state.iline_check:
        inline_slice = fetch_slice(state.iline_val, 0, 0, "inline")
        volume[state.iline_val, :, :] = inline_slice
        fetched_slices.append(inline_slice)

    if state.xline_check:
        cross_slice = fetch_slice(0, state.xline_val, 0, "crossline")
        volume[:, state.xline_val, :] = cross_slice
        fetched_slices.append(cross_slice)

    if state.time_check:
        time_slice = fetch_slice(0, 0, state.time_val, "time")
        volume[:, :, state.time_val] = time_slice
        fetched_slices.append(time_slice)

    return volume, fetched_slices


def initialize_viewer(shape, loaded_path):
    backend.shape = tuple(shape)
    backend.viewer_ready = True
    backend.loaded_path = loaded_path

    state.iline_max = backend.shape[0] - 1
    state.xline_max = backend.shape[1] - 1
    state.time_max = backend.shape[2] - 1

    state.iline_val = min(state.iline_val or backend.shape[0] // 2, state.iline_max)
    state.xline_val = min(state.xline_val or backend.shape[1] // 2, state.xline_max)
    state.time_val = min(state.time_val or backend.shape[2] // 2, state.time_max)
    state.viewer_loaded_name = os.path.basename(loaded_path)
    state.viewer_status_msg = f"Loaded {state.viewer_loaded_name}"


async def upload_ml_file(request):
    reader = await request.multipart()
    field = await reader.next()

    if field is None or field.name != "file":
        return web.json_response({"error": "No file uploaded."}, status=400)

    # Proxy the upload chunk directly to the Backend Server
    data = await field.read()
    try:
        resp = await asyncio.to_thread(
            requests.post, 
            f"{BACKEND_URL}/upload_ml", 
            files={"file": (field.filename, data)}
        )
        resp.raise_for_status()
        return web.json_response(resp.json())
    except Exception as e:
        return web.json_response({"error": str(e)}, status=500)


async def run_ml_file(request):
    try:
        payload = await request.json()
    except Exception:
        return web.json_response({"error": "Invalid request payload."}, status=400)

    input_path = (payload.get("input_path") or "").strip()
    if not input_path:
        return web.json_response({"error": "Missing input file path."}, status=400)

    try:
        # 1. Trigger the background job on backend
        resp = await asyncio.to_thread(requests.post, f"{BACKEND_URL}/run_ml", json={"input_path": input_path})
        resp.raise_for_status()
        
        # 2. Poll backend status until complete or error
        while True:
            await asyncio.sleep(1.5)
            status_resp = await asyncio.to_thread(requests.get, f"{BACKEND_URL}/ml_status")
            if status_resp.status_code == 200:
                data = status_resp.json()
                if not data.get("running"):
                    if data.get("stage") == "error":
                        return web.json_response({"error": data.get("message")}, status=500)
                    elif data.get("stage") == "complete":
                        # Tell UI what the backend output path is
                        output_path = data.get("output_path")
                        with state:
                            state.ml_output_path = output_path
                        return web.json_response({
                            "output_path": output_path,
                            # Provide a valid download URL if we implement proxy download,
                            # for now, placeholder or point direct to backend if exposed
                            "download_url": f"{BACKEND_URL}/download/{quote(os.path.basename(output_path))}" if output_path else ""
                        })
    except Exception as e:
        return web.json_response({"error": str(e)}, status=500)


async def ml_status(request):
    try:
        resp = await asyncio.to_thread(requests.get, f"{BACKEND_URL}/ml_status")
        resp.raise_for_status()
        data = resp.json()
        
        # When backend is complete, push the path into Trame's state so the viewer can use it
        if data.get("stage") == "complete" and data.get("output_path"):
            with state:
                state.ml_output_path = data.get("output_path")
                
        return web.json_response(data)
    except Exception as e:
        return web.json_response({"error": str(e)}, status=500)


def bind_server_routes(wslink_server):
    wslink_server.app.router.add_post("/upload_ml", upload_ml_file)
    wslink_server.app.router.add_post("/run_ml", run_ml_file)
    wslink_server.app.router.add_get("/ml_status", ml_status)

ctrl.on_server_bind.add(bind_server_routes)


def handle_viewer_upload(file_path=None):
    target_path = (file_path or state.ml_output_path or "").strip()
    
    if not target_path:
        try:
            resp = requests.get(f"{BACKEND_URL}/ml_status", timeout=5)
            if resp.status_code == 200:
                data = resp.json()
                if data.get("stage") == "complete" and data.get("output_path"):
                    target_path = data.get("output_path")
                    with state:
                        state.ml_output_path = target_path
        except Exception:
            pass

    if not target_path:
        state.viewer_status_msg = "Run ML first so there is an output SEG-Y to visualize."
        return

    state.viewer_processing = True
    state.viewer_status_msg = f"Connecting to {BACKEND_URL}..."

    try:
        metadata = configure_slice_server(target_path)
        initialize_viewer(metadata["shape"], target_path)
        update_slices()
    except Exception as err:
        state.viewer_status_msg = f"Viewer Error: {str(err)}"
    finally:
        state.viewer_processing = False


@state.change("iline_val", "xline_val", "time_val", "iline_check", "xline_check", "time_check")
def update_slices(**kwargs):
    if not backend.viewer_ready:
        return

    backend.renderer.RemoveAllViewProps()

    if not any((state.iline_check, state.xline_check, state.time_check)):
        backend.window.Render()
        ctrl.view_update()
        return

    state.viewer_processing = True

    try:
        sparse_cube, fetched_slices = build_sparse_remote_cube()
        amplitudes = np.concatenate([slice_.flatten() for slice_ in fetched_slices])

        image = build_vtk_image(sparse_cube)
        mapper = build_mapper(image, amplitudes)
        actors = create_actors(
            mapper,
            state.iline_val,
            state.xline_val,
            state.time_val,
            backend.shape,
        )

        if state.iline_check:
            backend.renderer.AddActor(actors[0])
        if state.xline_check:
            backend.renderer.AddActor(actors[1])
        if state.time_check:
            backend.renderer.AddActor(actors[2])

        backend.renderer.ResetCamera()
        backend.window.Render()
        state.viewer_status_msg = (
            f"Showing {state.viewer_loaded_name or 'volume'} "
            f"I:{state.iline_val} X:{state.xline_val} T:{state.time_val}"
        )
        ctrl.view_update()
    except Exception as err:
        state.viewer_status_msg = f"Viewer Error: {str(err)}"
    finally:
        state.viewer_processing = False


# --- UI ---
with SinglePageLayout(server) as layout:
    layout.title.set_text("Seismic Portal Pro")

    # Content
    with layout.content:
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

        .pulse-icon {
            animation: pulse-glow 2.5s infinite alternate;
        }
        @keyframes pulse-glow {
            0% { transform: scale(0.95); filter: drop-shadow(0 0 10px rgba(88, 166, 255, 0.2)); }
            100% { transform: scale(1.05); filter: drop-shadow(0 0 25px rgba(88, 166, 255, 0.6)); }
        }

        .v-tabs-bar { border-bottom: 1px solid rgba(255,255,255,0.05); }
        .v-tab { 
            text-transform: none !important; 
            font-weight: 600 !important; 
            letter-spacing: 0.5px; 
            color: #8b9bb4 !important; 
        }
        .v-tab--active {
            color: #00d2ff !important; 
        }
        .v-input__slider { margin-top: 0px; }
        """)

        with vuetify.VAppBar(app=True, flat=True, classes="glass-nav", dark=True):
            vuetify.VIcon("mdi-layers-triple", classes="mr-3", color="#00d2ff", size=28)
            vuetify.VToolbarTitle("Seismic Portal", classes="text-gradient text-h5")
            vuetify.VSpacer()

        with vuetify.VContainer(fluid=True, classes="pa-6 mt-12"):

            with vuetify.VTabs(
                v_model=("active_tab",),
                background_color="transparent",
                slider_color="#00d2ff", 
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
                with vuetify.VTabItem():
                    with vuetify.VRow(justify="center", classes="mt-6"):
                        with vuetify.VCol(cols="12", sm="11", md="9", lg="8"):
                            html.Iframe(
                                src="/mltool/ml_tool.html",
                                style="width: 100%; height: 620px; border: 0; border-radius: 16px; background: transparent;",
                            )
                with vuetify.VTabItem():
                    with vuetify.VRow(classes="mt-2", spacing=4):
                        with vuetify.VCol(cols="12", md="4", lg="3"):
                            with vuetify.VCard(classes="pa-6 card-modern", elevation=0, style="height: 100%; min-height: 600px;", dark=True):
                                html.Div("Data Source", classes="section-title")
                                html.P(
                                    "Ensure backend is running, wait for ML, then load the generated SEG-Y here.",
                                    classes="text-body-2 grey--text text--lighten-1 mb-3"
                                )
                                vuetify.VBtn(
                                    "Load Latest ML Output",
                                    block=True,
                                    rounded=True,
                                    class_="glow-btn mb-3",
                                    click=handle_viewer_upload,
                                )
                                html.Div("{{ viewer_status_msg }}", classes="text-caption cyan--text mb-6")

                                vuetify.VDivider(dark=True, classes="mb-6", style="border-color: rgba(255,255,255,0.05);")

                                html.Div("Slicing Controls", classes="section-title")
                                with vuetify.VRow(align="center", classes="mb-0"):
                                    with vuetify.VCol(cols="auto"):
                                        vuetify.VSwitch(v_model=("iline_check",), color="#00d2ff", hide_details=True, inset=True, dense=True, dark=True)
                                    with vuetify.VCol():
                                        html.Span("Inline", classes="text-body-2 white--text")
                                vuetify.VSlider(v_if=("iline_check",), v_model=("iline_val",), min=0, max=("iline_max",), 
                                                color="#00d2ff", track_color="grey darken-3", thumb_label=True, classes="mt-1 mb-4", dark=True)

                                with vuetify.VRow(align="center", classes="mb-0"):
                                    with vuetify.VCol(cols="auto"):
                                        vuetify.VSwitch(v_model=("xline_check",), color="#ff9800", hide_details=True, inset=True, dense=True, dark=True)
                                    with vuetify.VCol():
                                        html.Span("Crossline", classes="text-body-2 white--text")
                                vuetify.VSlider(v_if=("xline_check",), v_model=("xline_val",), min=0, max=("xline_max",), 
                                                color="#ff9800", track_color="grey darken-3", thumb_label=True, classes="mt-1 mb-4", dark=True)

                                with vuetify.VRow(align="center", classes="mb-0"):
                                    with vuetify.VCol(cols="auto"):
                                        vuetify.VSwitch(v_model=("time_check",), color="#b388ff", hide_details=True, inset=True, dense=True, dark=True)
                                    with vuetify.VCol():
                                        html.Span("Time Slice", classes="text-body-2 white--text")
                                vuetify.VSlider(v_if=("time_check",), v_model=("time_val",), min=0, max=("time_max",), 
                                                color="#b388ff", track_color="grey darken-3", thumb_label=True, classes="mt-1", dark=True)

                        with vuetify.VCol(cols="12", md="8", lg="9"):
                            with vuetify.VCard(
                                classes="viewer-box",
                                style="height: 100%; min-height: 600px;",
                                dark=True
                            ):
                                html.Div(classes="viewer-empty-bg")
                                vtk_view = vtk_widgets.VtkRemoteView(
                                    backend.window,
                                    interactive_ratio=1,
                                    style="width: 100%; height: 600px; position: relative; z-index: 1;",
                                )
                                ctrl.view_update = vtk_view.update

                with vuetify.VTabItem():
                    with vuetify.VRow(justify="center", align="center", style="height: 600px;"):
                        with vuetify.VCol(classes="text-center"):
                            vuetify.VIcon("mdi-hexagram-outline", size=100, color="rgba(0, 210, 255, 0.4)", classes="pulse-icon")
                            html.H3("ANALYSIS ENGINE", classes="text-h4 mt-6 text-gradient", style="letter-spacing: 3px;")
                            html.P("Advanced morphometric and curvature attributes are currently under development.",
                                   classes="text-body-1 grey--text text--lighten-1 mt-3")


if __name__ == "__main__":
    print("Starting Seismic Portal Pro on http://localhost:8081")
    print(f"Communicating with Backend Server at: {BACKEND_URL}")
    server.start(port=8081)
