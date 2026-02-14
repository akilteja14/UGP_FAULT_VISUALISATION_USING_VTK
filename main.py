import os
from nicegui import ui, run
from ml_core import load_model, process_segy

# Workspace configuration
UPLOAD_DIR, OUTPUT_DIR = 'data/uploads', 'data/outputs'
os.makedirs(UPLOAD_DIR, exist_ok=True)
os.makedirs(OUTPUT_DIR, exist_ok=True)

# Cache the model globally - load only once
_model_cache = None

def get_model():
    global _model_cache
    if _model_cache is None:
        print("Initializing ML Model...")
        _model_cache = load_model()
        print("Model ready.")
    return _model_cache

model = get_model()

# Reference for UI elements that will be created
progress_container = None
status_msg = None
upload_card = None
results_container = None
coord_card = None

async def handle_upload(e):
    global progress_container, status_msg, upload_card, results_container, coord_card
    
    filename = getattr(e, 'name', 'seismic_data.segy')
    input_path = os.path.join(UPLOAD_DIR, filename)
    output_path = os.path.join(OUTPUT_DIR, f"faults_{filename}")

    # Read and save the file
    try:
        file_obj = getattr(e, 'content', None) or getattr(e, 'file', None)
        content = await file_obj.read() if hasattr(file_obj, 'read') else file_obj
        with open(input_path, 'wb') as f:
            f.write(content)
        print(f"[UI] File uploaded: {input_path}")
    except Exception as err:
        ui.notify(f'Upload Error: {err}', type='negative')
        print(f"[UI] Upload error: {err}")
        return

    # --- UI UPDATE START ---
    # We explicitly set visibility and call update() to force the browser to show it
    try:
        if upload_card:
            upload_card.set_visibility(False)
        if progress_container:
            progress_container.set_visibility(True)
        if status_msg:
            status_msg.set_text("AI is analyzing 3D patches... Please wait.")
        print("[UI] Progress container shown")
    except Exception as err:
        print(f"[UI] Error updating visibility: {err}")
    
    # Add a small delay to let the UI render the spinner before starting heavy processing
    import asyncio
    await asyncio.sleep(1.0)
    # --- UI UPDATE END ---

    try:
        print("[ML] Starting ML process...")
        # Launch the ML process without callback (multiprocessing can't pickle nested functions)
        await run.cpu_bound(process_segy, input_path, output_path, model)
        
        print("[ML] ML process completed successfully")
        ui.notify('Success! Faults identified.', type='positive')
        if status_msg:
            status_msg.set_text("Analysis Complete! Result is ready for download.")
        if progress_container:
            progress_container.set_visibility(False)
        
        # Show coordinate selection card
        if coord_card:
            coord_card.set_visibility(True)
        
        if results_container:
            with results_container:
                ui.button('Download Result', on_click=lambda: ui.download(output_path, 'faults_seismic.segy')).props('icon=download color=positive')
                ui.button('Process New File', on_click=lambda: ui.navigate.to('/')).props('outline color=primary')
            
    except Exception as err:
        print(f"[UI] ML Failed: {type(err).__name__}: {err}")
        ui.notify(f'ML Failed: {str(err)}', type='negative')
        if status_msg:
            status_msg.set_text(f"Error: {str(err)}")
        if progress_container:
            progress_container.set_visibility(False)

# --- UI Layout ---
ui.query('body').style('background-color: #f0f2f5')

with ui.column().classes('w-full items-center q-pa-lg'):
    ui.label('3D Seismic Fault Identification').classes('text-h3 text-primary q-mb-lg')
    
    # Progress Section
    with ui.column().classes('w-full max-w-2xl items-center') as pc:
        progress_container = pc
        pc.set_visibility(False)
        status_msg = ui.label("Processing...")
        ui.spinner(size='50px').classes('q-my-md')
    
    # Initialize results container for later use
    results_container = ui.column().classes('w-full items-center q-mt-md')
    
    # Upload Card
    with ui.card().classes('w-full max-w-2xl p-8 shadow-2') as uc:
        upload_card = uc
        ui.label('Upload a SEG-Y file to begin.').classes('text-grey-7 q-mb-md')
        ui.upload(on_upload=handle_upload, auto_upload=True).classes('w-full').props('accept=.segy,.sgy')

    # Coordinate Input Section (Based on your hand-drawn plan)
    with ui.card().classes('w-full max-w-2xl p-6 q-mt-md shadow-1') as cc:
        coord_card = cc
        cc.set_visibility(False)
        ui.label('Cut Plane Selection').classes('text-h6 q-mb-sm')
        with ui.row().classes('w-full justify-around'):
            ui.number(label='Inline (X)', value=0).classes('w-32').props('outlined')
            ui.number(label='Crossline (Y)', value=0).classes('w-32').props('outlined')
            ui.number(label='Depth/Time (Z)', value=0).classes('w-32').props('outlined')
        ui.label('Enter coordinates to slice the volume after processing.').classes('text-caption text-grey q-mt-sm')

ui.run(title="Seismic Fault AI")