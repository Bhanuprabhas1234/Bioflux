# from fastapi import FastAPI, Request, HTTPException
# from fastapi.staticfiles import StaticFiles
# from fastapi.templating import Jinja2Templates
# from fastapi.responses import HTMLResponse, JSONResponse, FileResponse
# import os
# from main_rl_eg import NDVIPredatorPreySimulation
# from typing import Optional
# import uvicorn
# import numpy as np
# import json

# app = FastAPI()

# # Custom JSON encoder for numpy types
# class NumpyEncoder(json.JSONEncoder):
#     def default(self, obj):
#         if isinstance(obj, np.integer):
#             return int(obj)
#         elif isinstance(obj, np.floating):
#             return float(obj)
#         elif isinstance(obj, np.ndarray):
#             return obj.tolist()
#         return super(NumpyEncoder, self).default(obj)

# # Mount static files
# app.mount("/static", StaticFiles(directory="static"), name="static")
# templates = Jinja2Templates(directory="templates")

# # Global simulation object
# simulation: Optional[NDVIPredatorPreySimulation] = None

# @app.get("/", response_class=HTMLResponse)
# async def index(request: Request):
#     """Render the main page."""
#     return templates.TemplateResponse("index.html", {"request": request})

# @app.post("/start_simulation")
# async def start_simulation(request: Request):
#     """Start the simulation with user-provided parameters."""
#     global simulation
#     data = await request.json()
    
#     tiff_folder = data.get('tiff_folder', 'data_tiff')
#     num_herbivores = int(data.get('num_herbivores', 1000))
#     num_carnivores = int(data.get('num_carnivores', 200))
#     steps = int(data.get('steps', 100))
#     rl_herbivores = bool(data.get('rl_herbivores', False))

#     # Initialize simulation
#     simulation = NDVIPredatorPreySimulation(
#         tiff_folder_path=tiff_folder,
#         num_herbivores=num_herbivores,
#         num_carnivores=num_carnivores,
#         ndvi_update_frequency=10,
#         rl_herbivores=rl_herbivores
#     )

#     # Run simulation
#     simulation.run_simulation(steps=steps, animate=False, save_animation=False)

#     return JSONResponse({"message": "Simulation completed successfully!"})

# @app.get("/get_results")
# async def get_results():
#     """Return the results of the simulation."""
#     global simulation
#     if not simulation:
#         raise HTTPException(status_code=400, detail="Simulation not started yet.")

#     # Prepare results with proper serialization
#     results = {
#         'herbivore_count_history': convert_to_serializable(simulation.herbivore_count_history),
#         'carnivore_count_history': convert_to_serializable(simulation.carnivore_count_history),
#         'ndvi_mean_history': convert_to_serializable(simulation.ndvi_mean_history)
#     }
    
#     # Use our custom encoder
#     return JSONResponse(content=results, media_type="application/json")

# def convert_to_serializable(data):
#     """Convert numpy arrays and types to Python native types."""
#     if isinstance(data, np.ndarray):
#         return data.astype(float).tolist()
#     elif isinstance(data, (np.float32, np.float64)):
#         return float(data)
#     elif isinstance(data, (np.int32, np.int64)):
#         return int(data)
#     elif isinstance(data, list):
#         return [convert_to_serializable(item) for item in data]
#     return data

# @app.get("/get_plot_data")
# async def get_plot_data():
#     """Provide data for interactive Plotly plots."""
#     global simulation
#     if not simulation:
#         raise HTTPException(status_code=400, detail="Simulation not started yet.")

#     plot_data = simulation.get_plotly_graph_data()
    
#     # Convert all numpy arrays to lists and numpy types to native Python types
#     converted_data = {}
#     for key, value in plot_data.items():
#         converted_data[key] = convert_to_serializable(value)
    
#     return JSONResponse(content=converted_data, media_type="application/json")

# @app.get("/get_plot_image/{plot_type}")
# async def get_plot_image(plot_type: str):
#     """Generate and serve plots as images."""
#     global simulation
#     if not simulation:
#         raise HTTPException(status_code=400, detail="Simulation not started yet.")

#     plot_file = f'static/{plot_type}_plot.png'
    
#     try:
#         # Ensure the static directory exists
#         os.makedirs('static', exist_ok=True)
        
#         if plot_type == 'herbivores' and hasattr(simulation, 'plot_herbivore_population'):
#             simulation.plot_herbivore_population(save_path=plot_file)
#         elif plot_type == 'carnivores' and hasattr(simulation, 'plot_carnivore_population'):
#             simulation.plot_carnivore_population(save_path=plot_file)
#         elif plot_type == 'ndvi' and hasattr(simulation, 'plot_ndvi_mean'):
#             simulation.plot_ndvi_mean(save_path=plot_file)
#         else:
#             raise HTTPException(status_code=400, detail="Invalid or unsupported plot type.")
        
#         # Verify the file was created
#         if not os.path.exists(plot_file):
#             raise HTTPException(status_code=500, detail="Plot file was not created.")
            
#     except Exception as e:
#         raise HTTPException(status_code=500, detail=f"Failed to generate plot: {str(e)}")

#     return FileResponse(plot_file, media_type='image/png')

# if __name__ == "__main__":
#     uvicorn.run(app, host="0.0.0.0", port=8000)

from fastapi import FastAPI, Request, HTTPException
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from fastapi.responses import HTMLResponse, JSONResponse, FileResponse
import os
from main_rl_eg import NDVIPredatorPreySimulation
from typing import Optional
import uvicorn
import numpy as np
import json
import matplotlib.pyplot as plt
import io
import base64
import requests
from pathlib import Path

app = FastAPI()

# Create static directory if it doesn't exist
Path("static").mkdir(exist_ok=True)

# Custom JSON encoder for numpy types
class NumpyEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, np.integer):
            return int(obj)
        elif isinstance(obj, np.floating):
            return float(obj)
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        return super(NumpyEncoder, self).default(obj)

# Mount static files
app.mount("/static", StaticFiles(directory="static"), name="static")
app.mount("/model", StaticFiles(directory="public/models"), name="model")
templates = Jinja2Templates(directory="templates")

# Global simulation object
simulation: Optional[NDVIPredatorPreySimulation] = None

def download_3d_model(url, filename):
    """Download a 3D model from a URL if it doesn't exist locally"""
    filepath = f"static/{filename}"
    if not os.path.exists(filepath):
        response = requests.get(url)
        with open(filepath, 'wb') as f:
            f.write(response.content)
    return f"/static/{filename}"

def get_landing_page_3d_assets():
    """Get URLs for 3D models used in landing page"""
    return {
        "deer_model": download_3d_model(
            "https://github.com/KhronosGroup/glTF-Sample-Models/raw/master/2.0/Deer/glTF/Deer.gltf",
            "deer.gltf"
        ),
        "wolf_model": download_3d_model(
            "https://github.com/KhronosGroup/glTF-Sample-Models/raw/master/2.0/Wolf/glTF/Wolf.gltf",
            "wolf.gltf"
        ),
        "tree_model": download_3d_model(
            "https://github.com/KhronosGroup/glTF-Sample-Models/raw/master/2.0/Tree/glTF/Tree.gltf",
            "tree.gltf"
        ),
        "terrain_texture": download_3d_model(
            "https://raw.githubusercontent.com/KhronosGroup/glTF-Sample-Models/master/2.0/MetalRoughSpheres/glTF/MetalRoughSpheres_baseColor.png",
            "grass_texture.png"
        )
    }

@app.get("/", response_class=HTMLResponse)
async def landing_page(request: Request):
    """Render the landing page with 3D visualization"""
    assets = get_landing_page_3d_assets()
    return templates.TemplateResponse("landing.html", {
        "request": request,
        "assets": assets
    })

@app.get("/simulation", response_class=HTMLResponse)
async def simulation_page(request: Request):
    """Render the main simulation page"""
    assets = get_landing_page_3d_assets()
    return templates.TemplateResponse("index.html", {
        "request": request,
        "assets": assets
    })

@app.post("/start_simulation")
async def start_simulation(request: Request):
    """Start the simulation with user-provided parameters."""
    global simulation
    data = await request.json()
    
    tiff_folder = data.get('tiff_folder', 'data_tiff')
    num_herbivores = int(data.get('num_herbivores', 1000))
    num_carnivores = int(data.get('num_carnivores', 200))
    steps = int(data.get('steps', 100))
    rl_herbivores = bool(data.get('rl_herbivores', False))

    # Initialize simulation
    simulation = NDVIPredatorPreySimulation(
        tiff_folder_path=tiff_folder,
        num_herbivores=num_herbivores,
        num_carnivores=num_carnivores,
        ndvi_update_frequency=10,
        rl_herbivores=rl_herbivores
    )

    # Run simulation
    simulation.run_simulation(steps=steps, animate=False, save_animation=False)

    return JSONResponse({"message": "Simulation completed successfully!"})

@app.get("/get_results")
async def get_results():
    """Return the results of the simulation."""
    global simulation
    if not simulation:
        raise HTTPException(status_code=400, detail="Simulation not started yet.")

    results = {
        'herbivore_count_history': convert_to_serializable(simulation.herbivore_count_history),
        'carnivore_count_history': convert_to_serializable(simulation.carnivore_count_history),
        'ndvi_mean_history': convert_to_serializable(simulation.ndvi_mean_history)
    }
    
    return JSONResponse(content=results)

def convert_to_serializable(data):
    """Convert numpy arrays and types to Python native types."""
    if isinstance(data, np.ndarray):
        return data.astype(float).tolist()
    elif isinstance(data, (np.float32, np.float64)):
        return float(data)
    elif isinstance(data, (np.int32, np.int64)):
        return int(data)
    elif isinstance(data, list):
        return [convert_to_serializable(item) for item in data]
    return data

@app.get("/get_plot_data")
async def get_plot_data():
    """Provide data for interactive Plotly plots."""
    global simulation
    if not simulation:
        raise HTTPException(status_code=400, detail="Simulation not started yet.")

    plot_data = {
        "herbivore_count_history": simulation.herbivore_count_history,
        "carnivore_count_history": simulation.carnivore_count_history,
        "ndvi_mean_history": simulation.ndvi_mean_history
    }

    # Convert all numpy arrays to lists and numpy types to native Python types
    converted_data = {}
    for key, value in plot_data.items():
        converted_data[key] = convert_to_serializable(value)
    
    return JSONResponse(content=converted_data, media_type="application/json")

@app.get("/get_plot_image/{plot_type}")
async def get_plot_image(plot_type: str):
    """Generate and serve plots as images."""
    global simulation
    
    # Validate simulation exists
    if simulation is None:
        raise HTTPException(
            status_code=400,
            detail="Simulation not initialized. Please start the simulation first."
        )
    
    # Validate plot type
    valid_plot_types = ['herbivores', 'carnivores', 'ndvi']
    if plot_type not in valid_plot_types:
        raise HTTPException(
            status_code=400,
            detail=f"Invalid plot type. Must be one of: {', '.join(valid_plot_types)}"
        )
    
    # Create plots directory if it doesn't exist
    os.makedirs('static/plots', exist_ok=True)
    plot_file = f'static/plots/{plot_type}_plot.png'
    
    try:
        # Create figure with appropriate size
        plt.figure(figsize=(10, 8))
        
        # Determine which grid to plot
        if plot_type == 'herbivores':
            grid = simulation.herbivore_grid
            cmap = 'Blues'
            title = 'Herbivore Distribution'
            clim = (0, np.max(grid) if np.max(grid) > 0 else 1)  # Handle empty grid
        elif plot_type == 'carnivores':
            grid = simulation.carnivore_grid
            cmap = 'Reds'
            title = 'Carnivore Distribution'
            clim = (0, np.max(grid) if np.max(grid) > 0 else 1)
        else:  # ndvi
            grid = simulation.ndvi_grid
            cmap = 'YlGn'
            title = 'NDVI Distribution'
            clim = (0, 1)  # NDVI is normalized 0-1
        
        # Create the plot
        img = plt.imshow(grid, cmap=cmap)
        img.set_clim(*clim)  # Set color limits
        plt.title(title)
        plt.colorbar(label='Value')
        
        # Save and close
        plt.tight_layout()
        plt.savefig(plot_file, bbox_inches='tight', dpi=100)
        plt.close()
        
        # Verify the file was created
        if not os.path.exists(plot_file):
            raise HTTPException(
                status_code=500,
                detail="Plot file was not created successfully"
            )
            
        return FileResponse(plot_file, media_type='image/png')
        
    except Exception as e:
        # Clean up if there was an error
        if 'plt' in locals():
            plt.close()
        if os.path.exists(plot_file):
            os.remove(plot_file)
            
        raise HTTPException(
            status_code=500,
            detail=f"Failed to generate plot: {str(e)}"
        )

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)