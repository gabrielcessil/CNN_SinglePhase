import os
import sys
import json
import subprocess
import torch
import pyvista as pv
import numpy as np

config = {
    'samples': 4,
    'ambient_samples': 6,
    'resolution': [1200, 1200],
    'scalar_name': 'Density',
    'vector_name': 'Velocity',
    'solid_ambient': [0.6, 0.5804, 0.6706],
    'solid_diffuse': [0.6, 0.5804, 0.6706],
}

out_dir = './output_frames/'
jobs_dir = os.path.join(out_dir, 'job_configs')
os.makedirs(jobs_dir, exist_ok=True)

device = 'cuda' if torch.cuda.is_available() else 'cpu'

# 1. Load PyTorch Model & Dataset
print("[Orchestrator] Loading model and dataset...", flush=True)
from Architectures.Unet import Extended_DannyKo
from Architectures.Models import SubModels_Composition
from Utilities import dataset_reader as dr
from Utilities import velocity_usage as vu

danny_model = Extended_DannyKo()
concat_model = SubModels_Composition(
    main_model=danny_model,
    z_name="./Trained_Models/NN_Trainning_26_August_2026_03-45PM_Job27376/model_LowerValidationLoss.pth",
    x_name="./Trained_Models/NN_Trainning_26_August_2026_06-21PM_Job27380/model_LowerValidationLoss.pth",
    p_name="./Trained_Models/NN_Trainning_26_August_2026_03-47PM_Job27377/model_LowerValidationLoss.pth",
    device=device, is_eval=True
)

dataset_path = "../NN_Datasets_Grad_Dist_40_5_55/Test_Oliveira_Bentheimer_SAug_DNorm.h5"
dataset = dr.LazyDatasetTorch(h5_path=dataset_path, x_dtype=torch.float32, y_dtype=torch.float32, component=5)
net_input, net_target = dataset[23]
net_input = net_input.unsqueeze(0).to(device)

# 2. Run Inference & Export VTIs via PyVista
with torch.no_grad():
    pred = concat_model.predict(net_input)
    pred = vu.tensor_denorm(out=pred, inp=net_input)

geom = net_input[0, 0].cpu().numpy()
nz, ny, nx = geom.shape

def save_to_vti(geom_np, ux, uy, uz, filepath):
    grid = pv.ImageData(dimensions=(nx, ny, nz))
    grid["Density"] = np.where(geom_np > 0, 1.0, -1.0).astype(np.float32).flatten(order="C")
    grid["Velocity"] = np.column_stack((ux.flatten(order="C"), uy.flatten(order="C"), uz.flatten(order="C"))).astype(np.float32)
    grid.set_active_scalars("Density")
    grid.set_active_vectors("Velocity")
    grid.save(filepath)

target_vti = os.path.join(out_dir, "target_sample.vti")
pred_vti = os.path.join(out_dir, "pred_sample.vti")

save_to_vti(geom, net_target[0, 2].cpu().numpy(), net_target[0, 1].cpu().numpy(), net_target[0, 0].cpu().numpy(), target_vti)
save_to_vti(geom, pred[0, 2].cpu().numpy(), pred[0, 1].cpu().numpy(), pred[0, 0].cpu().numpy(), pred_vti)

sources = {'target': target_vti, 'pred': pred_vti}

# 3. Camera Manager Class
class CameraManager:
    def __init__(self, camera_pos, focal_point, view_up, view_angle=30, parallel_scale=103.05):
        self.camera_pos = np.array(camera_pos, dtype=float)
        self.focal_point = np.array(focal_point, dtype=float)
        self.view_up = np.array(view_up, dtype=float)
        self.view_angle = view_angle
        self.parallel_scale = parallel_scale

    def get_orbital_camera(self, frame_idx, total_frames, rotation_angle=360):
        r = self.camera_pos - self.focal_point
        theta = np.deg2rad((rotation_angle / total_frames) * frame_idx)
        cos_t, sin_t = np.cos(theta), np.sin(theta)
        
        rx = r[0] * cos_t + r[2] * sin_t
        ry = r[1]
        rz = -r[0] * sin_t + r[2] * cos_t
        
        ux = self.view_up[0] * cos_t + self.view_up[2] * sin_t
        uy = self.view_up[1]
        uz = -self.view_up[0] * sin_t + self.view_up[2] * cos_t
        
        pos = (self.focal_point + np.array([rx, ry, rz])).tolist()
        up = np.array([ux, uy, uz]).tolist()
        return pos, up

camera_mgr = CameraManager(
    camera_pos=[-145.62, 258.12, 337.27],
    focal_point=[54.23, 49.59, 63.19],
    view_up=[0.28, 0.85, -0.44]
)

panels = [
    {"title": "Solid Geometry", "show_solid": True, "solid_opacity": 1.0, "show_flow": False, "flow_source": None},
    {"title": "Predicted Flow", "show_solid": True, "solid_opacity": 0.25, "show_flow": True, "flow_source": 'pred'},
    {"title": "Simulated Flow", "show_solid": True, "solid_opacity": 0.25, "show_flow": True, "flow_source": 'target'}
]

total_frames = 180
print(f"[Orchestrator] Generating render jobs for {total_frames} frames...", flush=True)

# 4. Animation Loop: Dispatching to pvpython workers
for frame_idx in range(total_frames):
    pos, up = camera_mgr.get_orbital_camera(frame_idx, total_frames)
    camera_params = {
        'camera_pos': pos,
        'focal_point': camera_mgr.focal_point.tolist(),
        'camera_view_up': up,
        'camera_view_angle': camera_mgr.view_angle,
        'camera_parallel_scale': camera_mgr.parallel_scale
    }
    
    output_filename = os.path.join(out_dir, f"frame_{frame_idx:04d}.png")
    
    job_data = {
        'data_sources': sources,
        'panels': panels,
        'camera_params': camera_params,
        'output_filepath': output_filename,
        'config': config
    }
    
    job_path = os.path.join(jobs_dir, f"job_{frame_idx:04d}.json")
    with open(job_path, 'w') as f:
        json.dump(job_data, f)
        
    cmd = ["pvpython", "--force-offscreen-rendering", "paraview_viz.py", "--worker", job_path]
    subprocess.run(cmd, check=True)

print("[Orchestrator] All frames rendered successfully!")