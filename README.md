# Deep Learning for Lattice Boltzmann Method (LBM) Simulations in Porous Media

## Overview
This repository provides a PyTorch-based training pipeline for developing deep learning surrogates of Lattice Boltzmann Method (LBM) simulations.

---
# About the models

All the considered models are embbeded with nn.Module, and have the following neede structure:
* predict(): method used as forward() during test or applications. This method disables gradient computation and mask the output according to the input's solid.
* bin_input: attribute used to define the type of input. If True, thresholds the signal, if False let the input as it is.
* The models are image-to-image, not receiving out predicting any extra type of features. All inputs and features must be encoded as image channels.


# How to use our pre-trained models

Once the models are trained, they can be deployed to predict flow fields in porous media, generate initial conditions for Lattice Boltzmann Method (LBM) simulations, and visualize fluid pathways. All models are embedded with nn.Module and utilize a predict() method that disables gradient computation during testing and masks outputs based on the solid matrix.

The models used in our documented analysis are provided in the folder "Trained_Models". It contains the output logger with informations printed during training,
the '.json' useful to recreate their training and their '.pth' containing the model itself.

Below are provided workflow examples on how to use these models.

### 1. Generating Prediction and VTI file for Paraview

To evaluate the flow fields on a given geometry (.raw file) and output the results for 3D visualization.
This script loads the composed sub-models (Z, X, and P components) alongside the main architecture, evaluates the binary solid geometry, denormalizes the outputs, and runs critical sanity checks—such as ensuring velocities inside the solid matrix are strictly 0.0 and maximum velocities do not exceed 0.7. Finally, it exports a .vti file containing both density (geometry) and velocity arrays for **ParaView**.

```python
import numpy as np
from scipy.ndimage import distance_transform_edt as edt
import torch
import os
import pyvista as pv

# Adjust these imports according to your folder structure
from Architectures.Unet   import Extended_DannyKo
from Architectures.MSnet  import JavierSantos_Extended
from Architectures.Models import SubModels_Composition
from Utilities import start_handler as sh
from Utilities import velocity_usage as vu

def save_vti(filename, geometry, ux, uy, uz):
    """
    Exports a .vti file so ParaView can read both Density (Geometry) and Velocity.
    ParaView script thresholds Density <= 0.0 to render the solid.
    """
    Nz, Ny, Nx = geometry.shape
    grid = pv.ImageData(dimensions=(Nx, Ny, Nz))

    # Map geometry: Solid (-1.0) and Pore (1.0). ParaView thresholds < 0.0 as solid.
    density_array = np.where(geometry, 1.0, -1.0)
    grid.point_data["Density"] = density_array.flatten(order="F")

    # Stack velocity vectors
    grid.point_data["Velocity"] = np.column_stack((
        ux.flatten(order="F"),
        uy.flatten(order="F"),
        uz.flatten(order="F")
    ))

    grid.save(filename)

paths = [
    "./Example_Bentheimer/",
]

raw_file    = "domain.raw"
shape       = (120, 120, 120)
device      = "cpu"

danny_model = Extended_DannyKo()

# Z- component
model_full_z_name = "./Trained_Models/NN_Trainning_13_July_2026_06-02PM_Job26267/model_LowerValidationLoss.pth"
# X- component
model_full_x_name = "./Trained_Models/NN_Trainning_15_July_2026_03-59PM_Job26381/model_LowerValidationLoss.pth"
# P- component
model_full_p_name = "./Trained_Models/NN_Trainning_21_July_2026_05-22PM_Job26505/model_LowerValidationLoss.pth"

# Concatenation model
concat_model = SubModels_Composition(main_model=danny_model,
                                     z_name=model_full_z_name,
                                     x_name=model_full_x_name,
                                     p_name=model_full_p_name,
                                     device=device,
                                     is_eval=True)

# Force evaluation mode for deterministic predictions
concat_model.eval()
models = {"Composed Model ": concat_model}

for path in paths:
    for model_name, model in models.items():

        # Read Geometry
        geometry     = (np.fromfile(os.path.join(path, raw_file), dtype=np.uint8).reshape(shape) > 0)
        geometry_edt = edt(geometry).astype("float32")

        # Convert numpy array (Z,Y,X) to tensor (B=1,C=1, Z,Y,X)
        geometry_edt_tensor = torch.from_numpy(geometry_edt).unsqueeze(0).unsqueeze(0)

        # Make prediction
        print(f"Creating prediction with {model_name.strip()}")
        pred = model.predict(geometry_edt_tensor)

        # Denormalize predictions
        pred = vu.tensor_denorm(out=pred, inp=geometry_edt_tensor)

        # Extract components
        uz = pred[0,0].numpy()
        uy = pred[0,1].numpy()
        ux = pred[0,2].numpy()
        pr = pred[0,3].numpy()

        # Sanity Checks
        if not (uz.shape==shape and uy.shape==shape and ux.shape==shape and pr.shape==shape):
            raise Exception("Prediction doesn't match specified .raw shape.")
        if np.isnan(pred.numpy()).any() or np.isinf(pred.numpy()).any():
            raise ValueError(f"Model {model_name} predicted NaN or Inf values!")

        solid_vel_mag = np.sqrt(ux[~geometry]**2 + uy[~geometry]**2 + uz[~geometry]**2)
        if np.any(solid_vel_mag > 1e-6):
            print(f"   [!] WARNING: Predicted velocity inside solid! Forcing to 0.0.")
            ux[~geometry] = 0.0
            uy[~geometry] = 0.0
            uz[~geometry] = 0.0

        if np.max(np.abs(uz)) == 0.0 and np.max(np.abs(uy)) == 0.0 and np.max(np.abs(ux)) == 0.0:
            print(f"   [!] WARNING: Predicted a completely ZERO velocity field.")

        max_v = np.max(np.sqrt(ux**2 + uy**2 + uz**2))
        if max_v > 0.7:
            print(f"   [!] DANGER: Max velocity is {max_v:.4f}. LBPM may be unstable.")

        # Create output directory
        out_dir = os.path.join(path)
        os.makedirs(out_dir, exist_ok=True)

        # Calculate and print stats
        pred_perm = vu.permeability_calculation(pred, geometry_edt_tensor, denorm=False)
        print(f"   -> Perm | {float(pred_perm):.6e}")
        print(f"   -> Uz   | max: {uz.max():>13.6e} | mean: {uz.mean():>13.6e} | min: {uz.min():>13.6e}")

        # Write VTI for ParaView
        vti_path = os.path.join(out_dir, "output_data.vti")
        save_vti(vti_path, geometry, ux, uy, uz)
        print(f"   -> Saved VTI for ParaView: {vti_path}\n")
```


### Initilization for LBPM

In this work, we discuss flow prediction as LBM initial state. Our analysis were built on top of LBPM.

The initialization is performed by a '.raw' of doubles with 4 components: ux, uy, uz, pressure.

In order to initialize LBPM, predictions must be denormalized. Make sure the denormalization is coherent with the normalization used during the training of the particular model.

This code also verifies LBM stability (checking the Mach limit) and computes the required pressure drop based on the defined Reynolds number and density.

```python
import numpy as np
from scipy.ndimage import distance_transform_edt as edt
import torch
import matplotlib.pyplot as plt

from Architectures.Unet   import Extended_DannyKo
from Architectures.MSnet  import JavierSantos_Extended
from Architectures.Models import SubModels_Composition

from Utilities import start_handler as sh
from Utilities import velocity_usage as vu



data = {
      "./Example_BodyCenteredCubic/" : (32,32,32),
      "./Example_SphPore/"           : (120,120,120),
      "./Example_Bentheimer/"        : (120,120,120)
        }

raw_file    = "domain.raw"
device      = "cpu"



# ==============================================================================
# LOADING MODELs
# ==============================================================================
danny_model         = Extended_DannyKo()
# Z- component
model_full_z_name = "./Trained_Models/NN_Trainning_13_July_2026_06-02PM_Job26267/model_LowerValidationLoss.pth"
# X- component
model_full_x_name = "./Trained_Models/NN_Trainning_15_July_2026_03-59PM_Job26381/model_LowerValidationLoss.pth"
# P- component
model_full_p_name = "./Trained_Models/NN_Trainning_21_July_2026_05-22PM_Job26505/model_LowerValidationLoss.pth"

# Concatenation model (no main model)
concat_model      = SubModels_Composition(main_model=danny_model,
                                          z_name=model_full_z_name,
                                          x_name=model_full_x_name,
                                          p_name=model_full_p_name,
                                          device=device,
                                          is_eval=True)

models = {"Composed Model ": concat_model}

# ==============================================================================
# MAIN
# ==============================================================================

for path, shape in data.items():

    for model_name, model in models.items():


        geometry     = (np.fromfile(path+raw_file, dtype=np.uint8).reshape(shape)>0)
        geometry_edt = edt(geometry).astype("float32")



        # Convert numpy array (Z,Y,X) to tensor (B=1,C=1, Z,Y,X)
        geometry_edt = torch.from_numpy(geometry_edt).unsqueeze(0).unsqueeze(0)

        # Make prediction
        print(f"Creating prediction for {path}{raw_file} with {model_name}")
        pred    = model.predict(geometry_edt)
        uz      = pred[0,0].numpy()
        uy      = pred[0,1].numpy()
        ux      = pred[0,2].numpy()
        pr      = pred[0,3].numpy()

        # Denormalize predictions
        pred    = vu.tensor_denorm(out=pred, inp=geometry_edt)

        # Prepare data for start file
        uz      = pred[0,0].numpy()
        uy      = pred[0,1].numpy()
        ux      = pred[0,2].numpy()
        pr      = pred[0,3].numpy()

        # Sanity Checks
        #  - Shape Matching
        if not (uz.shape==shape and uy.shape==shape and ux.shape==shape and pr.shape==shape):
            raise Exception("Prediction dont match specified .raw shape.")
        #  - NaN and Inf presence check
        if np.isnan(pred.numpy()).any() or np.isinf(pred.numpy()).any():
            raise ValueError(f"Model {model_name} predicted NaN or Inf values!")
        #  - Solid Matching (No-Slip Condition)
        solid_vel_mag =  np.sqrt(ux[~geometry]**2 + uy[~geometry]**2 + uz[~geometry]**2)

        #  - LBM Stability Check (Max Velocity)
        max_v   = np.max(np.sqrt(ux**2 + uy**2 + uz**2))
        if max_v > 0.7:
            raise ValueError(f"   {model_name} predicted a max velocity of {max_v:.4f}. LBPM may be unstable due to Mach limit.")

        # Write start file
        print(f"   -> Creating Start.00000 file")
        sh.write_start_raw(
            filename = path+model_name+"/Start.00000",
            ux=ux, uy=uy, uz=uz, pr=pr
        )

        # Write the .db
        print(f"   -> Creating .db file")
        tau         = 1.5
        Re          = 0.1
        Dens        = 1.0
        p_drop      = vu.pressure_calculation(geometry, tau=tau, Re=Re, Dens=Dens)
        sh.write_lbpm_db(
            db_name = path+model_name+"/start_pressure.db",
            path    = "",
            tau     = tau,
            bc      = 3,
            din     = 1.0,
            dout    = 1.0-3*p_drop,
            nproc   = (1, 1, 1),
            n       = shape,
            N       = shape,
            analysis_interval = 1000,
            tolerance         = 1e-6,
            out_format        = "silo",
            Start             = True
        )

        # Rewrite .raw
        geometry.astype(np.uint8).tofile(path+model_name+"/"+raw_file)

        # Use prediction and show primary statistics
        pred_perm = vu.permeability_calculation(pred, geometry_edt, denorm=False)
        perm_val = float(pred_perm)
        print(f"   -> Perm | {perm_val:.6e}")
        print(f"   -> Uz   | max: {uz.max():>13.6e} | mean: {uz.mean():>13.6e} | min: {uz.min():>13.6e}")
        print(f"   -> Uy   | max: {uy.max():>13.6e} | mean: {uy.mean():>13.6e} | min: {uy.min():>13.6e}")
        print(f"   -> Ux   | max: {ux.max():>13.6e} | mean: {ux.mean():>13.6e} | min: {ux.min():>13.6e}")
        print(f"   -> Pr   | max: {pr.max():>13.6e} | mean: {pr.mean():>13.6e} | min: {pr.min():>13.6e}")
        print()
```

The Start file can be visualized in Paraview by selecting type double (LittleEndian) and 4 components.The components are sequenced as (Ux, Uy, Uz, Pr).


### Creating streamlines

This code helps to create streamlines directly from code using Paraview for a given created .vti. Run it with pvpython after installing paraview.

```python
import os
import glob
from paraview.simple import *

def _config_camera(view):
    view.CameraPosition = [-145.62500334229827, 258.12891252294224, 337.27420872575703]
    view.CameraFocalPoint = [54.23265623503145, 49.59503372400067, 63.19529352193284]
    view.CameraViewUp = [0.28077913995658044, 0.8513959344245512, -0.4430440580919559]
    view.CameraViewAngle = 30
    view.CameraParallelScale = 103.05702305034819
    view.CenterOfRotation = [59.5, 59.5, 59.5]

def _build_custom_wireframe():
    wireframe_coords = [
        ([59.5, 59.5, 59.5], [0.0, 59.5, 59.5]),
        ([59.5, 59.5, 59.5], [59.5, 119.0, 59.5]),
        ([59.5, 59.5, 59.5], [59.5, 59.5, 119.0]),
        ([0.0, 59.5, 59.5], [0.0, 119.0, 59.5]),
        ([0.0, 59.5, 59.5], [0.0, 59.5, 119.0]),
        ([59.5, 119.0, 59.5], [0.0, 119.0, 59.5]),
        ([59.5, 119.0, 59.5], [59.5, 119.0, 119.0]),
        ([59.5, 59.5, 119.0], [0.0, 59.5, 119.0]),
        ([59.5, 59.5, 119.0], [59.5, 119.0, 119.0]),
        ([0.0, 0.0, 0.0], [119.0, 0.0, 0.0]),
        ([0.0, 119.0, 0.0], [119.0, 119.0, 0.0]),
        ([0.0, 0.0, 119.0], [119.0, 0.0, 119.0]),
        ([0.0, 0.0, 0.0], [0.0, 119.0, 0.0]),
        ([119.0, 0.0, 0.0], [119.0, 119.0, 0.0]),
        ([119.0, 0.0, 119.0], [119.0, 119.0, 119.0]),
        ([0.0, 0.0, 0.0], [0.0, 0.0, 119.0]),
        ([119.0, 0.0, 0.0], [119.0, 0.0, 119.0]),
        ([119.0, 119.0, 0.0], [119.0, 119.0, 119.0]),
        ([0.0, 119.0, 0.0], [0.0, 119.0, 59.5]),
        ([119.0, 119.0, 119.0], [59.5, 119.0, 119.0]),
        ([0.0, 0.0, 119.0], [0.0, 59.5, 119.0])
    ]
    tubes = []
    for pt1, pt2 in wireframe_coords:
        l = Line(Point1=pt1, Point2=pt2)
        t = Tube(Input=l)
        t.Radius = 0.3
        t.Capping = 1
        tubes.append(t)
    return tubes

def plot_streamlines(vti_filepath, output_image_path, solid_color=[0.5, 0.5, 0.5], show_colorbar=True):
    ResetSession()

    reader = XMLImageDataReader(FileName=[vti_filepath])
    reader.PointArrayStatus = ['Density', 'Velocity']
    reader.UpdatePipeline()

    view = CreateRenderView()
    view.ViewSize = [1200, 1200]
    view.Background = [1.0, 1.0, 1.0]
    view.UseColorPaletteForBackground = 0
    view.OrientationAxesVisibility = 0
    view.EnableRayTracing = 1
    view.Shadows = 0
    view.SamplesPerPixel = 40
    view.AmbientSamples = 2
    if hasattr(view, "EnableOSPRayDenoiser"):
        view.EnableOSPRayDenoiser = 1

    thresh = Threshold(Input=reader)
    thresh.Scalars = ['POINTS', 'Density']
    thresh.ThresholdMethod = 'Between'
    thresh.LowerThreshold = -1e10
    thresh.UpperThreshold = 0.0

    clip1 = Clip(Input=thresh)
    clip1.ClipType = 'Plane'
    clip1.ClipType.Normal, clip1.ClipType.Origin = [-1.0, 0.0, 0.0], [59.5, 59.5, 59.5]

    clip2 = Clip(Input=thresh)
    clip2.ClipType = 'Plane'
    clip2.ClipType.Normal, clip2.ClipType.Origin = [0.0, 1.0, 0.0], [59.5, 59.5, 59.5]

    clip3 = Clip(Input=thresh)
    clip3.ClipType = 'Plane'
    clip3.ClipType.Normal, clip3.ClipType.Origin = [0.0, 0.0, 1.0], [59.5, 59.5, 59.5]

    for clip in [clip1, clip2, clip3]:
        disp = Show(clip, view)
        disp.ColorArrayName = ['POINTS', '']
        disp.DiffuseColor = solid_color
        disp.AmbientColor = solid_color
        disp.Opacity = 1.0

    wireframe_tubes = _build_custom_wireframe()
    for t in wireframe_tubes:
        disp_frame = Show(t, view)
        disp_frame.ColorArrayName = ['POINTS', '']
        disp_frame.DiffuseColor = [0.0, 0.0, 0.0]
        disp_frame.AmbientColor = [0.0, 0.0, 0.0]

    stream = StreamTracer(Input=reader, SeedType='Point Cloud')
    stream.Vectors = ['POINTS', 'Velocity']
    stream.MaximumStreamlineLength = 1000.0
    stream.SeedType.Center = [59.5, 59.5, 59.5]
    stream.SeedType.Radius = 120.0
    stream.SeedType.NumberOfPoints = 1150

    disp_stream = Show(stream, view)
    ColorBy(disp_stream, ('POINTS', 'Velocity', 'Magnitude'))

    velocityLUT = GetColorTransferFunction('Velocity')
    velocityLUT.ApplyPreset('autumn (matplotlib)', True)

    if show_colorbar:
        disp_stream.SetScalarBarVisibility(view, True)
        color_bar = GetScalarBar(velocityLUT, view)
        color_bar.TitleColor = [0.0, 0.0, 0.0]
        color_bar.LabelColor = [0.0, 0.0, 0.0]
        color_bar.TitleFontFamily = 'Times'
        color_bar.LabelFontFamily = 'Times'
        color_bar.TitleFontSize = 28
        color_bar.LabelFontSize = 28
        color_bar.AutomaticLabelFormat = 0
        color_bar.LabelFormat = '%.2e'
    else:
        disp_stream.SetScalarBarVisibility(view, False)

    _config_camera(view)

    Render()
    SaveScreenshot(output_image_path, view, ImageResolution=[1200, 1200], TransparentBackground=0)


# 1. Define where to search for the VTI files
BASE_DIR = "./Example_Bentheimer/"
FILE     = "output_data.vti"

# 2. Recursively find all generated output_data.vti files
target_files = []
for root, dirs, files in os.walk(BASE_DIR):
    for file in files:
        if file == FILE:
            target_files.append(os.path.join(root, file))

if not target_files:
    print("No output_data.vti files found. Did you run the prediction script first?")
    exit()

print(f"Found {len(target_files)} VTI files. Beginning batch render...")

# 3. Render and save each file in its native directory
for vti_path in target_files:
    dir_name = os.path.dirname(vti_path)
    folder_name = os.path.basename(dir_name)

    out_image = os.path.join(dir_name, f"streamlines_{folder_name}.png")
    print(f" -> Rendering: {out_image}")

    plot_streamlines(
        vti_filepath=vti_path,
        output_image_path=out_image,
        show_colorbar=True
    )

print("\nBatch rendering complete!")
```



---
# Training Process

The training process can be execute for main_Train_subModel.py or main_Train_mainModel.py. Sub-models stands for training processes with end-to-end adjustments. Main-models are trained with sub-models fixed while others keep being adjusted.

## Usage and Execution

The training pipeline is controlled via command-line arguments and reads hyperparameters from structured `.json` configuration files.

### 1. Standard Training

To start a new training process, provide a JSON configuration file.
If no file is specified, the script defaults to `config.json` in the root directory.

**Example:**
```bash
python main_Train_subModel.py --config experiment_01.json
```

---

### 2. Resuming an Experiment

To resume a previous training session or reproduce an experiment, provide the target results directory.

This feature is particularly useful for:
- Splitting long training runs into multiple executions
- Protecting against system interruptions or crashes
- Implementing curriculum learning strategies

The script automatically loads the `metadata.json` generated during the original run and ignores any `--config` file.

**Example:**
```bash
python main_TrainModel.py --folder ../NN_Results/NN_Trainning_23_March_2026_01-46PM
```

---

## Configuration Parameters

The `config.json` file controls all aspects of the model, dataset handling, and training process.

### General Structure

The following variables controls all aspects of the model, dataset handling, and training process.

```json
{
    "model_name": "danny_z",
    "binary_input": true,
    "NN_dataset_folder": "../NN_Datasets/",
    "dataset_train_name": "Train_Dataset.h5",
    "dataset_valid_name": "Valid_Dataset.h5",
    "train_range": [0, 8],
    "valid_range": [0, 2],
    "batch_size": 8,
    "num_workers": 4,
    "num_threads": 18,
    "N_epochs": 100,
    "partial_epochs": 100,
    "patience": 50,
    "learning_rate": 0.0006,
    "earlyStopping_loss": "PRPE",
    "backPropagation_loss": "Corr_MSE",
    "optimizer": "ADAM",
    "weight_init": null,
    "seed": 42,
    "train_comment": "Description of the current experiment."
}
```

---

### Parameter Description

- **`model_name`**
  Specifies the neural network architecture. Available options:
  `'javier_z'`, `'danny_z'`, `'danny_y'`, `'danny_x'`, `'danny_zyxp'`.

- **`binary_input`**
  Defines the input representation:
  - `true`: binary solid/void geometry
  - `false`: distance transform or continuous representation

- **`NN_dataset_folder`**
  Directory containing the dataset files.

- **`dataset_train_name`**
  Training dataset filename (must exist inside `NN_dataset_folder`).

- **`dataset_valid_name`**
  Validation dataset filename (must exist inside `NN_dataset_folder`).

- **`train_range`**
  Index range used from the training dataset.
  If `null`, the full dataset is used.

- **`valid_range`**
  Index range used from the validation dataset.
  If `null`, the full dataset is used.

- **`batch_size`**
  Number of samples per batch (i.e., per weight update).

- **`N_epochs`**
  Maximum number of training epochs.

- **`partial_epochs`**
  Number of epochs executed per run.
  Enables splitting long training jobs into multiple executions.

- **`patience`**
  Early stopping threshold. Training stops if no improvement in the monitored metric occurs for this number of epochs.

- **`learning_rate`**
  Learning rate used by the optimizer.

- **`optimizer`**
  Optimization algorithm. Supported options:
  - `"ADAM"`
  - `"ADAMW"`
  - `"SGD"`

- **`backPropagation_loss`**
  Loss function used to compute gradients and update model weights.

- **`earlyStopping_loss`**
  Metric used to track validation performance and determine the best model.

- **`weight_init`**
  Weight initialization strategy. Options:
  - `null` (default initialization)
  - `"xavier"`
  - `"he"`
  - `"zeros"`

- **`seed`**
  Random seed for reproducibility across runs.

- **`train_comment`**
  Free-text description of the experiment. Stored for tracking and reproducibility.

---

## Data Handling

### Lazy Loading

Datasets must be provided in `.h5` (HDF5) format.

The `LazyDatasetTorch` class performs on-the-fly data loading to minimize RAM usage, making it suitable for large-scale datasets. Instead of preloading all data into memory, batches are dynamically loaded from disk during training.

Solid regions (where velocity and pressure are strictly zero) are preserved using a binary geometric mask. This prevents normalization artifacts from distorting physical boundaries during convolutional operations.

---

### Custom Datasets

Custom dataset classes can be used if needed.

To implement a different dataset pipeline, modify the dataset object definition in:

```bash
main_TrainModel.py
```

---
# Validation Process
The validation process assesses the model's ability to generalize to new, unseen geometries, ensuring it has learned the underlying physics of the flow rather than just memorizing training data. The models are tested on out-of-distribution (OOD) domains limited to $120^3$ voxels, including synthetic geometries (e.g., spherical/cylindrical pores and grains) and real micro-CT rock images (e.g., Parker, Leopard, Kirby, Brown, Upper Gray, Sinter Gray, Bentheimer, Berea, Berea Buff, Castlegate, Bandera).

## Quantitative analysis
The quantitative evaluation relies on several physical and statistical metrics computed voxel-by-voxel or spatially averaged, comparing the Neural Network surrogate predictions against the Lattice Boltzmann Method (LBM) baselines:

* **Permeability Error ($e_k$)**: Evaluates the relative error in the predicted macroscopic permeability by comparing the spatial average of the velocity in the main flow direction.
* **Flow Residual ($e_f$)**: Measures global mass conservation by computing the L1 error of the flux across planes in the $x$, $y$, and $z$ directions.
* **Residual Divergence ($e_d$)**: Acts as a metric for point-wise mass conservation by evaluating the divergence of the predicted velocity field.
* **Tortuosity Error ($e_t$)**: Calculates the discrepancy in the predicted tortuosity of the flow pathways, a vital property for rock characterization.
* **Pearson Correlation Coefficient ($\sigma$)**: Statistically evaluates the spatial coherence and linear correlation between the predicted and true velocity fields.
* **Magnitude Error ($e_m$)**: Measures the local absolute error exclusively in regions where the velocity is above the sample's average, focusing on the main fluid channels.
* **Angular Error ($e_\theta$)**: Evaluates the directional alignment by computing the angle between the 3D velocity vectors of the prediction and the ground truth.

## Qualitative analysis
The qualitative analysis involves a visual inspection of the 3D velocity fields to determine the model's physical coherence:

* **Frontal Views**: Used to analyze the interaction between the fluid and the solid matrix, particularly checking boundary conditions at the walls.
* **Superior (Top) Views**: Evaluated to observe the continuity of the flow, making it easier to identify the model's handling of preferred flow pathways, constrictions, bifurcations, and ramifications.

