import os
import sys
import gc
import json
import numpy as np

# Conditional imports only when running as a ParaView worker
if '--worker' in sys.argv:
    from paraview.simple import *
    import matplotlib.pyplot as plt
    import matplotlib.image as mpimg

class ParaViewWorkerRenderer:
    def __init__(self, config):
        self.config = config
        self.view = None

    def _setup_view(self, resolution):
        for v in GetRenderViews():
            try: Delete(v)
            except: pass
        gc.collect()

        self.view = CreateRenderView()
        self.view.ViewSize = resolution
        self.view.Background = [1.0, 1.0, 1.0]
        self.view.Background2 = [1.0, 1.0, 1.0]
        self.view.OrientationAxesVisibility = 0
        self.view.EnableRayTracing = 1
        self.view.SamplesPerPixel = self.config.get('samples', 4)
        self.view.AmbientSamples = self.config.get('ambient_samples', 6)
        self.view.BackEnd = "OSPRay pathtracer"
        self.view.Shadows = 1
        self.view.Denoise = 1

    def render_job(self, job_data):
        resolution = self.config.get('resolution', [1200, 1200])
        self._setup_view(resolution)
        objects_to_delete = []

        data_sources = job_data['data_sources']
        panels_config = job_data['panels']
        camera_params = job_data['camera_params']
        output_filepath = job_data['output_filepath']

        readers = {}
        for key, path in data_sources.items():
            r = XMLImageDataReader(FileName=[path])
            r.PointArrayStatus = [self.config['scalar_name'], self.config['vector_name']]
            r.UpdatePipeline()
            readers[key] = r
            objects_to_delete.append(r)

        ref_reader = readers.get('target', list(readers.values())[0])
        bounds = ref_reader.GetDataInformation().GetBounds()
        xmin, xmax, ymin, ymax, zmin, zmax = bounds
        center = [0.5 * (xmin + xmax), 0.5 * (ymin + ymax), 0.5 * (zmin + zmax)]

        thresh = Threshold(Input=ref_reader)
        thresh.Scalars = ['POINTS', self.config['scalar_name']]
        thresh.ThresholdMethod = 'Between'
        thresh.LowerThreshold = -1e10
        thresh.UpperThreshold = 0.0

        clips = []
        for normal in [[-1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]]:
            c = Clip(Input=thresh)
            c.ClipType = "Plane"
            c.ClipType.Normal = normal
            c.ClipType.Origin = center
            clips.append(c)
        objects_to_delete.extend([thresh] + clips)

        solid_displays = []
        for clip_obj in clips:
            disp = Show(clip_obj, self.view)
            disp.Representation = 'Surface'
            disp.AmbientColor = self.config['solid_ambient']
            disp.DiffuseColor = self.config['solid_diffuse']
            disp.Specular = 0.3
            disp.OSPRayMaterial = "None"
            solid_displays.append(disp)

        flow_displays = {}
        for key, reader in readers.items():
            stream = StreamTracer(Input=reader, SeedType="Point Cloud")
            stream.Vectors = ["POINTS", self.config['vector_name']]
            stream.MaximumStreamlineLength = 1000.0
            stream.SeedType.Center = center
            stream.SeedType.Radius = (xmax - xmin)
            stream.SeedType.NumberOfPoints = 20000
            
            disp_stream = Show(stream, self.view)
            ColorBy(disp_stream, ("POINTS", self.config['vector_name'], "Magnitude"))
            disp_stream.RenderLinesAsTubes = 1
            disp_stream.LineWidth = 0.5
            disp_stream.Ambient = 1.0
            disp_stream.Diffuse = 0.0
            disp_stream.OSPRayMaterial = "None"

            disp_vol = Show(reader, self.view)
            disp_vol.Representation = "Volume"
            disp_vol.Opacity = 0.8
            disp_vol.Shade = 0
            disp_vol.Ambient = 1.0
            disp_vol.Diffuse = 0.0
            ColorBy(disp_vol, ("POINTS", self.config['vector_name'], "Magnitude"))
            
            flow_displays[key] = (stream, disp_stream, disp_vol)
            objects_to_delete.append(stream)

        self.view.CameraPosition = camera_params['camera_pos']
        self.view.CameraFocalPoint = camera_params['focal_point']
        self.view.CameraViewUp = camera_params['camera_view_up']
        self.view.CameraViewAngle = camera_params['camera_view_angle']
        self.view.CameraParallelScale = camera_params['camera_parallel_scale']
        self.view.Update()

        temp_files = []
        dirname, basename = os.path.split(output_filepath)
        name, ext = os.path.splitext(basename)

        for idx, panel in enumerate(panels_config):
            for d in solid_displays:
                d.Visibility = 1 if panel['show_solid'] else 0
                d.Opacity = panel['solid_opacity']

            for k, (s, d_stream, d_vol) in flow_displays.items():
                active = (panel['show_flow'] and panel['flow_source'] == k)
                d_stream.Visibility = 1 if active else 0
                d_vol.Visibility = 1 if active else 0

            self.view.Shadows = 0 if panel['solid_opacity'] < 1.0 else 1
            
            temp_path = os.path.join(dirname, f"{name}_panel_{idx}{ext}")
            SaveScreenshot(temp_path, self.view, ImageResolution=resolution, OverrideColorPalette='WhiteBackground')
            temp_files.append((temp_path, panel['title']))

        if len(temp_files) > 1:
            fig, axes = plt.subplots(1, len(temp_files), figsize=(6 * len(temp_files), 6))
            if len(temp_files) == 1: axes = [axes]
            
            for ax, (path, title) in zip(axes, temp_files):
                ax.imshow(mpimg.imread(path))
                ax.set_title(title, fontsize=16, fontname="serif")
                ax.axis("off")
            plt.tight_layout()
            plt.savefig(output_filepath, dpi=150, bbox_inches="tight", facecolor="white")
            plt.close(fig)
            for path, _ in temp_files:
                if os.path.exists(path): os.remove(path)
        else:
            os.replace(temp_files[0][0], output_filepath)

        for obj in reversed(objects_to_delete):
            try: Delete(obj)
            except: pass
        Disconnect()
        gc.collect()

if __name__ == '__main__':
    if '--worker' in sys.argv:
        job_file = sys.argv[sys.argv.index('--worker') + 1]
        with open(job_file, 'r') as f:
            job_data = json.load(f)
        renderer = ParaViewWorkerRenderer(job_data['config'])
        renderer.render_job(job_data)