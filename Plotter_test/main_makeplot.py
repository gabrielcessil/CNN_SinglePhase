from paraview_viz import CameraManager, Panel, SceneRenderer

config = {
    'resolution': [1200, 1200],
    'scalar_name': 'Density',
    'vector_name': 'Velocity',
    'solid_ambient': [0.6, 0.5804, 0.6706],
    'solid_diffuse': [0.6, 0.5804, 0.6706],
}

camera_mgr = CameraManager(
    camera_pos=[-145.62, 258.12, 337.27],
    focal_point=[54.23, 49.59, 63.19],
    view_up=[0.28, 0.85, -0.44]
)

# Define panels similar to your original 3-image workflow[cite: 1]
panels = [
    Panel("Solid Geometry", show_solid=True, solid_opacity=1.0, show_flow=False),
    Panel("Predicted Flow", show_solid=True, solid_opacity=0.25, show_flow=True, flow_source='pred'),
    Panel("Simulated Flow", show_solid=False, show_flow=True, flow_source='target')
]

renderer = SceneRenderer(config)
sources = {'target': './output_single_frame/target_sample.vti', 'pred': './output_single_frame/pred_sample.vti'}

# Render one shot composite
renderer.render_panels(
    data_sources=sources,
    panels=panels,
    camera_params=camera_mgr.get_scaled_camera([-10, 120, -10, 120, -10, 120]), # Pass sample bounds here
    output_filepath='./output_single_frame/static_comparison.png'
)