# vuer-mujoco-demos

Demo examples for using Vuer with MuJoCo physics simulation.

## Description

This repository provides demos of Vuer and examples for testing the integration of Vuer with MuJoCo for robotics and physics simulation visualization.

## Getting Started

### Installation

To install the virtual environment with all dependencies:
```bash
uv sync
```

This will create a `.venv` directory and install all required packages from `pyproject.toml`.

### Running Demos

Example scenes are provided in the `scenes/` folder and can be loaded using `load_scene.py`.

To load a scene:
```bash
uv run load_scene.py
```

`uv` will automatically handle all dependencies for you.

### Interactive Controls

Once a scene is loaded, you can interact with it using the following controls:

- **Reset scene**: Click the cube or diamond shape
  - Cube: Resets and saves the current trajectory
  - Diamond: Resets without saving (delete trajectory)
- **Record initial position**: Click the red circle to save the current pose as an initial keyframe

## License

TBD
