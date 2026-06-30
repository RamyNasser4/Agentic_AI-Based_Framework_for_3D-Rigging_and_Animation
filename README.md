# Generator Panel: AI-Assisted Animation and Rigging Add-on for Blender

## Overview

Generator Panel is a Blender add-on for AI-assisted animation and rigging workflows. It combines scene/object analysis, LLM-based animation planning, keyframe instruction generation, Blender Python execution, and an optional ComfyUI + UniRig rigging workflow.

The system helps users animate existing armatures or rig mesh objects through a modular workflow inside Blender.

## Architecture

```text
User Prompt / Selected Object
 |
 v
Blender Add-on UI
 |
 v
Scene Analysis
 |
 v
LLM Animation Planner
 |
 v
Keyframe Instruction Generator
 |
 v
Blender Python API
 |
 +-- Animation Execution
 +-- Rigging Mode via ComfyUI + UniRig
```

## Core Features

- Blender add-on interface in the 3D View sidebar.
- Animate and Rig modes for armature animation and mesh rigging workflows.
- Natural-language prompt input for animation generation.
- Selected object and scene hierarchy analysis.
- LLM-based animation planning using LangChain and Google Generative AI integration.
- Keyframe generation and execution through the Blender Python API.
- Prompt refinement, direction inference, preview generation, accept, and cancel workflow for animation.
- Rigging mode integration with local ComfyUI and UniRig.
- GLB mesh export to ComfyUI input and FBX rig import back into Blender.
- Imported rigged mesh smoothing with smooth shading, weighted normals, corrective smoothing, and optional subdivision.
- Modular Python files for planning, keyframe generation, execution, preview management, ComfyUI communication, and utility logic.

## Repository Structure

```text
GP/
|-- __init__.py                         # Blender add-on registration, UI panel, operators, modes, and scene properties
|-- planner_agent.py                    # LLM animation planner and plan validation helpers
|-- keyframe_agent.py                   # Converts animation plans into keyframe instructions
|-- animation_keyframe_agent.py         # Standalone/agent-style keyframe generation utilities
|-- Blender_Executer.py                 # Parses and applies generated keyframes to Blender armatures
|-- SceneParser.py                      # Serializes selected Blender objects and armature hierarchy
|-- comfy_unirig_client.py              # ComfyUI/UniRig client, GLB export, workflow patching, FBX import, mesh smoothing
|-- unirig_api.json                     # ComfyUI workflow template for UniRig
|-- mesh_renderer.py                    # Renders selected objects/views for visual AI workflows
|-- multimodal_utils.py                 # Image compression and JSON/text response helpers for multimodal LLM use
|-- prompt_refinement_agent.py          # Clarification and prompt refinement flow
|-- direction_inference_agent.py        # Visual direction-axis inference helper
|-- critic_agent.py                     # Critic pass for generated animation previews
|-- plan_fix_generator.py               # Generates fixes for animation plans from critic feedback
|-- plan_patcher.py                     # Applies plan edits
|-- preview_manager.py                  # Preview collection/action management
|-- skeleton_recorder.py                # Skeleton state recording helpers
|-- skeleton_visualizer.py              # Skeleton visualization helpers
|-- litellm-config.yaml                 # LiteLLM model configuration
|-- common/                             # Shared skeleton and quaternion utilities
|-- utils/                              # BVH, transform, rotation, and human model utilities
|-- body_models/, HumanML3D/, stats/    # Motion/model data folders used by related motion-processing scripts
|-- bvh_output/, motion_data/, texts/   # Generated or dataset-style motion/text assets
`-- *.blend, *.glb, *.bvh, *.ipynb      # Blender scenes, model assets, BVH files, and notebooks present in the repo
```

Python dependencies are listed in `requirements.txt` and should be installed into Blender's bundled Python environment.

## Tech Stack

| Layer | Technologies |
| --- | --- |
| Blender Add-on | Python, `bpy`, Blender 4.x |
| AI Planning | LangChain, Google Generative AI, LLM APIs |
| Prompt and Vision Helpers | LangChain messages, multimodal image utilities, OpenCV, NumPy |
| Animation Execution | Blender Python API, armatures, pose bones, keyframes |
| Rigging | ComfyUI, UniRig |
| 3D Processing | GLB export, FBX import, Blender mesh modifiers |
| Configuration | Environment variables, `.env`, local ComfyUI paths |

## Installation Steps

1. Open Blender 4.x.
2. Make sure the add-on folder is placed inside Blender's `scripts/addons` directory.
3. Open Blender Preferences, go to Add-ons, and enable the add-on named `Generator Panel`.
4. Before launching Blender, set the required environment variables, especially `GOOGLE_API_KEY`. This is required if the AI modules are used.
5. For Rigging Mode, start ComfyUI locally and make sure the UniRig nodes are installed correctly.
6. Check the ComfyUI input and output paths inside `comfy_unirig_client.py`. These paths may be machine-specific and might need to be updated depending on the local setup.
7. For additional ComfyUI-UniRig installation instructions, check: <https://github.com/PozzettiAndrea/ComfyUI-UniRig>
8. Open a Blender scene that contains either an armature for Animate Mode or a mesh object for Rig Mode.

Example environment variable setup:

```powershell
$env:GOOGLE_API_KEY="your-google-api-key"
```

## Installing Python Dependencies

Install the packages with Blender's bundled Python executable, not the system Python. Replace `<BLENDER_VERSION>` with the installed Blender version, such as `4.5`, `5.0`, or `5.1`. Replace `PATH_TO_PROJECT` with the path to this repository.

```powershell
cd "C:\Program Files\Blender Foundation\Blender <BLENDER_VERSION>\<BLENDER_VERSION>\python\bin"
.\python.exe -m ensurepip
.\python.exe -m pip install --upgrade pip
.\python.exe -m pip install -r "PATH_TO_PROJECT\requirements.txt"
```

## Quick Start

1. Start Blender.
2. Enable the `Generator Panel` add-on.
3. Open the `Generator` tab in the 3D View sidebar.
4. Select the needed mode:
   - `Animate` for an existing armature.
   - `Rig` for a mesh object that should be sent to UniRig.
5. Select the target scene object.
6. Enter an animation prompt or configure the rigging options.
7. Run the generation process.
8. Review the generated preview, animation, or imported rigged result in the Blender scene.

## Configuration

- `GOOGLE_API_KEY` is used by the Google Generative AI powered LLM modules.
- `.env` files are supported by modules that call `load_dotenv()`.
- `comfy_unirig_client.py` defines:
  - `COMFYUI_SERVER`
  - `COMFYUI_INPUT_DIR`
  - `COMFYUI_OUTPUT_DIR`
  - `WORKFLOW_PATH`
- The ComfyUI input and output directories are local machine paths and should be updated for the user's ComfyUI installation.
- `unirig_api.json` is the workflow template patched at runtime with the exported mesh path, output name, skeleton template, and target face count.

## Notes / Limitations

- ComfyUI must be running locally for Rigging Mode.
- UniRig nodes must be installed separately in ComfyUI.
- Some file paths in `comfy_unirig_client.py` may be machine-specific.
- Animation generation depends on the input prompt, selected object hierarchy, semantic direction settings, and available rig structure.
- Rigging quality depends on the input mesh, selected UniRig skeleton template, target face count, and the local ComfyUI/UniRig setup.
- Python dependencies must be installed into Blender's bundled Python environment.

## License

This project is licensed under the MIT License. See the LICENSE file for details.
