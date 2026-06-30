import copy
import json
import re
import threading
import time
import urllib.error
import urllib.request
import uuid
from pathlib import Path


COMFYUI_SERVER = "http://127.0.0.1:8188"
COMFYUI_INPUT_DIR = Path(
    r"C:\Users\Khali\AppData\Local\Comfy-Desktop\ComfyUI-Installs\ComfyUI\ComfyUI\input"
)
COMFYUI_OUTPUT_DIR = Path(
    r"C:\Users\Khali\AppData\Local\Comfy-Desktop\ComfyUI-Installs\ComfyUI\ComfyUI\output"
)
WORKFLOW_PATH = Path(__file__).with_name("unirig_api.json")
_UNIRIG_RUN_LOCK = threading.Lock()
ALLOWED_SKELETON_TEMPLATES = {"articulationxl", "mixamo"}


def safe_filename(name):
    clean_name = re.sub(r"[^A-Za-z0-9_-]", "_", str(name or "").strip())
    return clean_name or "untitled"


def _notify_status(status_callback, message):
    print(message)
    if status_callback is None:
        return

    def _update_status():
        try:
            status_callback(message)
        except Exception as error:
            print(f"Failed to update UniRig status: {error}")
        return None

    try:
        import bpy  # type: ignore

        bpy.app.timers.register(_update_status, first_interval=0.0)
    except Exception as error:
        print(f"Failed to schedule UniRig status update: {error}")


def _notify_finished(finished_callback, success, message):
    if finished_callback is None:
        return

    def _finish():
        try:
            finished_callback(success, message)
        except Exception as error:
            print(f"Failed to update UniRig completion state: {error}")
        return None

    try:
        import bpy  # type: ignore

        bpy.app.timers.register(_finish, first_interval=0.0)
    except Exception as error:
        print(f"Failed to schedule UniRig completion update: {error}")


def request_json(url, payload=None, timeout=30):
    data = None
    headers = {"Accept": "application/json"}

    if payload is not None:
        data = json.dumps(payload).encode("utf-8")
        headers["Content-Type"] = "application/json"

    request = urllib.request.Request(url, data=data, headers=headers)

    try:
        with urllib.request.urlopen(request, timeout=timeout) as response:
            body = response.read().decode("utf-8")
    except urllib.error.HTTPError as error:
        try:
            error_body = error.read().decode("utf-8", errors="replace")
        except Exception:
            error_body = ""
        raise RuntimeError(
            f"ComfyUI HTTP error {error.code} for {url}: {error.reason}. {error_body}"
        ) from error
    except urllib.error.URLError as error:
        raise RuntimeError(f"Could not reach ComfyUI at {url}: {error.reason}") from error

    if not body.strip():
        return None

    try:
        return json.loads(body)
    except json.JSONDecodeError as error:
        raise RuntimeError(f"ComfyUI returned invalid JSON from {url}: {body}") from error


def check_comfyui_running():
    try:
        request_json(f"{COMFYUI_SERVER}/system_stats", timeout=5)
    except Exception as error:
        raise RuntimeError(
            "ComfyUI is not running. Open ComfyUI Desktop and keep it running at "
            "127.0.0.1:8188."
        ) from error


def queue_prompt(workflow):
    response = request_json(
        f"{COMFYUI_SERVER}/prompt",
        {
            "prompt": workflow,
            "client_id": str(uuid.uuid4()),
        },
    )
    prompt_id = response.get("prompt_id") if isinstance(response, dict) else None
    if not prompt_id:
        raise RuntimeError(f"ComfyUI did not return a prompt_id. Response: {response}")
    return prompt_id


def wait_for_history(prompt_id, timeout_seconds=1800):
    deadline = time.time() + timeout_seconds
    url = f"{COMFYUI_SERVER}/history/{prompt_id}"

    while time.time() < deadline:
        history = request_json(url, timeout=30)
        if isinstance(history, dict) and prompt_id in history:
            return history[prompt_id]
        time.sleep(2)

    raise RuntimeError(f"Timed out waiting for UniRig prompt {prompt_id} after {timeout_seconds} seconds.")


def _bbox_info_from_world_points(points, object_name=""):
    from mathutils import Vector  # type: ignore

    if not points:
        return None

    min_corner = Vector(
        (
            min(point.x for point in points),
            min(point.y for point in points),
            min(point.z for point in points),
        )
    )
    max_corner = Vector(
        (
            max(point.x for point in points),
            max(point.y for point in points),
            max(point.z for point in points),
        )
    )
    dimensions = max_corner - min_corner
    center = (min_corner + max_corner) / 2
    max_dimension = max(dimensions.x, dimensions.y, dimensions.z)

    return {
        "object_name": object_name,
        "center": center,
        "dimensions": dimensions,
        "max_dimension": max_dimension,
    }


def get_world_bbox_info(obj):
    from mathutils import Vector  # type: ignore

    world_points = [obj.matrix_world @ Vector(corner) for corner in obj.bound_box]
    bbox_info = _bbox_info_from_world_points(world_points, object_name=obj.name)
    if bbox_info is None:
        return None

    bbox_info.update(
        {
            "matrix_world": obj.matrix_world.copy(),
            "location": obj.location.copy(),
            "rotation_euler": obj.rotation_euler.copy() if hasattr(obj, "rotation_euler") else None,
            "scale": obj.scale.copy(),
        }
    )
    return bbox_info


def get_objects_world_bbox_info(objects):
    from mathutils import Vector  # type: ignore

    world_points = []
    for obj in objects:
        if getattr(obj, "type", None) != "MESH":
            continue
        world_points.extend(obj.matrix_world @ Vector(corner) for corner in obj.bound_box)

    return _bbox_info_from_world_points(world_points)


def export_object_to_comfy_input(object_name, status_callback=None):
    import bpy  # type: ignore

    obj = bpy.data.objects.get(object_name)
    if obj is None:
        raise RuntimeError(f"Object `{object_name}` was not found.")
    if getattr(obj, "type", None) != "MESH":
        raise RuntimeError(f"Object `{object_name}` is not a mesh.")

    original_bbox_info = get_world_bbox_info(obj)
    if original_bbox_info is None:
        raise RuntimeError(f"Could not calculate world bounding box for `{object_name}`.")
    print(f"[UniRig] Original bbox dimensions: {original_bbox_info['dimensions']}")

    clean_name = safe_filename(obj.name)
    unique_id = uuid.uuid4().hex[:8]
    export_dir = COMFYUI_INPUT_DIR / "3d"
    export_dir.mkdir(parents=True, exist_ok=True)
    export_path = export_dir / f"{clean_name}_{unique_id}.glb"

    view_layer = bpy.context.view_layer
    previous_active = view_layer.objects.active
    previous_selected = [item for item in bpy.context.selected_objects]

    try:
        for selected in previous_selected:
            selected.select_set(False)
        obj.select_set(True)
        view_layer.objects.active = obj
        bpy.ops.export_scene.gltf(
            filepath=str(export_path),
            export_format="GLB",
            use_selection=True,
            export_apply=True,
        )
    finally:
        obj.select_set(False)
        for selected in previous_selected:
            if selected.name in bpy.data.objects:
                selected.select_set(True)
        if previous_active and previous_active.name in bpy.data.objects:
            view_layer.objects.active = previous_active

    if not export_path.exists() or export_path.stat().st_size <= 0:
        raise RuntimeError(f"GLB export failed. Expected exported file at: {export_path}")

    export_size = export_path.stat().st_size
    mesh_relative_path = f"3d/{export_path.name}"
    print(f"[UniRig] Exported selected mesh to: {mesh_relative_path}")
    print(f"[UniRig] Export path: {export_path}")
    print(f"[UniRig] Export size: {export_size}")
    _notify_status(status_callback, f"[UniRig] Exported GLB: {mesh_relative_path}")

    return mesh_relative_path, clean_name, unique_id, original_bbox_info


def _find_node_by_class_type(workflow, class_type):
    for node in workflow.values():
        if isinstance(node, dict) and node.get("class_type") == class_type:
            return node
    return None


def validate_skeleton_template(skeleton_template):
    if skeleton_template not in ALLOWED_SKELETON_TEMPLATES:
        raise RuntimeError(f"Unsupported skeleton template: {skeleton_template}")
    return skeleton_template


def patch_unirig_workflow(
    template_workflow,
    mesh_relative_path,
    fbx_name,
    skeleton_template="articulationxl",
    target_face_count=10000,
):
    skeleton_template = validate_skeleton_template(skeleton_template)
    target_face_count = int(target_face_count)
    workflow = copy.deepcopy(template_workflow)

    load_mesh_node = _find_node_by_class_type(workflow, "UniRigLoadMesh")
    if load_mesh_node is None:
        load_mesh_node = workflow.get("5")

    auto_rig_node = _find_node_by_class_type(workflow, "UniRigAutoRig")
    if auto_rig_node is None:
        auto_rig_node = workflow.get("2")

    load_mesh_patched = False
    auto_rig_patched = False

    if isinstance(load_mesh_node, dict):
        inputs = load_mesh_node.setdefault("inputs", {})
        inputs["source_folder"] = "input"
        inputs["file_path"] = mesh_relative_path
        load_mesh_patched = True

    if isinstance(auto_rig_node, dict):
        inputs = auto_rig_node.setdefault("inputs", {})
        inputs["fbx_name"] = fbx_name
        inputs["skeleton_template"] = skeleton_template
        inputs["target_face_count"] = target_face_count
        auto_rig_patched = True

    if not load_mesh_patched:
        raise RuntimeError("Could not patch UniRigLoadMesh node in UniRig workflow.")
    if not auto_rig_patched:
        raise RuntimeError("Could not patch UniRigAutoRig node in UniRig workflow.")

    return workflow


def find_generated_fbx(fbx_name, started_at, skeleton_template="articulationxl"):
    skeleton_template = validate_skeleton_template(skeleton_template)
    expected_path = COMFYUI_OUTPUT_DIR / f"{fbx_name}_{skeleton_template}.fbx"
    try:
        if expected_path.exists():
            stat = expected_path.stat()
            if stat.st_mtime >= started_at and stat.st_size > 0:
                return expected_path
    except OSError:
        pass

    if COMFYUI_OUTPUT_DIR.exists():
        matches = []
        for path in COMFYUI_OUTPUT_DIR.rglob(f"{fbx_name}*.fbx"):
            try:
                stat = path.stat()
                if stat.st_mtime >= started_at and stat.st_size > 0:
                    matches.append(path)
            except OSError:
                pass

        if matches:
            return max(matches, key=lambda path: path.stat().st_mtime)

    raise RuntimeError("UniRig finished but no new FBX was created for this run.")


def delete_stale_expected_fbx(fbx_name, skeleton_template="articulationxl"):
    skeleton_template = validate_skeleton_template(skeleton_template)
    expected_path = COMFYUI_OUTPUT_DIR / f"{fbx_name}_{skeleton_template}.fbx"
    if not expected_path.exists():
        return

    try:
        expected_path.unlink()
        print(f"[UniRig] Deleted stale output FBX: {expected_path}")
    except OSError as error:
        raise RuntimeError(f"Could not delete stale FBX before run `{expected_path}`: {error}") from error


def _find_imported_root(imported_objects):
    for obj in imported_objects:
        if getattr(obj, "type", None) == "ARMATURE":
            return obj

    for obj in imported_objects:
        if getattr(obj, "parent", None) is None:
            return obj

    return imported_objects[0] if imported_objects else None


def smooth_imported_meshes(imported_objects, add_subdivision=True):
    for obj in imported_objects:
        if getattr(obj, "type", None) != "MESH":
            continue

        for poly in obj.data.polygons:
            poly.use_smooth = True

        try:
            obj.data.update()
        except Exception:
            pass

        if not any(mod.type == "WEIGHTED_NORMAL" for mod in obj.modifiers):
            modifier = obj.modifiers.new("UniRig Weighted Normals", "WEIGHTED_NORMAL")
            if hasattr(modifier, "keep_sharp"):
                modifier.keep_sharp = False
            if hasattr(modifier, "weight"):
                modifier.weight = 50

        if not any(mod.type == "CORRECTIVE_SMOOTH" for mod in obj.modifiers):
            corrective = obj.modifiers.new("UniRig Corrective Smooth", "CORRECTIVE_SMOOTH")
            if hasattr(corrective, "factor"):
                corrective.factor = 0.25
            if hasattr(corrective, "iterations"):
                corrective.iterations = 2

        if add_subdivision and not any(mod.type == "SUBSURF" for mod in obj.modifiers):
            subdivision = obj.modifiers.new("UniRig Smooth Subdivision", "SUBSURF")
            subdivision.levels = 1
            subdivision.render_levels = 1

        print(f"[UniRig] Smoothed imported mesh: {obj.name}")


def import_fbx_on_main_thread(fbx_path, original_bbox_info=None, add_subdivision=True):
    import bpy  # type: ignore

    def _import_fbx():
        try:
            before_objects = set(bpy.data.objects)
            bpy.ops.import_scene.fbx(filepath=str(fbx_path))
            imported_objects = [obj for obj in bpy.data.objects if obj not in before_objects]
            smooth_imported_meshes(imported_objects, add_subdivision=add_subdivision)
            print(f"Imported rigged FBX: {fbx_path}")
            if original_bbox_info is not None:
                root = _find_imported_root(imported_objects)
                imported_bbox_info = get_objects_world_bbox_info(imported_objects)
                if root is None or imported_bbox_info is None:
                    print("[UniRig] Could not align imported rig because no imported mesh bbox was found.")
                    return None

                original_max_dimension = original_bbox_info.get("max_dimension", 0)
                imported_max_dimension = imported_bbox_info.get("max_dimension", 0)
                print(f"[UniRig] Imported bbox dimensions before scale: {imported_bbox_info['dimensions']}")

                if original_max_dimension <= 0 or imported_max_dimension <= 0:
                    print("[UniRig] Could not align imported rig because bbox dimensions were empty.")
                    return None

                scale_factor = original_max_dimension / imported_max_dimension
                root.scale = root.scale * scale_factor
                bpy.context.view_layer.update()
                print(f"[UniRig] Scale factor applied: {scale_factor}")

                imported_bbox_info = get_objects_world_bbox_info(imported_objects)
                if imported_bbox_info is None:
                    print("[UniRig] Could not recompute imported rig bbox after scaling.")
                    return None

                delta = original_bbox_info["center"] - imported_bbox_info["center"]
                root.location += delta
                bpy.context.view_layer.update()
                print("[UniRig] Imported rig repositioned to original center.")
        except Exception as error:
            print(f"Failed to import rigged FBX `{fbx_path}`: {error}")
        return None

    bpy.app.timers.register(_import_fbx, first_interval=0.1)


def _load_workflow_template():
    if not WORKFLOW_PATH.exists():
        raise RuntimeError(f"UniRig workflow template is missing: {WORKFLOW_PATH}")

    with WORKFLOW_PATH.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def _call_on_blender_main_thread(callback):
    import bpy  # type: ignore

    event = threading.Event()
    result = {}

    def _timer_callback():
        try:
            result["value"] = callback()
        except Exception as error:
            result["error"] = error
        finally:
            event.set()
        return None

    bpy.app.timers.register(_timer_callback, first_interval=0.0)
    event.wait()

    if "error" in result:
        raise result["error"]
    return result.get("value")


def run_unirig_for_object(
    object_name,
    output_name="",
    scene_name=None,
    skeleton_template="articulationxl",
    target_face_count=10000,
    add_subdivision=True,
    status_callback=None,
    finished_callback=None,
):
    def _worker():
        success = False
        final_message = "UniRig failed."
        try:
            with _UNIRIG_RUN_LOCK:
                selected_skeleton_template = validate_skeleton_template(skeleton_template)
                selected_target_face_count = int(target_face_count)
                _notify_status(status_callback, "[UniRig] Checking ComfyUI...")
                check_comfyui_running()
                _notify_status(status_callback, "[UniRig] Exporting selected mesh...")
                mesh_relative_path, clean_name, unique_id, original_bbox_info = _call_on_blender_main_thread(
                    lambda: export_object_to_comfy_input(object_name, status_callback=status_callback)
                )

                requested_output_name = str(output_name or "").strip()
                if requested_output_name:
                    fbx_name = safe_filename(f"{safe_filename(requested_output_name)}_{unique_id}")
                else:
                    fbx_name = safe_filename(f"{clean_name}_rigged_{unique_id}")

                print(f"[UniRig] Requested output name: {requested_output_name}")
                print(f"[UniRig] Actual FBX name: {fbx_name}")
                print(f"[UniRig] Skeleton template: {selected_skeleton_template}")
                print(f"[UniRig] Target face count: {selected_target_face_count}")
                delete_stale_expected_fbx(fbx_name, skeleton_template=selected_skeleton_template)
                template_workflow = _load_workflow_template()
                workflow = patch_unirig_workflow(
                    template_workflow,
                    mesh_relative_path,
                    fbx_name,
                    skeleton_template=selected_skeleton_template,
                    target_face_count=selected_target_face_count,
                )

                _notify_status(status_callback, "[UniRig] Running UniRig in ComfyUI...")
                print("Sending UniRig workflow to ComfyUI...")
                started_at = time.time()
                prompt_id = queue_prompt(workflow)
                print(f"Prompt ID: {prompt_id}")
                print("Waiting for UniRig...")
                wait_for_history(prompt_id)
                fbx_path = find_generated_fbx(
                    fbx_name,
                    started_at,
                    skeleton_template=selected_skeleton_template,
                )
                print(f"Generated FBX found: {fbx_path}")
                import_fbx_on_main_thread(
                    fbx_path,
                    original_bbox_info=original_bbox_info,
                    add_subdivision=add_subdivision,
                )
                success = True
                final_message = f"UniRig completed. Generated FBX: {fbx_path}"
                _notify_status(status_callback, f"[UniRig] Completed: {fbx_path}")
        except Exception as error:
            final_message = str(error)
            _notify_status(status_callback, f"[UniRig] Failed: {error}")
            print(f"UniRig ComfyUI error: {error}")
        finally:
            _notify_finished(finished_callback, success, final_message)

    thread = threading.Thread(target=_worker, name="GradUniRigComfyUI", daemon=True)
    thread.start()
    return thread
