from __future__ import annotations

from typing import Dict, Iterable, List, Optional, Tuple

try:
    import bpy  # type: ignore
except ImportError:  # pragma: no cover - only available inside Blender
    bpy = None


PREVIEW_SCENE_NAME = "AI_Preview"
PREVIEW_ARMATURE_SUFFIX = "_AI_Preview"
PREVIEW_ACTION_SUFFIX = "_AI_Preview_Action"
PREVIEW_MARKER = "generator_preview_resource"


def _require_bpy():
    if bpy is None:
        raise RuntimeError("preview_manager must run inside Blender because bpy is unavailable.")
    return bpy


def _get_scene_prop(scene, name: str, default=""):
    return getattr(scene, name, default) if scene is not None else default


def _set_scene_prop(scene, name: str, value) -> None:
    if scene is not None and hasattr(scene, name):
        setattr(scene, name, value)


def _clear_preview_properties(scene) -> None:
    _set_scene_prop(scene, "preview_scene_name", "")
    _set_scene_prop(scene, "preview_armature_name", "")
    _set_scene_prop(scene, "preview_action_name", "")
    _set_scene_prop(scene, "preview_original_armature_name", "")
    _set_scene_prop(scene, "preview_active", False)


def _copy_scene_timing(source_scene, target_scene) -> None:
    if source_scene is None or target_scene is None:
        return

    target_scene.frame_start = source_scene.frame_start
    target_scene.frame_end = source_scene.frame_end
    target_scene.frame_current = source_scene.frame_current
    target_scene.render.fps = source_scene.render.fps
    target_scene.render.fps_base = source_scene.render.fps_base


def ensure_preview_scene(original_scene=None):
    bpy_module = _require_bpy()
    preview_scene = bpy_module.data.scenes.get(PREVIEW_SCENE_NAME)
    if preview_scene is None:
        print("[Preview] Creating preview scene")
        preview_scene = bpy_module.data.scenes.new(PREVIEW_SCENE_NAME)
    else:
        print("[Preview] Reusing preview scene")

    _copy_scene_timing(original_scene, preview_scene)
    return preview_scene


def _iter_preview_objects(preview_scene):
    for obj in list(getattr(preview_scene, "objects", [])):
        if obj.get(PREVIEW_MARKER) or PREVIEW_ARMATURE_SUFFIX in obj.name:
            yield obj


def _remove_data_block(data) -> None:
    if data is None or getattr(data, "users", 0) != 0:
        return

    data_type = getattr(getattr(data, "bl_rna", None), "identifier", "")
    if data_type == "Armature":
        bpy.data.armatures.remove(data)
    elif data_type == "Mesh":
        bpy.data.meshes.remove(data)


def _remove_action(action_name: str) -> None:
    if not action_name:
        return

    action = bpy.data.actions.get(action_name)
    if action is not None:
        bpy.data.actions.remove(action)


def cleanup_preview_resources(owner_scene=None, remove_scene: bool = False, original_scene=None) -> None:
    bpy_module = _require_bpy()
    print("[Preview] Removing preview resources")

    preview_scene_name = _get_scene_prop(owner_scene, "preview_scene_name", PREVIEW_SCENE_NAME) or PREVIEW_SCENE_NAME
    preview_scene = bpy_module.data.scenes.get(preview_scene_name)
    preview_armature_name = _get_scene_prop(owner_scene, "preview_armature_name", "")
    preview_action_name = _get_scene_prop(owner_scene, "preview_action_name", "")

    if preview_scene is not None:
        print("[Preview] Cleaning preview scene")
        objects_to_remove = list(_iter_preview_objects(preview_scene))
        if preview_armature_name:
            preview_armature = bpy_module.data.objects.get(preview_armature_name)
            if preview_armature is not None and preview_armature not in objects_to_remove:
                objects_to_remove.append(preview_armature)

        for obj in objects_to_remove:
            action = getattr(getattr(obj, "animation_data", None), "action", None)
            data = getattr(obj, "data", None)
            bpy_module.data.objects.remove(obj, do_unlink=True)
            _remove_data_block(data)
            if action is not None and action.users == 0:
                bpy_module.data.actions.remove(action)

    _remove_action(preview_action_name)

    for action in list(bpy_module.data.actions):
        if action.get(PREVIEW_MARKER):
            bpy_module.data.actions.remove(action)

    if remove_scene and preview_scene is not None:
        _restore_windows_from_preview(preview_scene, original_scene)
        if len(preview_scene.objects) == 0:
            bpy_module.data.scenes.remove(preview_scene)

    _clear_preview_properties(owner_scene)
    print("[Preview] Cleanup complete")


def _unique_preview_name(base_name: str) -> str:
    name = f"{base_name}{PREVIEW_ARMATURE_SUFFIX}"
    if bpy.data.objects.get(name) is None:
        return name

    index = 1
    while bpy.data.objects.get(f"{name}.{index:03d}") is not None:
        index += 1
    return f"{name}.{index:03d}"


def _make_preview_action(original_armature, preview_armature):
    print("[Preview] Creating preview action")
    source_action = getattr(getattr(original_armature, "animation_data", None), "action", None)
    if source_action is not None:
        action = source_action.copy()
        action.name = f"{preview_armature.name}{PREVIEW_ACTION_SUFFIX}"
    else:
        action = bpy.data.actions.new(name=f"{preview_armature.name}{PREVIEW_ACTION_SUFFIX}")

    action[PREVIEW_MARKER] = True
    preview_armature.animation_data_create()
    preview_armature.animation_data.action = action
    return action


def _iter_candidate_meshes(original_armature, original_scene=None) -> Iterable[object]:
    objects = getattr(original_scene, "objects", None) if original_scene is not None else None
    if objects is None:
        objects = bpy.data.objects

    for obj in objects:
        if getattr(obj, "type", None) == "MESH":
            yield obj


def _mesh_uses_armature_modifier(mesh_obj, original_armature) -> bool:
    for modifier in getattr(mesh_obj, "modifiers", []):
        if getattr(modifier, "type", None) == "ARMATURE" and getattr(modifier, "object", None) == original_armature:
            return True
    return False


def _find_linked_meshes(original_armature, original_scene=None) -> List[object]:
    print("[Preview] Searching for meshes linked to armature")
    linked_meshes = []

    for mesh_obj in _iter_candidate_meshes(original_armature, original_scene):
        is_parented = getattr(mesh_obj, "parent", None) == original_armature
        uses_modifier = _mesh_uses_armature_modifier(mesh_obj, original_armature)
        if is_parented or uses_modifier:
            linked_meshes.append(mesh_obj)

    print(f"[Preview] Found {len(linked_meshes)} linked mesh(es)")
    return linked_meshes


def _copy_visibility_settings(source_obj, preview_obj) -> None:
    for attr_name in (
        "hide_viewport",
        "hide_render",
        "hide_select",
        "visible_camera",
        "visible_diffuse",
        "visible_glossy",
        "visible_shadow",
        "visible_transmission",
        "visible_volume_scatter",
    ):
        if hasattr(source_obj, attr_name) and hasattr(preview_obj, attr_name):
            setattr(preview_obj, attr_name, getattr(source_obj, attr_name))

    if hasattr(source_obj, "hide_get") and hasattr(preview_obj, "hide_set"):
        preview_obj.hide_set(source_obj.hide_get())


def _rebind_armature_modifiers(preview_mesh, original_armature, preview_armature) -> None:
    for modifier in getattr(preview_mesh, "modifiers", []):
        if getattr(modifier, "type", None) == "ARMATURE" and getattr(modifier, "object", None) == original_armature:
            print("[Preview] Rebinding Armature modifier")
            modifier.object = preview_armature


def _copy_object_animation_data(preview_obj) -> None:
    action = getattr(getattr(preview_obj, "animation_data", None), "action", None)
    if action is None:
        return

    copied_action = action.copy()
    copied_action.name = f"{preview_obj.name}_Action"
    copied_action[PREVIEW_MARKER] = True
    preview_obj.animation_data.action = copied_action


def _copy_shape_key_animation_data(preview_mesh) -> None:
    shape_keys = getattr(getattr(preview_mesh, "data", None), "shape_keys", None)
    action = getattr(getattr(shape_keys, "animation_data", None), "action", None)
    if action is None:
        return

    copied_action = action.copy()
    copied_action.name = f"{preview_mesh.name}_ShapeKeys_Action"
    copied_action[PREVIEW_MARKER] = True
    shape_keys.animation_data.action = copied_action


def _duplicate_preview_mesh(original_mesh, preview_armature):
    print(f"[Preview] Copying mesh: {original_mesh.name}")
    preview_mesh = original_mesh.copy()
    preview_mesh.data = original_mesh.data.copy()
    preview_mesh.name = f"{original_mesh.name}{PREVIEW_ARMATURE_SUFFIX}"
    preview_mesh.data.name = f"{preview_mesh.name}_Data"
    preview_mesh[PREVIEW_MARKER] = True
    preview_mesh["preview_source_mesh"] = original_mesh.name
    preview_mesh.matrix_world = original_mesh.matrix_world.copy()
    preview_mesh.matrix_basis = original_mesh.matrix_basis.copy()
    preview_mesh.matrix_parent_inverse = original_mesh.matrix_parent_inverse.copy()
    _copy_visibility_settings(original_mesh, preview_mesh)
    _copy_object_animation_data(preview_mesh)
    _copy_shape_key_animation_data(preview_mesh)
    return preview_mesh


def _restore_preview_parenting(original_mesh, preview_mesh, preview_armature, mesh_map: Dict[object, object]) -> None:
    original_parent = getattr(original_mesh, "parent", None)
    if original_parent is None:
        preview_mesh.parent = None
        return

    if getattr(original_parent, "type", None) == "ARMATURE":
        print("[Preview] Parenting mesh to preview armature")
        preview_mesh.parent = preview_armature
        preview_mesh.parent_type = original_mesh.parent_type
        preview_mesh.parent_bone = original_mesh.parent_bone
        preview_mesh.matrix_parent_inverse = original_mesh.matrix_parent_inverse.copy()
        return

    if original_parent in mesh_map:
        preview_mesh.parent = mesh_map[original_parent]
        preview_mesh.parent_type = original_mesh.parent_type
        preview_mesh.parent_bone = original_mesh.parent_bone
        preview_mesh.matrix_parent_inverse = original_mesh.matrix_parent_inverse.copy()


def _copy_character_to_preview(original_armature, preview_armature, preview_scene, original_scene=None) -> List[object]:
    linked_meshes = _find_linked_meshes(original_armature, original_scene)
    preview_meshes = []
    mesh_map: Dict[object, object] = {}

    for original_mesh in linked_meshes:
        preview_mesh = _duplicate_preview_mesh(original_mesh, preview_armature)
        _rebind_armature_modifiers(preview_mesh, original_armature, preview_armature)
        mesh_map[original_mesh] = preview_mesh
        preview_meshes.append(preview_mesh)

    for original_mesh, preview_mesh in mesh_map.items():
        _restore_preview_parenting(original_mesh, preview_mesh, preview_armature, mesh_map)
        print("[Preview] Linking preview mesh")
        preview_scene.collection.objects.link(preview_mesh)

    print("[Preview] Preview character ready")
    return preview_meshes


def prepare_preview(context, original_armature, owner_scene=None):
    bpy_module = _require_bpy()
    if original_armature is None:
        raise ValueError("No armature was provided for preview generation.")
    if isinstance(original_armature, str):
        original_armature = bpy_module.data.objects.get(original_armature)
    if original_armature is None:
        raise ValueError("The original armature was not found.")
    if getattr(original_armature, "type", None) != "ARMATURE":
        raise TypeError(f"Object `{original_armature.name}` is not an armature.")

    owner_scene = owner_scene or context.scene
    preview_scene = ensure_preview_scene(owner_scene)
    cleanup_preview_resources(owner_scene, remove_scene=False, original_scene=owner_scene)
    preview_scene = ensure_preview_scene(owner_scene)

    print("[Preview] Duplicating armature")
    preview_armature = original_armature.copy()
    preview_armature.data = original_armature.data.copy()
    preview_armature.name = _unique_preview_name(original_armature.name)
    preview_armature.data.name = f"{preview_armature.name}_Data"
    preview_armature[PREVIEW_MARKER] = True
    preview_armature["preview_source_armature"] = original_armature.name
    preview_armature.matrix_world = original_armature.matrix_world.copy()
    preview_armature.animation_data_clear()

    preview_scene.collection.objects.link(preview_armature)
    preview_action = _make_preview_action(original_armature, preview_armature)
    _copy_character_to_preview(original_armature, preview_armature, preview_scene, owner_scene)

    _set_scene_prop(owner_scene, "preview_scene_name", preview_scene.name)
    _set_scene_prop(owner_scene, "preview_armature_name", preview_armature.name)
    _set_scene_prop(owner_scene, "preview_action_name", preview_action.name)
    _set_scene_prop(owner_scene, "preview_original_armature_name", original_armature.name)
    _set_scene_prop(owner_scene, "preview_active", True)

    ensure_preview_viewport(context, preview_scene, owner_scene)

    print("[Preview] Preview armature ready")
    return {
        "preview_scene_name": preview_scene.name,
        "preview_armature_name": preview_armature.name,
        "preview_action_name": preview_action.name,
    }


def preview_exists(scene) -> bool:
    bpy_module = _require_bpy()
    if not _get_scene_prop(scene, "preview_active", False):
        return False

    preview_scene = bpy_module.data.scenes.get(_get_scene_prop(scene, "preview_scene_name", ""))
    preview_armature = bpy_module.data.objects.get(_get_scene_prop(scene, "preview_armature_name", ""))
    if preview_scene is None or preview_armature is None:
        _clear_preview_properties(scene)
        return False

    return getattr(preview_armature, "type", None) == "ARMATURE"


def _view3d_areas(screen):
    return [area for area in screen.areas if area.type == "VIEW_3D"]


def _space_uses_scene(space, scene) -> bool:
    return hasattr(space, "scene") and getattr(space, "scene", None) == scene


def _bind_scene_to_area(window, area, preview_scene) -> None:
    print("[Preview] Binding preview scene")
    space = getattr(area.spaces, "active", None)
    if space is not None and hasattr(space, "scene"):
        space.scene = preview_scene
        return

    if hasattr(window, "scene"):
        window.scene = preview_scene


def _find_preview_viewport(context, preview_scene) -> Optional[Tuple[object, object]]:
    print("[Preview] Searching for preview viewport")
    window_manager = getattr(context, "window_manager", None)
    if window_manager is None:
        return None

    for window in window_manager.windows:
        screen = getattr(window, "screen", None)
        if screen is None:
            continue
        for area in _view3d_areas(screen):
            space = getattr(area.spaces, "active", None)
            if space is not None and _space_uses_scene(space, preview_scene):
                print("[Preview] Reusing preview viewport")
                return window, area

        if screen.get("generator_preview_viewport_created"):
            viewports = _view3d_areas(screen)
            if len(viewports) > 1:
                print("[Preview] Reusing preview viewport")
                return window, viewports[-1]

    return None


def _split_viewport(context):
    window = context.window
    screen = context.screen
    viewports = _view3d_areas(screen)
    if not viewports:
        return None

    if len(viewports) > 1 or screen.get("generator_preview_viewport_created"):
        return window, viewports[-1]

    print("[Preview] Creating preview viewport")
    area = viewports[0]
    region = next((item for item in area.regions if item.type == "WINDOW"), None)
    with context.temp_override(window=window, screen=screen, area=area, region=region):
        bpy.ops.screen.area_split(direction="VERTICAL", factor=0.5)

    screen["generator_preview_viewport_created"] = True
    viewports = _view3d_areas(screen)
    return window, viewports[-1] if viewports else area


def ensure_preview_viewport(context, preview_scene, original_scene=None) -> None:
    found = _find_preview_viewport(context, preview_scene)
    if found is None:
        found = _split_viewport(context)

    if found is None:
        return

    window, area = found
    _bind_scene_to_area(window, area, preview_scene)


def _restore_windows_from_preview(preview_scene, original_scene=None) -> None:
    window_manager = getattr(bpy.context, "window_manager", None)
    if window_manager is None:
        return

    replacement_scene = original_scene
    if replacement_scene is None:
        replacement_scene = next((scene for scene in bpy.data.scenes if scene != preview_scene), None)

    if replacement_scene is None:
        return

    for window in window_manager.windows:
        if getattr(window, "scene", None) == preview_scene:
            window.scene = replacement_scene


def accept_preview(context, owner_scene=None):
    bpy_module = _require_bpy()
    owner_scene = owner_scene or context.scene

    print("[Preview] Accepting preview")
    preview_armature = bpy_module.data.objects.get(_get_scene_prop(owner_scene, "preview_armature_name", ""))
    original_armature = bpy_module.data.objects.get(_get_scene_prop(owner_scene, "preview_original_armature_name", ""))

    if preview_armature is None:
        raise ValueError("Preview armature is missing.")
    if original_armature is None:
        raise ValueError("Original armature is missing.")
    if getattr(original_armature, "type", None) != "ARMATURE":
        raise TypeError(f"Object `{original_armature.name}` is not an armature.")

    preview_action = getattr(getattr(preview_armature, "animation_data", None), "action", None)
    if preview_action is None:
        raise ValueError("Preview armature has no animation action to accept.")

    print("[Preview] Copying action")
    accepted_action = preview_action.copy()
    accepted_action.name = f"{original_armature.name}_Accepted_Preview"
    if accepted_action.get(PREVIEW_MARKER):
        del accepted_action[PREVIEW_MARKER]

    original_armature.animation_data_create()
    original_armature.animation_data.action = accepted_action

    cleanup_preview_resources(owner_scene, remove_scene=True, original_scene=owner_scene)
    print("[Preview] Preview accepted")
    return accepted_action


def cancel_preview(context, owner_scene=None) -> None:
    owner_scene = owner_scene or context.scene
    print("[Preview] Canceling preview")
    cleanup_preview_resources(owner_scene, remove_scene=True, original_scene=owner_scene)
