from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, Iterable, List, Optional, Tuple

try:
    import bpy  # type: ignore
except ImportError:  # pragma: no cover - only available inside Blender
    bpy = None


PREVIEW_COLLECTION_NAME = "AI_Preview"
PREVIEW_ARMATURE_SUFFIX = "_AI_Preview"
PREVIEW_ACTION_SUFFIX = "_AI_Preview_Action"
PREVIEW_MARKER = "generator_preview_resource"
_PENDING_SCENE_PROPS: Dict[int, Dict[str, Any]] = {}
_SCENE_PROP_TIMER_REGISTERED = False


def _require_bpy():
    if bpy is None:
        raise RuntimeError("preview_manager must run inside Blender because bpy is unavailable.")
    return bpy


def _get_scene_prop(scene, name: str, default=""):
    return getattr(scene, name, default) if scene is not None else default


def _set_scene_prop(scene, name: str, value) -> None:
    if scene is None or not hasattr(scene, name):
        return
    _queue_scene_prop(scene, name, value)


def _queue_scene_prop(scene, name: str, value) -> None:
    global _SCENE_PROP_TIMER_REGISTERED

    key = int(scene.as_pointer()) if hasattr(scene, "as_pointer") else id(scene)
    record = _PENDING_SCENE_PROPS.setdefault(
        key,
        {
            "scene": scene,
            "values": {},
            "attempts": 0,
        },
    )
    record["values"][name] = value

    if not _SCENE_PROP_TIMER_REGISTERED and bpy is not None:
        _SCENE_PROP_TIMER_REGISTERED = True
        bpy.app.timers.register(_flush_pending_scene_props, first_interval=0.0)


def _flush_pending_scene_props():
    global _SCENE_PROP_TIMER_REGISTERED

    retry_needed = False
    for key, record in list(_PENDING_SCENE_PROPS.items()):
        scene = record.get("scene")
        values = dict(record.get("values") or {})
        try:
            for name, value in values.items():
                if scene is not None and hasattr(scene, name):
                    setattr(scene, name, value)
            _PENDING_SCENE_PROPS.pop(key, None)
        except RuntimeError as error:
            if "Writing to ID classes in this context is not allowed" not in str(error):
                _PENDING_SCENE_PROPS.pop(key, None)
                continue
            record["attempts"] = int(record.get("attempts") or 0) + 1
            if record["attempts"] <= 20:
                retry_needed = True
            else:
                _PENDING_SCENE_PROPS.pop(key, None)
        except ReferenceError:
            _PENDING_SCENE_PROPS.pop(key, None)

    if retry_needed and _PENDING_SCENE_PROPS:
        return 0.1

    _SCENE_PROP_TIMER_REGISTERED = False
    return None


def _clear_preview_properties(scene) -> None:
    _set_scene_prop(scene, "preview_collection_name", "")
    _set_scene_prop(scene, "preview_armature_name", "")
    _set_scene_prop(scene, "preview_action_name", "")
    _set_scene_prop(scene, "preview_original_armature_name", "")
    _set_scene_prop(scene, "preview_active", False)


def _sync_preview_properties(
    scene,
    preview_collection=None,
    preview_armature=None,
    preview_action=None,
    original_armature=None,
) -> None:
    _set_scene_prop(scene, "preview_collection_name", getattr(preview_collection, "name", ""))
    _set_scene_prop(scene, "preview_armature_name", getattr(preview_armature, "name", ""))
    _set_scene_prop(scene, "preview_action_name", getattr(preview_action, "name", ""))
    _set_scene_prop(scene, "preview_original_armature_name", getattr(original_armature, "name", ""))
    _set_scene_prop(scene, "preview_active", preview_armature is not None)


@dataclass
class PreviewSession:
    owner_scene: object | None = None
    preview_collection: object | None = None
    preview_armature: object | None = None
    preview_action: object | None = None
    original_armature: object | None = None
    viewport_window: object | None = None
    viewport_area: object | None = None
    viewport_region: object | None = None
    preview_meshes: List[object] = field(default_factory=list)

    def capture_viewport(self, context) -> None:
        print("[Preview] Capturing active viewport")
        self.viewport_window = getattr(context, "window", None)
        self.viewport_area = getattr(context, "area", None)
        self.viewport_region = _active_window_region(self.viewport_area)

        if getattr(self.viewport_area, "type", None) != "VIEW_3D":
            fallback = _find_existing_view3d(context)
            if fallback is not None:
                self.viewport_window, self.viewport_area, self.viewport_region = fallback

    def resolve_viewport(self, context) -> Optional[Tuple[object, object, object]]:
        if _viewport_is_valid(self.viewport_window, self.viewport_area, self.viewport_region):
            return self.viewport_window, self.viewport_area, self.viewport_region

        print("[Preview] Stored viewport is unavailable; falling back to an existing VIEW_3D")
        fallback = _find_existing_view3d(context)
        if fallback is None:
            return None

        self.viewport_window, self.viewport_area, self.viewport_region = fallback
        return fallback

    def has_preview(self) -> bool:
        return (
            self.preview_collection is not None
            and self.preview_armature is not None
            and getattr(self.preview_armature, "type", None) == "ARMATURE"
        )

    def state_dict(self) -> Dict[str, str]:
        return {
            "preview_collection_name": getattr(self.preview_collection, "name", ""),
            "preview_armature_name": getattr(self.preview_armature, "name", ""),
            "preview_action_name": getattr(self.preview_action, "name", ""),
        }

    def reset(self) -> None:
        self.owner_scene = None
        self.preview_collection = None
        self.preview_armature = None
        self.preview_action = None
        self.original_armature = None
        self.viewport_window = None
        self.viewport_area = None
        self.viewport_region = None
        self.preview_meshes = []


_PREVIEW_SESSION = PreviewSession()


def get_preview_session() -> PreviewSession:
    return _PREVIEW_SESSION


def _active_window_region(area):
    if area is None:
        return None
    return next((item for item in area.regions if item.type == "WINDOW"), None)


def _viewport_is_valid(window, area, region) -> bool:
    if window is None or area is None or region is None:
        return False
    screen = getattr(window, "screen", None)
    return (
        screen is not None
        and any(candidate is area for candidate in screen.areas)
        and getattr(area, "type", None) == "VIEW_3D"
    )


def _find_existing_view3d(context) -> Optional[Tuple[object, object, object]]:
    window_manager = getattr(context, "window_manager", None)
    if window_manager is None:
        return None

    for window in window_manager.windows:
        screen = getattr(window, "screen", None)
        if screen is None:
            continue
        for area in screen.areas:
            if area.type != "VIEW_3D":
                continue
            region = _active_window_region(area)
            if region is not None:
                return window, area, region
    return None


def ensure_preview_collection(owner_scene=None):
    bpy_module = _require_bpy()
    owner_scene = owner_scene or bpy_module.context.scene
    if owner_scene is None:
        raise ValueError("A scene is required to create the preview collection.")

    preview_collection = bpy_module.data.collections.get(PREVIEW_COLLECTION_NAME)
    if preview_collection is None:
        print("[Preview] Creating preview collection")
        preview_collection = bpy_module.data.collections.new(PREVIEW_COLLECTION_NAME)
        preview_collection[PREVIEW_MARKER] = True
    else:
        print("[Preview] Reusing preview collection")

    if not _collection_is_child(owner_scene.collection, preview_collection):
        owner_scene.collection.children.link(preview_collection)

    return preview_collection


def _collection_is_child(parent_collection, child_collection) -> bool:
    return any(collection is child_collection for collection in parent_collection.children)


def _iter_preview_objects(preview_collection):
    for obj in list(getattr(preview_collection, "objects", [])):
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


def _unlink_collection_from_scene(scene, collection) -> None:
    if scene is None or collection is None:
        return

    try:
        if _collection_is_child(scene.collection, collection):
            scene.collection.children.unlink(collection)
    except RuntimeError:
        pass


def cleanup_preview_collection(owner_scene=None, remove_collection: bool = True) -> None:
    bpy_module = _require_bpy()
    session = get_preview_session()
    owner_scene = owner_scene or session.owner_scene or bpy_module.context.scene
    preview_collection = session.preview_collection or bpy_module.data.collections.get(
        _get_scene_prop(owner_scene, "preview_collection_name", "") or PREVIEW_COLLECTION_NAME
    )
    preview_armature = session.preview_armature
    preview_action_name = getattr(session.preview_action, "name", "")

    print("[Preview] Cleaning preview collection")

    objects_to_remove = []
    if preview_collection is not None:
        objects_to_remove.extend(_iter_preview_objects(preview_collection))
    if preview_armature is not None and preview_armature not in objects_to_remove:
        objects_to_remove.append(preview_armature)

    for obj in objects_to_remove:
        action = getattr(getattr(obj, "animation_data", None), "action", None)
        data = getattr(obj, "data", None)
        bpy_module.data.objects.remove(obj, do_unlink=True)
        _remove_data_block(data)
        if action is session.preview_action:
            continue
        if action is not None and action.users == 0:
            bpy_module.data.actions.remove(action)

    preview_action = bpy_module.data.actions.get(preview_action_name)
    if preview_action is not None and preview_action is not session.preview_action and preview_action.users == 0:
        bpy_module.data.actions.remove(preview_action)

    for action in list(bpy_module.data.actions):
        if action is session.preview_action:
            continue
        if action.get(PREVIEW_MARKER):
            bpy_module.data.actions.remove(action)

    if remove_collection and preview_collection is not None:
        _unlink_collection_from_scene(owner_scene, preview_collection)
        if len(preview_collection.objects) == 0:
            bpy_module.data.collections.remove(preview_collection)


def cleanup_preview_resources(owner_scene=None, remove_scene: bool = False, original_scene=None) -> None:
    _require_bpy()
    session = get_preview_session()
    owner_scene = owner_scene or original_scene or session.owner_scene
    print("[Preview] Removing preview resources")
    cleanup_preview_collection(owner_scene, remove_collection=True)
    _clear_preview_properties(owner_scene)
    session.reset()
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
    action.use_fake_user = True
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


def _copy_character_to_preview(original_armature, preview_armature, preview_collection, original_scene=None) -> List[object]:
    linked_meshes = _find_linked_meshes(original_armature, original_scene)
    preview_meshes = []
    mesh_map: Dict[object, object] = {}

    for original_mesh in linked_meshes:
        preview_mesh = _duplicate_preview_mesh(original_mesh, preview_armature)
        _rebind_armature_modifiers(preview_mesh, original_armature, preview_armature)
        mesh_map[original_mesh] = preview_mesh
        preview_meshes.append(preview_mesh)

    print("[Preview] Linking preview objects")
    for original_mesh, preview_mesh in mesh_map.items():
        _restore_preview_parenting(original_mesh, preview_mesh, preview_armature, mesh_map)
        print("[Preview] Linking preview mesh")
        preview_collection.objects.link(preview_mesh)

    print("[Preview] Preview character ready")
    return preview_meshes


def prepare_preview(context, original_armature, owner_scene=None):
    bpy_module = _require_bpy()
    session = get_preview_session()
    if original_armature is None:
        raise ValueError("No armature was provided for preview generation.")
    if isinstance(original_armature, str):
        original_armature = bpy_module.data.objects.get(original_armature)
    if original_armature is None:
        raise ValueError("The original armature was not found.")
    if getattr(original_armature, "type", None) != "ARMATURE":
        raise TypeError(f"Object `{original_armature.name}` is not an armature.")

    owner_scene = owner_scene or context.scene
    cleanup_preview_resources(owner_scene, remove_scene=False, original_scene=owner_scene)
    session.owner_scene = owner_scene
    session.original_armature = original_armature
    session.capture_viewport(context)
    preview_collection = ensure_preview_collection(owner_scene)

    print("[Preview] Duplicating armature")
    preview_armature = original_armature.copy()
    preview_armature.data = original_armature.data.copy()
    preview_armature.name = _unique_preview_name(original_armature.name)
    preview_armature.data.name = f"{preview_armature.name}_Data"
    preview_armature[PREVIEW_MARKER] = True
    preview_armature["preview_source_armature"] = original_armature.name
    preview_armature.matrix_world = original_armature.matrix_world.copy()
    preview_armature.animation_data_clear()

    preview_collection.objects.link(preview_armature)
    preview_action = _make_preview_action(original_armature, preview_armature)
    session.preview_action = preview_action
    preview_action.use_fake_user = True
    preview_meshes = _copy_character_to_preview(
        original_armature,
        preview_armature,
        preview_collection,
        owner_scene,
    )

    session.preview_collection = preview_collection
    session.preview_armature = preview_armature
    session.preview_meshes = preview_meshes
    _sync_preview_properties(
        owner_scene,
        preview_collection=preview_collection,
        preview_armature=preview_armature,
        preview_action=preview_action,
        original_armature=original_armature,
    )
    _tag_preview_viewport_redraw(context)

    print("[Preview] Preview armature ready")
    return session.state_dict()


def preview_exists(scene=None) -> bool:
    session = get_preview_session()
    if session.has_preview():
        return True

    if scene is not None:
        _clear_preview_properties(scene)
    return False


def _tag_preview_viewport_redraw(context) -> None:
    session = get_preview_session()
    resolved = session.resolve_viewport(context)
    if resolved is None:
        return

    _window, area, _region = resolved
    try:
        area.tag_redraw()
    except Exception:
        pass


def get_preview_armature():
    return get_preview_session().preview_armature


def get_preview_action():
    return get_preview_session().preview_action


def refresh_preview_action(owner_scene=None):
    session = get_preview_session()
    if session.preview_armature is None:
        return None

    session.preview_action = getattr(
        getattr(session.preview_armature, "animation_data", None),
        "action",
        None,
    )
    if session.preview_action is not None:
        session.preview_action.use_fake_user = True
    _set_scene_prop(owner_scene or session.owner_scene, "preview_action_name", getattr(session.preview_action, "name", ""))
    return session.preview_action


def _assign_first_action_slot(animation_data, action) -> None:
    slots = getattr(action, "slots", None)
    if slots is None or not hasattr(animation_data, "action_slot"):
        return

    try:
        slot_count = len(slots)
    except TypeError:
        return
    if slot_count == 0:
        return

    for slot in slots:
        try:
            animation_data.action_slot = slot
            print("[Accept] Assigned action slot:", getattr(slot, "name", slot))
            return
        except Exception:
            continue


def accept_preview(context, owner_scene=None):
    bpy_module = _require_bpy()
    session = get_preview_session()
    owner_scene = owner_scene or session.owner_scene or context.scene

    print("[Preview] Accepting preview")
    preview_armature = session.preview_armature
    original_armature = session.original_armature

    if preview_armature is None:
        raise ValueError("Preview armature is missing.")
    if original_armature is None:
        raise ValueError("Original armature is missing.")
    if getattr(original_armature, "type", None) != "ARMATURE":
        raise TypeError(f"Object `{original_armature.name}` is not an armature.")

    assert session.preview_action is not None
    preview_action = session.preview_action
    if preview_action is None:
        raise ValueError("Preview session has no animation action to accept.")

    print(
        "[Preview] Accept action: "
        f"name={getattr(preview_action, 'name', '')}, "
        f"users={getattr(preview_action, 'users', 0)}, "
        f"preview_marker={bool(preview_action.get(PREVIEW_MARKER))}"
    )

    print("[Preview] Assigning accepted preview action")
    accepted_action = preview_action.copy()
    assert accepted_action is not None
    assert isinstance(accepted_action, bpy_module.types.Action)
    assert accepted_action.users >= 0

    accepted_action.name = f"{original_armature.name}_Accepted_Preview"
    if accepted_action.get(PREVIEW_MARKER):
        del accepted_action[PREVIEW_MARKER]
    accepted_action.use_fake_user = True
    session.preview_action = accepted_action
    print("[Accept] Copied action f-curves:", len(getattr(accepted_action, "fcurves", [])))

    if original_armature.animation_data is None:
        original_armature.animation_data_create()

    ad = original_armature.animation_data
    ad.use_nla = False
    ad.use_tweak_mode = False
    ad.action = None
    bpy.context.view_layer.update()
    ad.action = accepted_action
    _assign_first_action_slot(ad, accepted_action)
    bpy.context.view_layer.update()

    print("[Accept] Assigned action:", original_armature.animation_data.action)
    print("[Accept] Action name:", getattr(original_armature.animation_data.action, "name", None))

    cleanup_preview_resources(owner_scene, remove_scene=True, original_scene=owner_scene)
    accepted_action.use_fake_user = False
    if original_armature.animation_data is None or original_armature.animation_data.action is not accepted_action:
        raise RuntimeError("Accepted preview action was not retained on the original armature after cleanup.")
    print("[Preview] Preview accepted")
    return accepted_action


def cancel_preview(context, owner_scene=None) -> None:
    session = get_preview_session()
    owner_scene = owner_scene or session.owner_scene or context.scene
    print("[Preview] Canceling preview")
    cleanup_preview_resources(owner_scene, remove_scene=True, original_scene=owner_scene)
    print("[Preview] Preview cancelled")
