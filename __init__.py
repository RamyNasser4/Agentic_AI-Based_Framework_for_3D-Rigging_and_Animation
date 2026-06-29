import ast
import os
import queue
import re
import textwrap
import threading
import traceback

import bpy  # type: ignore

from .Blender_Executer import BlenderExecutor
from .planner_agent import run_llm

bl_info = {
    "name": "Generator Panel",
    "author": "Rana",
    "version": (1, 0),
    "blender": (4, 0, 0),
    "category": "3D View",
}


_GENERATION_JOBS = {}
_GENERATION_JOBS_LOCK = threading.Lock()
_TIMER_INTERVAL = 0.1
_AXIS_ENUM_ITEMS = (
    ("+X", "+X", ""),
    ("-X", "-X", ""),
    ("+Y", "+Y", ""),
    ("-Y", "-Y", ""),
    ("+Z", "+Z", ""),
    ("-Z", "-Z", ""),
)
_AXIS_VALUES = {item[0] for item in _AXIS_ENUM_ITEMS}


def dropdown_items(self, context):
    items = []

    for obj in context.scene.objects:
        if obj.type == "ARMATURE":
            items.append((obj.name, obj.name, ""))

    return items


def split_animation_plan(plan_text):
    lines = [line.strip() for line in str(plan_text).splitlines() if line.strip()]
    if not lines:
        return []

    grouped_steps = []
    current_step = []
    step_header = re.compile(
        r"^(?:\[generator\]\s*plan\s*step\s*\d+\s*:|step\s*\d+\s*:|\d+[\.\)])\s*",
        re.IGNORECASE,
    )

    for line in lines:
        if step_header.match(line):
            if current_step:
                grouped_steps.append(" ".join(current_step).strip())
            current_step = [line]
        elif current_step:
            current_step.append(line)
        else:
            grouped_steps.append(line)

    if current_step:
        grouped_steps.append(" ".join(current_step).strip())

    cleaned_steps = []
    for step_text in grouped_steps:
        cleaned_step = step_header.sub("", step_text).strip()
        if cleaned_step:
            cleaned_steps.append(cleaned_step)

    return cleaned_steps


def format_plan_step(step_text, step_index):
    cleaned_step = str(step_text or "").strip()
    cleaned_step = re.sub(
        r"^\sStep\s*\d+\s*:\s*",
        "",
        cleaned_step,
        flags=re.IGNORECASE,
    )
    cleaned_step = re.sub(r"^\s*Step\s*\d+\s*:\s*", "", cleaned_step, flags=re.IGNORECASE)
    cleaned_step = re.sub(r"\s+", " ", cleaned_step).strip()
    return f"Step {step_index}: {cleaned_step}"


def load_keyframe_agent_class():
    try:
        from .keyframe_agent import KeyFrameAgent

        return KeyFrameAgent
    except Exception as import_error:
        print(f"[Generator] Direct KeyFrameAgent import failed: {import_error}")
        print("[Generator] Falling back to loading KeyFrameAgent without demo code.")

        module_path = os.path.join(os.path.dirname(__file__), "keyframe_agent.py")
        with open(module_path, "r", encoding="utf-8") as handle:
            source = handle.read()

        module_ast = ast.parse(source, filename=module_path)
        allowed_nodes = []
        allowed_assignments = {"SYSTEM_MESSAGE", "REPAIR_SYSTEM_MESSAGE", "animation_examples"}

        for node in module_ast.body:
            if isinstance(node, (ast.Import, ast.ImportFrom, ast.ClassDef)):
                allowed_nodes.append(node)
            elif isinstance(node, ast.Expr):
                call = getattr(node, "value", None)
                if (
                    isinstance(call, ast.Call)
                    and isinstance(call.func, ast.Name)
                    and call.func.id == "load_dotenv"
                ):
                    allowed_nodes.append(node)
            elif isinstance(node, ast.Assign):
                target_names = {
                    target.id
                    for target in node.targets
                    if isinstance(target, ast.Name)
                }
                if target_names and target_names.issubset(allowed_assignments):
                    allowed_nodes.append(node)

        namespace = {"__file__": module_path, "__name__": "GP.keyframe_agent_runtime"}
        sanitized_module = ast.Module(body=allowed_nodes, type_ignores=[])
        exec(compile(sanitized_module, module_path, "exec"), namespace)
        return namespace["KeyFrameAgent"]


def _scene_job_key(scene):
    return scene.as_pointer()


def _get_generation_job(scene):
    with _GENERATION_JOBS_LOCK:
        return _GENERATION_JOBS.get(_scene_job_key(scene))


def _set_generation_job(scene, job_state):
    with _GENERATION_JOBS_LOCK:
        _GENERATION_JOBS[_scene_job_key(scene)] = job_state


def _clear_generation_job(scene):
    with _GENERATION_JOBS_LOCK:
        _GENERATION_JOBS.pop(_scene_job_key(scene), None)


def _tag_redraw():
    window_manager = getattr(bpy.context, "window_manager", None)
    if window_manager is None:
        return

    for window in window_manager.windows:
        screen = getattr(window, "screen", None)
        if screen is None:
            continue
        for area in screen.areas:
            if area.type == "VIEW_3D":
                area.tag_redraw()


def _set_status(job_state, status, animate=False):
    job_state["status"] = status
    job_state["status_base"] = status if animate else ""
    job_state["status_dots"] = 0


def _animate_waiting_status(scene, job_state):
    if not job_state["waiting_for_llm"]:
        return

    status_base = job_state.get("status_base")
    if not status_base:
        return

    job_state["status_dots"] = (job_state.get("status_dots", 0) + 1) % 4
    dots = "." * job_state["status_dots"]
    updated_status = f"{status_base}{dots}"

    if updated_status != job_state["status"]:
        job_state["status"] = updated_status
        _refresh_scene_output(scene, job_state)


def _build_output_text(job_state):
    parts = [
        f"Status: {job_state.get('status') or 'Idle'}",
        f"Progress: {job_state.get('current_step', 0)}/{job_state.get('total_steps', 0)}",
        f"Current Step: {job_state.get('current_instruction') or '-'}",
    ]

    if job_state["plan_text"]:
        parts.append("")
        parts.append(job_state["plan_text"])

    if job_state["step_logs"]:
        parts.append("")
        parts.extend(job_state["step_logs"])

    return "\n".join(parts).strip()


def _refresh_scene_output(scene, job_state):
    scene.gen_output = _build_output_text(job_state)
    _tag_redraw()


def _shutdown_generation_worker(job_state):
    worker_thread = job_state.get("thread")
    request_queue = job_state.get("request_queue")

    if worker_thread is None or request_queue is None:
        return

    if not worker_thread.is_alive():
        return

    try:
        request_queue.put_nowait({"type": "shutdown"})
    except Exception:
        pass


def _finalize_generation(scene, job_state=None):
    if job_state is None:
        job_state = _get_generation_job(scene)

    if job_state is not None:
        _shutdown_generation_worker(job_state)

    scene.gen_loading = False
    _clear_generation_job(scene)
    _tag_redraw()


def _queue_worker_error(result_queue, message, item_type="error", **extra):
    payload = {
        "type": item_type,
        "message": message,
    }
    payload.update(extra)
    result_queue.put(payload)


def _queue_refinement_status(
    result_queue,
    status,
    current_instruction="",
    log=None,
    animate=False,
    **extra,
):
    payload = {
        "type": "refinement_status",
        "status": status,
        "current_instruction": current_instruction,
        "log": log,
        "animate": animate,
    }
    payload.update(extra)
    result_queue.put(payload)


def _axis_world_component(axis):
    axis = str(axis or "")
    return axis[-1:] if axis in _AXIS_VALUES else ""


def _validate_direction_axes(forward_axis, up_axis, right_axis):
    axes = (forward_axis, up_axis, right_axis)
    if any(axis not in _AXIS_VALUES for axis in axes):
        return "Choose a valid Forward, Up, and Right axis."

    world_components = [_axis_world_component(axis) for axis in axes]
    if len(set(world_components)) != 3:
        return "Forward, Up, and Right must use three different world axes."

    return ""


def _direction_settings_from_scene(scene):
    return {
        "forward_axis": getattr(scene, "gen_forward_axis", "+Y"),
        "up_axis": getattr(scene, "gen_up_axis", "+Z"),
        "right_axis": getattr(scene, "gen_right_axis", "+X"),
    }


def _validate_scene_direction_settings(scene):
    settings = _direction_settings_from_scene(scene)
    return _validate_direction_axes(
        settings["forward_axis"],
        settings["up_axis"],
        settings["right_axis"],
    )


def _format_direction_context(direction_settings):
    forward_axis = direction_settings["forward_axis"]
    up_axis = direction_settings["up_axis"]
    right_axis = direction_settings["right_axis"]
    return (
        "Semantic direction inference: "
        f"forward_axis={forward_axis}, "
        f"up_axis={up_axis}, "
        f"right_axis={right_axis}, "
        "is_humanoid=null, "
        "confidence=1.00, "
        "needs_user_confirmation=false"
    )


def _append_direction_context(object_json, direction_settings):
    if direction_settings is None:
        return object_json

    validation_error = _validate_direction_axes(
        direction_settings.get("forward_axis"),
        direction_settings.get("up_axis"),
        direction_settings.get("right_axis"),
    )
    if validation_error:
        raise ValueError(validation_error)

    return f"{object_json}\n{_format_direction_context(direction_settings)}"


def _generation_worker(request_queue, result_queue, object_name, object_json, prompt):
    try:
        print(f"[Generator] Worker started for object `{object_name}`.")
        print("[Generator] Stage 2: Generating animation plan in background thread.")
        animation_plan = run_llm(object_name, object_json, prompt)
        print(animation_plan)
        plan_steps = split_animation_plan(animation_plan)

        if not plan_steps:
            fallback_step = str(prompt).strip()
            plan_steps = [fallback_step] if fallback_step else []
            if plan_steps:
                animation_plan = "\n".join(
                    f"Step {index}: {step}"
                    for index, step in enumerate(plan_steps, start=1)
                )

        result_queue.put(
            {
                "type": "plan",
                "plan": str(animation_plan),
                "steps": list(plan_steps),
            }
        )

        if not plan_steps:
            _queue_worker_error(result_queue, "Animation planner returned no executable steps.")
            return

        print(f"[Generator] Parsed {len(plan_steps)} plan step(s) in worker thread.")
        KeyFrameAgent = load_keyframe_agent_class()
        keyframe = KeyFrameAgent()
        keyframe.initialize_chain()
        print("[Generator] Keyframe agent initialized in worker thread.")

        while True:
            request = request_queue.get()
            if request is None:
                break

            request_type = request.get("type")
            if request_type == "shutdown":
                print("[Generator] Worker received shutdown signal.")
                break
            if request_type != "step":
                continue

            index = request["index"]
            current_plan_step = request["current_plan_step"]
            previous_animation = request.get("previous_animation")
            plan_history = request.get("plan_history")
            last_step_index = request.get("last_step_index")
            user_instruction = request.get("user_instruction")

            print(f"[Generator] Worker generating keyframes for step {index}.")

            try:
                response = keyframe.invoke_chain(
                    {
                        "object": object_name,
                        "object_json": request["object_json"],
                        "user_instruction": user_instruction,
                        "current_plan_step": current_plan_step,
                        "plan_history": plan_history,
                        "last_step_index": last_step_index,
                        "previous_animation": previous_animation,
                    }
                )
                print(f"[Generator] Worker generated response for step {index}.")
            except Exception as error:
                print(f"[Generator] Keyframe generation failed for step {index}: {error}")
                _queue_worker_error(
                    result_queue,
                    f"Keyframe generation failed for step {index}: {error}",
                    item_type="step_error",
                    index=index,
                    step=current_plan_step,
                )
                continue

            if response is None:
                print(f"[Generator] Keyframe generation returned no result for step {index}.")
                _queue_worker_error(
                    result_queue,
                    f"Keyframe generation returned no result for step {index}.",
                    item_type="step_error",
                    index=index,
                    step=current_plan_step,
                )
                continue

            result_queue.put(
                {
                    "type": "step_result",
                    "index": index,
                    "step": current_plan_step,
                    "response": response,
                }
            )
    except Exception as error:
        print("[Generator] Worker thread failed:")
        print(traceback.format_exc())
        _queue_worker_error(result_queue, f"Generation failed: {error}")


def _refinement_worker(
    request_queue,
    result_queue,
    object_name,
    object_json,
    prompt,
    preview_armature_name=None,
):
    try:
        print(f"[Generator] Worker started for object `{object_name}`.")
        print(f"[Generator] Refinement worker started for object `{object_name}`.")
        plan, keyframes, feedback = run_with_refinement(
            prompt,
            object_name=object_name,
            object_json=object_json,
            request_queue=request_queue,
            result_queue=result_queue,
            preview_armature_name=preview_armature_name,
            debug=True,
        )
        result_queue.put(
            {
                "type": "refinement_complete",
                "plan": plan,
                "keyframes": keyframes,
                "feedback": feedback,
            }
        )
    except Exception as error:
        print("[Generator] Worker thread failed:")
        print(traceback.format_exc())
        _queue_worker_error(result_queue, f"Generation failed: {error}")


def _execute_generation_step(scene, job_state, item):
    index = item["index"]
    step = item["step"]
    response = item["response"]

    _set_status(job_state, f"Executing Step {index}")
    _refresh_scene_output(scene, job_state)

    try:
        print(f"[Generator] Stage 5.{index}: Executing keyframes in Blender.")
        job_state["executor"].execute_from_text(response)
        print(f"[Generator] Blender execution completed for step {index}.")
        job_state["all_outputs"].append(response)
        job_state["previous_animation"] = "\n".join(
            output for output in job_state["all_outputs"] if str(output).strip()
        )
        job_state["plan_history"].append(step)
        job_state["final_animation"] = job_state["previous_animation"]
        job_state["step_logs"].append(f"Executed Step {index}: {step}")
    except Exception as error:
        print(f"[Generator] Blender execution failed for step {index}: {error}")
        job_state["step_logs"].append(f"Execution failed for Step {index}: {error}")
        _set_status(job_state, f"Execution failed on Step {index}")
        _refresh_scene_output(scene, job_state)
        _finalize_generation(scene, job_state)
        return

    job_state["current_step"] = index
    job_state["waiting_for_llm"] = False

    if job_state["current_step"] >= job_state["total_steps"]:
        job_state["current_instruction"] = ""
        _set_status(job_state, "Completed")
        _refresh_scene_output(scene, job_state)
        _finalize_generation(scene, job_state)
        return

    next_step_index = job_state["current_step"] + 1
    job_state["current_instruction"] = format_plan_step(
        job_state["steps"][job_state["current_step"]],
        next_step_index,
    )
    _set_status(job_state, f"Step {index} complete")
    _refresh_scene_output(scene, job_state)


def _execute_refinement_iteration(scene, job_state, item):
    from .skeleton_recorder import extract_skeleton_frames

    iteration = item["iteration"]
    keyframes = item["keyframes"]
    preview_armature_name = job_state.get("preview_armature_name") or job_state["object_name"]

    job_state["current_step"] = iteration
    job_state["current_instruction"] = f"Iteration {iteration}"
    job_state["waiting_for_llm"] = False
    _set_status(job_state, f"Executing iteration {iteration}")
    _refresh_scene_output(scene, job_state)

    try:
        print(f"[Generator] Refinement Stage {iteration}: Executing keyframes in Blender.")
        print(f"[Preview] Executing refinement on preview armature `{preview_armature_name}`.")
        job_state["executor"].execute_from_text(
            keyframes,
            armature_name=preview_armature_name,
            clear_existing_action=True,
        )
        preview_armature = bpy.data.objects.get(preview_armature_name)
        preview_action = getattr(getattr(preview_armature, "animation_data", None), "action", None)
        if preview_action is not None:
            job_state["preview_action_name"] = preview_action.name
            scene.preview_action_name = preview_action.name
            print(f"[Preview] Preview action updated to `{preview_action.name}`.")
        print(f"[Generator] Refinement Blender execution completed for iteration {iteration}.")
        job_state["final_animation"] = keyframes
        job_state["previous_animation"] = keyframes

        _set_status(job_state, "Rendering visual evidence")
        _refresh_scene_output(scene, job_state)
        print(
            f"[Generator] Refinement Stage {iteration}: "
            f"Extracting skeleton frames from preview armature `{preview_armature_name}`."
        )
        skeleton_frames = extract_skeleton_frames(preview_armature_name)
        print(f"[Generator] Skeleton extraction completed for iteration {iteration}.")
        visual = _build_visual_evidence_for_iteration(
            prompt=job_state.get("user_prompt", ""),
            armature_name=preview_armature_name,
            skeleton_frames=skeleton_frames,
            scene=bpy.data.scenes.get(job_state.get("preview_scene_name") or "") or bpy.context.scene,
        )

        job_state["request_queue"].put(
            {
                "type": "refinement_iteration_data",
                "iteration": iteration,
                "visual": visual,
            }
        )
    except Exception as error:
        print(f"[Generator] Refinement iteration {iteration} failed in Blender: {error}")
        print(traceback.format_exc())
        job_state["step_logs"].append(f"Refinement iteration {iteration} failed: {error}")
        job_state["request_queue"].put(
            {
                "type": "refinement_iteration_error",
                "iteration": iteration,
                "message": str(error),
            }
        )
        _set_status(job_state, f"Refinement iteration {iteration} failed")
        _refresh_scene_output(scene, job_state)
        _finalize_generation(scene, job_state)


def _build_visual_evidence_for_iteration(prompt, armature_name, skeleton_frames, scene=None):
    from .mesh_renderer import MeshRenderer, build_visual_evidence
    from .skeleton_visualizer import SkeletonVisualizer

    armature = bpy.data.objects.get(armature_name)
    if armature is None:
        raise ValueError(f"Armature `{armature_name}` was not found while building visual evidence.")

    print(f"[Generator] Building VisualEvidence for `{armature_name}`.")
    skeleton_collage_sequence = SkeletonVisualizer().render_collage_sequence(skeleton_frames)
    frame_count = max(len(skeleton_collage_sequence), 1)
    frame_objects = [[armature] for _ in range(frame_count)]
    mesh_collage_sequence = MeshRenderer(scene=scene).render_collage_sequence(frame_objects)

    return build_visual_evidence(
        prompt=prompt,
        mesh_collage_sequence=mesh_collage_sequence,
        skeleton_collage_sequence=skeleton_collage_sequence,
    )


def _dispatch_next_step(scene, job_state):
    if job_state["waiting_for_llm"]:
        return
    if not job_state["plan_ready"]:
        return
    if job_state["current_step"] >= job_state["total_steps"]:
        return

    step_index = job_state["current_step"] + 1
    instruction = job_state["steps"][job_state["current_step"]]
    current_plan_step = format_plan_step(instruction, step_index)
    job_state["current_instruction"] = current_plan_step

    _set_status(job_state, f"Parsing scene for Step {step_index}")
    _refresh_scene_output(scene, job_state)

    try:
        object_json = job_state["parser"].generate_object_json([job_state["object_name"]])
        object_json = _append_direction_context(
            object_json,
            job_state.get("direction_settings"),
        )
        print(f"[Generator] Parsed fresh scene state for step {step_index}.")
    except Exception as error:
        print(f"[Generator] Scene parse failed for step {step_index}: {error}")
        job_state["step_logs"].append(f"Scene parse failed for Step {step_index}: {error}")
        _set_status(job_state, f"Scene parse failed on Step {step_index}")
        _refresh_scene_output(scene, job_state)
        _finalize_generation(scene, job_state)
        return

    _set_status(job_state, f"Sending Step {step_index} to LLM")
    _refresh_scene_output(scene, job_state)

    job_state["request_queue"].put(
        {
            "type": "step",
            "index": step_index,
            "current_plan_step": current_plan_step,
            "plan_history": list(job_state["plan_history"]),
            "last_step_index": len(job_state["plan_history"]),
            "user_instruction": job_state["user_prompt"],
            "object_json": object_json,
            "previous_animation": job_state["previous_animation"],
        }
    )

    job_state["waiting_for_llm"] = True
    _set_status(job_state, f"Waiting for LLM (Step {step_index})", animate=True)
    _refresh_scene_output(scene, job_state)


def _handle_generation_queue(scene):
    job_state = _get_generation_job(scene)
    if job_state is None:
        scene.gen_loading = False
        _tag_redraw()
        return None

    result_queue = job_state["result_queue"]

    try:
        item = result_queue.get_nowait()
    except queue.Empty:
        if not job_state["thread"].is_alive():
            job_state["step_logs"].append("Generation worker stopped before completion.")
            _set_status(job_state, "Worker stopped")
            _refresh_scene_output(scene, job_state)
            _finalize_generation(scene, job_state)
            return None

        if job_state["waiting_for_llm"]:
            _animate_waiting_status(scene, job_state)
            return _TIMER_INTERVAL

        if (
            job_state.get("mode") != "refinement"
            and job_state["plan_ready"]
            and job_state["current_step"] < job_state["total_steps"]
        ):
            _dispatch_next_step(scene, job_state)
            return _TIMER_INTERVAL if _get_generation_job(scene) is not None else None

        return _TIMER_INTERVAL

    item_type = item["type"]

    if item_type == "plan":
        job_state["plan_text"] = item["plan"]
        job_state["steps"] = item["steps"]
        job_state["total_steps"] = len(item["steps"])
        job_state["plan_ready"] = True
        job_state["current_instruction"] = (
            format_plan_step(job_state["steps"][0], 1) if job_state["steps"] else ""
        )
        job_state["step_logs"].append(f"Planner returned {job_state['total_steps']} step(s).")
        _set_status(job_state, f"Plan ready ({job_state['total_steps']} steps)")
        _refresh_scene_output(scene, job_state)
        return _TIMER_INTERVAL

    if item_type == "refinement_status":
        job_state["waiting_for_llm"] = bool(item.get("animate", False))
        if item.get("current_instruction") is not None:
            job_state["current_instruction"] = item.get("current_instruction") or job_state["current_instruction"]
        log_message = item.get("log")
        if log_message:
            job_state["step_logs"].append(log_message)
        _set_status(job_state, item["status"], animate=bool(item.get("animate", False)))
        _refresh_scene_output(scene, job_state)
        return _TIMER_INTERVAL

    if item_type == "refinement_plan":
        job_state["plan_text"] = item["plan"]
        job_state["steps"] = item.get("steps", [])
        job_state["plan_ready"] = True
        job_state["step_logs"].append(f"Initial planner returned {len(job_state['steps'])} step(s).")
        _set_status(job_state, "Generating keyframes", animate=True)
        _refresh_scene_output(scene, job_state)
        return _TIMER_INTERVAL

    if item_type == "refinement_execute":
        _execute_refinement_iteration(scene, job_state, item)
        return _TIMER_INTERVAL if _get_generation_job(scene) is not None else None

    if item_type == "refinement_complete":
        job_state["waiting_for_llm"] = False
        print("[Generator] Refinement completed.")
        job_state["plan_text"] = item["plan"]
        job_state["final_plan"] = item["plan"]
        job_state["final_keyframes"] = item["keyframes"]
        job_state["final_feedback"] = item["feedback"]
        job_state["final_animation"] = item["keyframes"]
        job_state["previous_animation"] = item["keyframes"]
        job_state["all_outputs"] = [item["keyframes"]]

        faithfulness_score = float(item["feedback"].get("faithfulness", {}).get("score", 0.0))
        realism_score = float(item["feedback"].get("realism", {}).get("score", 0.0))
        job_state["step_logs"].append(
            f"Final feedback: faithfulness={faithfulness_score:.3f}, realism={realism_score:.3f}"
        )
        job_state["current_instruction"] = ""
        _set_status(job_state, "Completed")
        _refresh_scene_output(scene, job_state)
        _finalize_generation(scene, job_state)
        return None

    if item_type == "step_result":
        job_state["waiting_for_llm"] = False
        _execute_generation_step(scene, job_state, item)
        return _TIMER_INTERVAL if _get_generation_job(scene) is not None else None

    if item_type == "step_error":
        job_state["waiting_for_llm"] = False
        job_state["step_logs"].append(item["message"])
        job_state["current_instruction"] = item.get("step") or job_state["current_instruction"]
        _set_status(job_state, f"Step {item.get('index', '?')} failed")
        _refresh_scene_output(scene, job_state)
        _finalize_generation(scene, job_state)
        return None

    if item_type == "error":
        job_state["waiting_for_llm"] = False
        job_state["step_logs"].append(item["message"])
        _set_status(job_state, "Generation failed")
        _refresh_scene_output(scene, job_state)
        _finalize_generation(scene, job_state)
        return None

    return _TIMER_INTERVAL


def _register_generation_timer(scene):
    def timer_callback():
        return _handle_generation_queue(scene)

    bpy.app.timers.register(timer_callback, first_interval=_TIMER_INTERVAL)


def _resolve_target_object_name(context, scene):
    if getattr(scene, "gen_mode", ""):
        return scene.gen_mode
    if context.active_object is not None:
        return context.active_object.name
    return ""


def run_with_refinement(
    prompt,
    object_name=None,
    scene=None,
    max_iters=4,
    score_threshold=0.9,
    debug=False,
    object_json=None,
    request_queue=None,
    result_queue=None,
    preview_armature_name=None,
):
    from .SceneParser import SceneParser
    from .critic_agent import evaluate_motion
    from .refinement import generate_keyframes_for_plan, refine
    from .skeleton_recorder import extract_skeleton_frames

    queue_mode = request_queue is not None and result_queue is not None
    active_scene = None if queue_mode else scene or bpy.context.scene
    active_object_name = object_name
    if not active_object_name and active_scene is not None:
        active_object_name = _resolve_target_object_name(bpy.context, active_scene)
    if not active_object_name:
        raise ValueError("run_with_refinement requires an armature object name.")
    execution_armature_name = preview_armature_name or active_object_name

    if object_json is None:
        if queue_mode:
            raise ValueError("Queue-based run_with_refinement requires pre-parsed object_json.")
        parser = SceneParser(precision=1)
        object_json = parser.generate_object_json([active_object_name])
        direction_settings = _direction_settings_from_scene(active_scene)
        object_json = _append_direction_context(object_json, direction_settings)

    executor = None if queue_mode else BlenderExecutor()

    if result_queue is not None:
        _queue_refinement_status(result_queue, "Generating plan", animate=True)

    print("[Generator] Stage 2: Generating animation plan in background thread.")
    plan = run_llm(active_object_name, object_json, prompt)
    print(plan)
    plan_steps = split_animation_plan(plan)
    if result_queue is not None:
        result_queue.put(
            {
                "type": "refinement_plan",
                "plan": str(plan),
                "steps": list(plan_steps),
            }
        )

    if result_queue is not None:
        _queue_refinement_status(result_queue, "Generating keyframes", animate=True)
    print("[Generator] Stage 3: Generating keyframes for refinement.")
    keyframes = generate_keyframes_for_plan(
        object_name=active_object_name,
        object_json=object_json,
        prompt=prompt,
        plan_text=plan,
    )

    previous_score = 0.0
    feedback = {
        "faithfulness": {"score": 0.0, "issues": []},
        "realism": {"score": 0.0, "issues": []},
    }
    critic_state = {
        "iteration": 0,
        "previous_issues": [],
        "resolved_issues": [],
        "previous_score": {},
    }

    for iteration in range(1, int(max_iters) + 1):
        if result_queue is not None:
            _queue_refinement_status(
                result_queue,
                f"Executing iteration {iteration}",
                current_instruction=f"Iteration {iteration}",
            )

        if queue_mode:
            visual = _request_refinement_iteration_execution(
                request_queue=request_queue,
                result_queue=result_queue,
                iteration=iteration,
                keyframes=keyframes,
            )
        else:
            print(f"[Generator] Refinement Stage {iteration}: Executing keyframes in Blender.")
            print(f"[Preview] Executing refinement on armature `{execution_armature_name}`.")
            executor.execute_from_text(
                keyframes,
                armature_name=execution_armature_name,
                clear_existing_action=True,
            )
            print(f"[Generator] Refinement Blender execution completed for iteration {iteration}.")
            print(
                f"[Generator] Refinement Stage {iteration}: "
                f"Extracting skeleton frames from `{execution_armature_name}`."
            )
            skeleton_frames = extract_skeleton_frames(execution_armature_name)
            visual = _build_visual_evidence_for_iteration(
                prompt=prompt,
                armature_name=execution_armature_name,
                skeleton_frames=skeleton_frames,
                scene=active_scene,
            )

        if result_queue is not None:
            _queue_refinement_status(
                result_queue,
                "Evaluating motion",
                current_instruction=f"Iteration {iteration}",
                animate=True,
            )
        print(f"[Generator] Refinement Stage {iteration}: Evaluating motion.")
        feedback = evaluate_motion(prompt, visual, critic_state=critic_state)

        faithfulness_score = float(feedback.get("faithfulness", {}).get("score", 0.0))
        realism_score = float(feedback.get("realism", {}).get("score", 0.0))
        score = (faithfulness_score + realism_score) / 2.0

        faithfulness_issues = feedback.get("faithfulness", {}).get("issues", [])
        realism_issues = feedback.get("realism", {}).get("issues", [])
        issue_count = len(faithfulness_issues) + len(realism_issues)

        if debug:
            print(
                "[Refinement] "
                f"iteration={iteration} "
                f"faithfulness={faithfulness_score:.3f} "
                f"realism={realism_score:.3f} "
                f"issues={issue_count}"
            )
        if result_queue is not None:
            _queue_refinement_status(
                result_queue,
                f"Iteration {iteration} evaluated",
                current_instruction=f"Iteration {iteration}",
                log=(
                    f"Iteration {iteration}: "
                    f"faithfulness={faithfulness_score:.3f}, "
                    f"realism={realism_score:.3f}, "
                    f"issues={issue_count}"
                ),
            )

        if iteration > 1 and score <= previous_score:
            print(f"[Refinement] Stopping because score did not improve at iteration {iteration}.")
            break

        if faithfulness_score > score_threshold and realism_score > score_threshold:
            print(f"[Refinement] Stopping because threshold was reached at iteration {iteration}.")
            break

        if iteration >= int(max_iters):
            print(f"[Refinement] Stopping because max iterations was reached at iteration {iteration}.")
            break

        if result_queue is not None:
            _queue_refinement_status(
                result_queue,
                "Generating plan fixes",
                current_instruction=f"Iteration {iteration}",
                animate=True,
            )
        print(f"[Generator] Refinement Stage {iteration}: Generating plan fixes and patching plan.")
        plan, keyframes = refine(
            plan,
            keyframes,
            feedback,
            object_name=active_object_name,
            object_json=object_json,
            prompt=prompt,
            visual=visual,
        )
        previous_score = score

    return plan, keyframes, feedback


def _request_refinement_iteration_execution(
    request_queue,
    result_queue,
    iteration,
    keyframes,
):
    result_queue.put(
        {
            "type": "refinement_execute",
            "iteration": iteration,
            "keyframes": keyframes,
        }
    )

    while True:
        response = request_queue.get()
        if response is None:
            raise RuntimeError("Refinement worker received an empty main-thread response.")

        response_type = response.get("type")
        if response_type == "shutdown":
            print("[Generator] Worker received shutdown signal.")
            raise RuntimeError("Refinement worker was shut down before completion.")

        if response.get("iteration") != iteration:
            continue

        if response_type == "refinement_iteration_error":
            raise RuntimeError(response.get("message") or "Refinement iteration failed in Blender.")

        if response_type == "refinement_iteration_data":
            return response["visual"]


def _scene_has_preview(scene):
    try:
        from . import preview_manager

        return preview_manager.preview_exists(scene)
    except Exception as error:
        print(f"[Preview] Preview state check failed: {error}")
        return False


def _run_direction_inference_into_scene(context, object_name):
    from .direction_inference_agent import DirectionInferenceAgent

    scene = context.scene
    print("[Generator] Inferring semantic direction on main thread.")
    direction_result = DirectionInferenceAgent(context=context).infer([object_name])

    inferred_settings = {
        "forward_axis": direction_result.forward_axis,
        "up_axis": direction_result.up_axis,
        "right_axis": direction_result.right_axis,
    }
    validation_error = _validate_direction_axes(
        inferred_settings["forward_axis"],
        inferred_settings["up_axis"],
        inferred_settings["right_axis"],
    )
    if validation_error:
        raise ValueError(f"Direction inference returned unusable axes: {validation_error}")

    scene.gen_forward_axis = inferred_settings["forward_axis"]
    scene.gen_up_axis = inferred_settings["up_axis"]
    scene.gen_right_axis = inferred_settings["right_axis"]
    scene.gen_direction_inference_ready = True
    scene.gen_direction_inferred_object_name = object_name
    scene.gen_direction_error = ""

    print(
        "[Generator] Direction inference populated UI controls: "
        f"forward={direction_result.forward_axis}, "
        f"up={direction_result.up_axis}, "
        f"right={direction_result.right_axis}, "
        f"humanoid={direction_result.is_humanoid}, "
        f"confidence={direction_result.confidence:.2f}."
    )
    scene.gen_output = (
        "Status: Direction inference completed\n"
        "Review the Direction Settings, adjust them if needed, then generate again.\n"
        f"Forward Axis: {scene.gen_forward_axis}\n"
        f"Up Axis: {scene.gen_up_axis}\n"
        f"Right Axis: {scene.gen_right_axis}\n"
        f"Confidence: {direction_result.confidence:.2f}"
    )
    _tag_redraw()
    return direction_result


class GENERATOR_OT_infer_directions(bpy.types.Operator):
    bl_label = "Infer Directions"
    bl_idname = "generator.infer_directions"

    def execute(self, context):
        scene = context.scene

        if context.active_object is None:
            self.report({"WARNING"}, "Select an object before inferring directions")
            return {"CANCELLED"}

        object_name = _resolve_target_object_name(context, scene)
        if not object_name:
            self.report({"WARNING"}, "Select a valid object before inferring directions")
            return {"CANCELLED"}

        try:
            _run_direction_inference_into_scene(context, object_name)
        except Exception as error:
            scene.gen_direction_error = str(error)
            scene.gen_direction_inference_ready = False
            scene.gen_output = f"Status: Direction inference failed\n{error}"
            print(f"[Generator] Direction inference failed: {error}")
            print(traceback.format_exc())
            self.report({"WARNING"}, f"Direction inference failed: {error}")
            _tag_redraw()
            return {"CANCELLED"}

        self.report({"INFO"}, "Direction inference completed")
        return {"FINISHED"}


class GENERATOR_OT_generate(bpy.types.Operator):
    bl_label = "Generate Preview"
    bl_idname = "generator.generate"

    def execute(self, context):
        from . import preview_manager
        from .SceneParser import SceneParser

        scene = context.scene
        user_prompt = scene.gen_prompt

        if user_prompt == "Write your prompt here...":
            self.report({"WARNING"}, "Please enter a real prompt")
            return {"CANCELLED"}

        if context.active_object is None:
            self.report({"WARNING"}, "Select an object before generating")
            return {"CANCELLED"}

        existing_job = _get_generation_job(scene)
        if scene.gen_loading and existing_job is not None and existing_job["thread"].is_alive():
            self.report({"WARNING"}, "Generation is already running")
            return {"CANCELLED"}

        if scene.gen_loading and (existing_job is None or not existing_job["thread"].is_alive()):
            scene.gen_loading = False
            _clear_generation_job(scene)

        object_name = _resolve_target_object_name(context, scene)
        if not object_name:
            self.report({"WARNING"}, "Select a valid object before generating")
            return {"CANCELLED"}

        original_armature = bpy.data.objects.get(object_name)
        if original_armature is None:
            self.report({"ERROR"}, f"Armature `{object_name}` was not found")
            return {"CANCELLED"}
        if getattr(original_armature, "type", None) != "ARMATURE":
            self.report({"ERROR"}, f"Object `{object_name}` is not an armature")
            return {"CANCELLED"}

        needs_direction_inference = (
            scene.gen_auto_infer_directions
            and (
                not scene.gen_direction_inference_ready
                or scene.gen_direction_inferred_object_name != object_name
            )
        )
        if needs_direction_inference:
            try:
                _run_direction_inference_into_scene(context, object_name)
            except Exception as error:
                scene.gen_direction_error = str(error)
                scene.gen_direction_inference_ready = False
                scene.gen_output = f"Status: Direction inference failed\n{error}"
                print(f"[Generator] Direction inference failed: {error}")
                print(traceback.format_exc())
                self.report({"WARNING"}, f"Direction inference failed: {error}")
                _tag_redraw()
                return {"CANCELLED"}

            self.report({"INFO"}, "Direction inference completed; review settings before generating")
            return {"FINISHED"}

        direction_validation_error = _validate_scene_direction_settings(scene)
        if direction_validation_error:
            scene.gen_direction_error = direction_validation_error
            scene.gen_output = f"Status: Direction settings invalid\n{direction_validation_error}"
            self.report({"ERROR"}, direction_validation_error)
            _tag_redraw()
            return {"CANCELLED"}
        scene.gen_direction_error = ""
        direction_settings = _direction_settings_from_scene(scene)

        print(f"[Generator] Starting pipeline for object `{object_name}`.")
        print(f"[Generator] Prompt: {user_prompt}")

        parser = SceneParser(precision=1)

        try:
            print("[Generator] Stage 1: Parsing initial scene state on main thread.")
            object_json = parser.generate_object_json([object_name])
            object_json = _append_direction_context(object_json, direction_settings)
            print("[Generator] Initial scene parse completed.")
        except Exception as error:
            print(f"[Generator] Initial scene parse failed: {error}")
            self.report({"ERROR"}, f"Scene parse failed: {error}")
            return {"CANCELLED"}

        try:
            print("[Preview] Preparing non-destructive preview resources.")
            preview_state = preview_manager.prepare_preview(
                context,
                original_armature,
                owner_scene=scene,
            )
            print(
                "[Preview] Generation target is preview armature "
                f"`{preview_state['preview_armature_name']}`."
            )
        except Exception as error:
            print(f"[Preview] Preview setup failed: {error}")
            print(traceback.format_exc())
            self.report({"ERROR"}, f"Preview setup failed: {error}")
            return {"CANCELLED"}

        request_queue = queue.Queue()
        result_queue = queue.Queue()
        worker_thread = threading.Thread(
            target=_refinement_worker,
            args=(
                request_queue,
                result_queue,
                object_name,
                object_json,
                user_prompt,
                preview_state["preview_armature_name"],
            ),
            name="GradRefinementWorker",
            daemon=True,
        )

        job_state = {
            "scene": scene,
            "mode": "refinement",
            "object_name": object_name,
            "preview_scene_name": preview_state["preview_scene_name"],
            "preview_armature_name": preview_state["preview_armature_name"],
            "preview_action_name": preview_state["preview_action_name"],
            "preview_ready": True,
            "user_prompt": user_prompt,
            "request_queue": request_queue,
            "result_queue": result_queue,
            "thread": worker_thread,
            "parser": parser,
            "direction_settings": direction_settings,
            "executor": BlenderExecutor(),
            "steps": [],
            "current_step": 0,
            "total_steps": 4,
            "current_instruction": "",
            "status": "Generating plan",
            "status_base": "",
            "status_dots": 0,
            "waiting_for_llm": False,
            "previous_animation": None,
            "plan_history": [],
            "all_outputs": [],
            "final_animation": "",
            "final_plan": "",
            "final_keyframes": "",
            "final_feedback": None,
            "plan_text": "",
            "step_logs": [
                (
                    "Direction settings: "
                    f"forward={direction_settings['forward_axis']}, "
                    f"up={direction_settings['up_axis']}, "
                    f"right={direction_settings['right_axis']}"
                )
            ],
            "plan_ready": False,
        }

        scene.gen_loading = True
        _refresh_scene_output(scene, job_state)
        _set_generation_job(scene, job_state)

        worker_thread.start()
        _register_generation_timer(scene)
        _tag_redraw()

        self.report({"INFO"}, "Generation started")
        return {"FINISHED"}


class GENERATOR_OT_accept_preview(bpy.types.Operator):
    bl_label = "Accept Preview"
    bl_idname = "generator.accept_preview"

    def execute(self, context):
        from . import preview_manager

        scene = context.scene
        if scene.gen_loading:
            self.report({"WARNING"}, "Wait for generation to finish before accepting the preview")
            return {"CANCELLED"}

        if not _scene_has_preview(scene):
            self.report({"WARNING"}, "No preview is available to accept")
            return {"CANCELLED"}

        try:
            preview_manager.accept_preview(context, owner_scene=scene)
        except Exception as error:
            print(f"[Preview] Accept preview failed: {error}")
            print(traceback.format_exc())
            self.report({"ERROR"}, f"Accept preview failed: {error}")
            return {"CANCELLED"}

        self.report({"INFO"}, "Preview accepted")
        return {"FINISHED"}


class GENERATOR_OT_cancel_preview(bpy.types.Operator):
    bl_label = "Cancel Preview"
    bl_idname = "generator.cancel_preview"

    def execute(self, context):
        from . import preview_manager

        scene = context.scene
        existing_job = _get_generation_job(scene)
        if existing_job is not None:
            print("[Preview] Cancel requested while generation job is active.")
            _finalize_generation(scene, existing_job)

        if not _scene_has_preview(scene):
            self.report({"WARNING"}, "No preview is available to cancel")
            return {"CANCELLED"}

        try:
            preview_manager.cancel_preview(context, owner_scene=scene)
        except Exception as error:
            print(f"[Preview] Cancel preview failed: {error}")
            print(traceback.format_exc())
            self.report({"ERROR"}, f"Cancel preview failed: {error}")
            return {"CANCELLED"}

        self.report({"INFO"}, "Preview canceled")
        return {"FINISHED"}


class GENERATOR_PT_panel(bpy.types.Panel):
    bl_label = "Generator"
    bl_idname = "GENERATOR_PT_panel"
    bl_space_type = "VIEW_3D"
    bl_region_type = "UI"
    bl_category = "Generator"

    def draw(self, context):
        layout = self.layout
        scene = context.scene

        box = layout.box()
        box.label(text="Prompt", icon="TEXT")
        box.prop(scene, "gen_prompt", text="")
        box.prop(scene, "gen_mode")

        direction_box = layout.box()
        direction_box.label(text="Direction Settings", icon="ORIENTATION_LOCAL")
        direction_box.prop(scene, "gen_auto_infer_directions")
        if scene.gen_auto_infer_directions:
            infer_row = direction_box.row()
            infer_row.enabled = not scene.gen_loading
            infer_row.operator("generator.infer_directions", text="Infer Directions", icon="VIEW_CAMERA")
        direction_box.prop(scene, "gen_forward_axis")
        direction_box.prop(scene, "gen_up_axis")
        direction_box.prop(scene, "gen_right_axis")

        direction_error = scene.gen_direction_error or _validate_scene_direction_settings(scene)
        if direction_error:
            error_col = direction_box.column()
            error_col.alert = True
            for line in textwrap.wrap(direction_error, width=45):
                error_col.label(text=line, icon="ERROR")

        action_box = layout.box()
        generate_row = action_box.row()
        generate_row.enabled = not (direction_error and not scene.gen_auto_infer_directions)
        generate_row.operator("generator.generate", text="Generate Preview", icon="PLAY")

        preview_available = _scene_has_preview(scene)
        accept_row = action_box.row()
        accept_row.enabled = preview_available and not scene.gen_loading
        accept_row.operator("generator.accept_preview", text="Accept Preview", icon="CHECKMARK")

        cancel_row = action_box.row()
        cancel_row.enabled = preview_available
        cancel_row.operator("generator.cancel_preview", text="Cancel Preview", icon="CANCEL")

        result_box = layout.box()
        result_box.label(text="Result", icon="CONSOLE")

        if scene.gen_loading:
            result_box.label(text="\u23f3 Generating...")
            if scene.gen_output:
                result_box.separator(factor=0.5)

        if scene.gen_output:
            paragraphs = scene.gen_output.split("\n")

            for paragraph in paragraphs:
                if paragraph.strip():
                    wrapped_lines = textwrap.wrap(paragraph, width=45)

                    for line in wrapped_lines:
                        result_box.label(text=line)

                    result_box.separator(factor=0.5)
        elif not scene.gen_loading:
            result_box.label(text="Waiting for generation...")


classes = [
    GENERATOR_OT_infer_directions,
    GENERATOR_OT_generate,
    GENERATOR_OT_accept_preview,
    GENERATOR_OT_cancel_preview,
    GENERATOR_PT_panel,
]


def _on_auto_infer_directions_changed(self, context):
    self.gen_direction_inference_ready = False
    self.gen_direction_inferred_object_name = ""
    if not self.gen_auto_infer_directions:
        self.gen_direction_error = _validate_scene_direction_settings(self)


def _on_direction_axis_changed(self, context):
    self.gen_direction_error = _validate_scene_direction_settings(self)


def register():
    for cls in classes:
        bpy.utils.register_class(cls)

    bpy.types.Scene.gen_prompt = bpy.props.StringProperty(
        name="",
        description="Enter your prompt",
        default="Write your prompt here...",
    )

    bpy.types.Scene.gen_mode = bpy.props.EnumProperty(
        name="Object",
        description="Select Object",
        items=dropdown_items,
    )

    bpy.types.Scene.gen_auto_infer_directions = bpy.props.BoolProperty(
        name="Automatically Infer Directions",
        description="Use the vision direction helper to populate the direction controls",
        default=False,
        update=_on_auto_infer_directions_changed,
    )

    bpy.types.Scene.gen_forward_axis = bpy.props.EnumProperty(
        name="Forward Axis",
        description="Semantic forward axis for the selected object",
        items=_AXIS_ENUM_ITEMS,
        default="+Y",
        update=_on_direction_axis_changed,
    )

    bpy.types.Scene.gen_up_axis = bpy.props.EnumProperty(
        name="Up Axis",
        description="Semantic up axis for the selected object",
        items=_AXIS_ENUM_ITEMS,
        default="+Z",
        update=_on_direction_axis_changed,
    )

    bpy.types.Scene.gen_right_axis = bpy.props.EnumProperty(
        name="Right Axis",
        description="Semantic right axis for the selected object",
        items=_AXIS_ENUM_ITEMS,
        default="+X",
        update=_on_direction_axis_changed,
    )

    bpy.types.Scene.gen_direction_error = bpy.props.StringProperty(
        name="Direction Error",
        description="Direction settings validation or inference error",
        default="",
    )

    bpy.types.Scene.gen_direction_inference_ready = bpy.props.BoolProperty(
        name="Direction Inference Ready",
        description="True after AI inference has populated the direction controls for review",
        default=False,
    )

    bpy.types.Scene.gen_direction_inferred_object_name = bpy.props.StringProperty(
        name="Direction Inferred Object",
        description="Object name used for the last AI direction inference",
        default="",
    )

    bpy.types.Scene.gen_output = bpy.props.StringProperty(
        name="Output",
        description="LLM Result",
        default="",
    )

    bpy.types.Scene.gen_loading = bpy.props.BoolProperty(
        name="Generating",
        description="True while the generator is running",
        default=False,
    )

    bpy.types.Scene.preview_scene_name = bpy.props.StringProperty(
        name="Preview Scene",
        description="Name of the active AI preview scene",
        default="",
    )

    bpy.types.Scene.preview_armature_name = bpy.props.StringProperty(
        name="Preview Armature",
        description="Name of the active AI preview armature",
        default="",
    )

    bpy.types.Scene.preview_action_name = bpy.props.StringProperty(
        name="Preview Action",
        description="Name of the active AI preview action",
        default="",
    )

    bpy.types.Scene.preview_original_armature_name = bpy.props.StringProperty(
        name="Original Armature",
        description="Name of the original armature for the active preview",
        default="",
    )

    bpy.types.Scene.preview_active = bpy.props.BoolProperty(
        name="Preview Active",
        description="True while a generated preview is available",
        default=False,
    )


def unregister():
    del bpy.types.Scene.preview_active
    del bpy.types.Scene.preview_original_armature_name
    del bpy.types.Scene.preview_action_name
    del bpy.types.Scene.preview_armature_name
    del bpy.types.Scene.preview_scene_name
    del bpy.types.Scene.gen_direction_inferred_object_name
    del bpy.types.Scene.gen_direction_inference_ready
    del bpy.types.Scene.gen_direction_error
    del bpy.types.Scene.gen_right_axis
    del bpy.types.Scene.gen_up_axis
    del bpy.types.Scene.gen_forward_axis
    del bpy.types.Scene.gen_auto_infer_directions
    del bpy.types.Scene.gen_prompt
    del bpy.types.Scene.gen_mode
    del bpy.types.Scene.gen_output
    del bpy.types.Scene.gen_loading

    for cls in reversed(classes):
        bpy.utils.unregister_class(cls)


if __name__ == "__main__":
    register()
