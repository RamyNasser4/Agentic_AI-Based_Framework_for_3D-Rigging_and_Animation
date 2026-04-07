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


def dropdown_items(self, context):
    items = []

    for obj in context.scene.objects:
        if obj.type == "MESH" or obj.type == "ARMATURE":
            items.append((obj.name, obj.name, ""))

    return items


def split_animation_plan(plan_text):
    lines = [line.strip() for line in str(plan_text).splitlines() if line.strip()]
    if not lines:
        return []

    grouped_steps = []
    current_step = []
    step_header = re.compile(r"^(?:step\s*\d+\s*:|\d+[\.\)])\s*", re.IGNORECASE)

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
        allowed_assignments = {"SYSTEM_MESSAGE", "animation_examples"}

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


def _build_output_text(job_state):
    parts = []

    if job_state["plan_text"]:
        parts.append(job_state["plan_text"])

    if job_state["step_logs"]:
        if parts:
            parts.append("")
        parts.extend(job_state["step_logs"])

    return "\n".join(parts).strip()


def _refresh_scene_output(scene, job_state):
    scene.gen_output = _build_output_text(job_state)
    _tag_redraw()


def _finalize_generation(scene):
    scene.gen_loading = False
    _clear_generation_job(scene)
    _tag_redraw()


def _queue_worker_error(result_queue, message):
    result_queue.put(
        {
            "type": "error",
            "message": message,
        }
    )


def _generation_worker(result_queue, object_name, object_json, prompt, mode):
    try:
        print(f"[Generator] Worker started for mode `{mode}`.")
        print("[Generator] Stage 2: Generating animation plan in background thread.")
        animation_plan = run_llm(object_name, object_json, prompt)
        plan_steps = split_animation_plan(animation_plan)

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

        previous_animation = None
        generated_results = []

        for index, step in enumerate(plan_steps, start=1):
            print(f"[Generator] Worker generating keyframes for step {index}/{len(plan_steps)}.")

            try:
                response = keyframe.invoke_chain(
                    {
                        "object": object_name,
                        "object_json": object_json,
                        "instruction": step,
                        "previous_animation": previous_animation,
                    }
                )
                print(f"[Generator] Worker generated response for step {index}.")
            except Exception as error:
                generated_results.append(
                    {
                        "type": "step_error",
                        "index": index,
                        "step": step,
                        "message": f"Keyframe generation failed for step {index}: {error}",
                    }
                )
                print(f"[Generator] Keyframe generation failed for step {index}: {error}")
                continue

            if response is None:
                generated_results.append(
                    {
                        "type": "step_error",
                        "index": index,
                        "step": step,
                        "message": f"Keyframe generation returned no result for step {index}.",
                    }
                )
                print(f"[Generator] Keyframe generation returned no result for step {index}.")
                continue

            generated_results.append(
                {
                    "type": "step_result",
                    "index": index,
                    "step": step,
                    "response": response,
                }
            )
            previous_animation = response

        for item in generated_results:
            result_queue.put(item)

        result_queue.put(
            {
                "type": "done",
                "mode": mode,
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

    try:
        print(f"[Generator] Stage 5.{index}: Executing keyframes in Blender.")
        job_state["executor"].execute_from_text(response)
        print(f"[Generator] Blender execution completed for step {index}.")
        job_state["step_logs"].append(f"Executed Step {index}: {step}")
    except Exception as error:
        print(f"[Generator] Blender execution failed for step {index}: {error}")
        job_state["step_logs"].append(f"Execution failed for Step {index}: {error}")

    _refresh_scene_output(scene, job_state)


def _handle_generation_queue(scene):
    job_state = _get_generation_job(scene)
    if job_state is None:
        scene.gen_loading = False
        _tag_redraw()
        return None

    result_queue = job_state["queue"]

    try:
        item = result_queue.get_nowait()
    except queue.Empty:
        if job_state["thread"].is_alive():
            return _TIMER_INTERVAL

        if not job_state["done"]:
            job_state["step_logs"].append("Generation stopped before completion.")
            _refresh_scene_output(scene, job_state)

        _finalize_generation(scene)
        return None

    item_type = item["type"]

    if item_type == "plan":
        job_state["plan_text"] = item["plan"]
        _refresh_scene_output(scene, job_state)
        return _TIMER_INTERVAL

    if item_type == "step_result":
        _execute_generation_step(scene, job_state, item)
        return _TIMER_INTERVAL

    if item_type == "step_error":
        job_state["step_logs"].append(item["message"])
        _refresh_scene_output(scene, job_state)
        return _TIMER_INTERVAL

    if item_type == "error":
        job_state["step_logs"].append(item["message"])
        _refresh_scene_output(scene, job_state)
        _finalize_generation(scene)
        return None

    if item_type == "done":
        job_state["done"] = True
        if not job_state["step_logs"]:
            job_state["step_logs"].append(f"Generated with {item['mode']}.")
        _refresh_scene_output(scene, job_state)
        _finalize_generation(scene)
        return None

    return _TIMER_INTERVAL


def _register_generation_timer(scene):
    def timer_callback():
        return _handle_generation_queue(scene)

    bpy.app.timers.register(timer_callback, first_interval=_TIMER_INTERVAL)


class GENERATOR_OT_generate(bpy.types.Operator):
    bl_label = "Generate"
    bl_idname = "generator.generate"

    def execute(self, context):
        from .SceneParser import SceneParser

        scene = context.scene
        user_prompt = scene.gen_prompt
        mode = scene.gen_mode

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

        object_name = "Human Male"
        print(f"[Generator] Starting pipeline for object `{object_name}`.")
        print(f"[Generator] Prompt: {user_prompt}")
        print(f"[Generator] Mode: {mode}")

        parser = SceneParser(precision=1)

        try:
            print("[Generator] Stage 1: Parsing initial scene state on main thread.")
            object_json = parser.generate_object_json(["SMPLX-lh-male"])
            print("[Generator] Initial scene parse completed.")
        except Exception as error:
            print(f"[Generator] Initial scene parse failed: {error}")
            self.report({"ERROR"}, f"Scene parse failed: {error}")
            return {"CANCELLED"}

        result_queue = queue.Queue()
        worker_thread = threading.Thread(
            target=_generation_worker,
            args=(result_queue, object_name, object_json, user_prompt, mode),
            name="GradGeneratorWorker",
            daemon=True,
        )

        job_state = {
            "scene": scene,
            "queue": result_queue,
            "thread": worker_thread,
            "executor": BlenderExecutor(),
            "plan_text": "",
            "step_logs": [],
            "done": False,
        }

        scene.gen_loading = True
        scene.gen_output = ""
        _set_generation_job(scene, job_state)

        worker_thread.start()
        _register_generation_timer(scene)
        _tag_redraw()

        self.report({"INFO"}, "Generation started")
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
        box.operator("generator.generate", icon="PLAY")

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
    GENERATOR_OT_generate,
    GENERATOR_PT_panel,
]


def register():
    for cls in classes:
        bpy.utils.register_class(cls)

    bpy.types.Scene.gen_prompt = bpy.props.StringProperty(
        name="",
        description="Enter your prompt",
        default="Write your prompt here...",
    )

    bpy.types.Scene.gen_mode = bpy.props.EnumProperty(
        name="Mode",
        description="Select Mode",
        items=dropdown_items,
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


def unregister():
    del bpy.types.Scene.gen_prompt
    del bpy.types.Scene.gen_mode
    del bpy.types.Scene.gen_output
    del bpy.types.Scene.gen_loading

    for cls in reversed(classes):
        bpy.utils.unregister_class(cls)


if __name__ == "__main__":
    register()
