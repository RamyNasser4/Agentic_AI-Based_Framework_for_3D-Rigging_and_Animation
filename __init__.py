import ast
import json
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
    "version": (1, 1),
    "blender": (4, 0, 0),
    "category": "3D View",
}


_GENERATION_JOBS = {}
_GENERATION_JOBS_LOCK = threading.Lock()
_REFINEMENT_JOBS = {}
_REFINEMENT_JOBS_LOCK = threading.Lock()
_TIMER_INTERVAL = 0.1
MAX_REFINEMENT_QUESTIONS = 10


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


def _get_refinement_job(scene):
    with _REFINEMENT_JOBS_LOCK:
        return _REFINEMENT_JOBS.get(_scene_job_key(scene))


def _set_refinement_job(scene, job_state):
    with _REFINEMENT_JOBS_LOCK:
        _REFINEMENT_JOBS[_scene_job_key(scene)] = job_state


def _clear_refinement_job(scene):
    with _REFINEMENT_JOBS_LOCK:
        _REFINEMENT_JOBS.pop(_scene_job_key(scene), None)


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


def _get_refinement_history(scene):
    raw_history = getattr(scene, "gen_refinement_history", "")
    if not raw_history:
        return []

    try:
        history = json.loads(raw_history)
    except json.JSONDecodeError:
        return []

    if not isinstance(history, list):
        return []

    cleaned_history = []
    for item in history:
        if not isinstance(item, dict):
            continue
        question = str(item.get("question") or "").strip()
        answer = str(item.get("answer") or "").strip()
        if question and answer:
            cleaned_history.append({"question": question, "answer": answer})

    return cleaned_history


def _set_refinement_history(scene, history):
    scene.gen_refinement_history = json.dumps(history or [], ensure_ascii=True)


def _format_clarification_history(clarification_history):
    if not clarification_history:
        return "None."

    lines = []
    for index, item in enumerate(clarification_history, start=1):
        if not isinstance(item, dict):
            continue
        question = str(item.get("question") or "").strip()
        answer = str(item.get("answer") or "").strip()
        if question and answer:
            lines.append(f"{index}. Q: {question} A: {answer}")

    return "\n".join(lines) if lines else "None."


def _build_refined_prompt(original_prompt, clarification_history):
    prompt = str(original_prompt or "").strip()
    answered_items = []

    for item in clarification_history or []:
        if not isinstance(item, dict):
            continue

        question = str(item.get("question") or "").strip()
        answer = str(item.get("answer") or "").strip().title()
        if not question or answer == "Skip":
            continue

        answered_items.append(f"- {question} Answer: {answer}.")

    if not answered_items:
        return prompt

    return (
        f"{prompt}\n\n"
        "Resolved animation clarifications that are part of this request:\n"
        + "\n".join(answered_items)
    ).strip()


def _reset_refinement_state(scene, status="Not refined"):
    scene.gen_current_question = ""
    scene.gen_refined_prompt = ""
    scene.gen_refinement_status = status
    scene.gen_refinement_loading = False
    scene.gen_refinement_prompt_snapshot = ""
    scene.gen_refinement_object_snapshot = ""
    scene.gen_refinement_object_json_snapshot = ""
    _set_refinement_history(scene, [])


def _finish_refinement(scene, history, final_prompt=None):
    scene.gen_current_question = ""
    scene.gen_refined_prompt = str(final_prompt or "").strip() or _build_refined_prompt(
        scene.gen_prompt,
        history,
    )
    scene.gen_refinement_status = "Complete"
    scene.gen_refinement_loading = False
    scene.gen_refinement_prompt_snapshot = str(scene.gen_prompt or "").strip()
    _tag_redraw()


def _refinement_worker(result_queue, prompt_text, object_name, object_json, history):
    try:
        from .prompt_refinement_agent import PromptRefinementAgent

        if len(history) >= MAX_REFINEMENT_QUESTIONS:
            agent = PromptRefinementAgent()
            final_prompt = agent.synthesize_final_prompt(prompt_text, history)
            result_queue.put({"type": "stop", "final_prompt": final_prompt})
            return

        agent = PromptRefinementAgent()
        current_prompt = _build_refined_prompt(prompt_text, history)
        decision = agent.decide(
            current_prompt=current_prompt,
            object_name=object_name,
            object_json=object_json,
            clarification_history=history,
            question_count=len(history),
        )

        if decision["status"] == "QUESTION":
            result_queue.put({"type": "question", "question": decision["question"]})
            return

        final_prompt = agent.synthesize_final_prompt(prompt_text, history)
        result_queue.put({"type": "stop", "final_prompt": final_prompt})
    except Exception as error:
        result_queue.put({"type": "error", "message": str(error)})


def _resolve_refinement_object_context(context, scene, reset=False):
    from .SceneParser import SceneParser

    current_object_name = _resolve_target_object_name(context, scene)
    snapshot_object_name = str(getattr(scene, "gen_refinement_object_snapshot", "") or "").strip()
    snapshot_object_json = str(getattr(scene, "gen_refinement_object_json_snapshot", "") or "").strip()

    if reset:
        if not current_object_name:
            raise ValueError("Select an object before refining.")
        parser = SceneParser(precision=1)
        object_json = parser.generate_object_json([current_object_name])
        scene.gen_refinement_object_snapshot = current_object_name
        scene.gen_refinement_object_json_snapshot = object_json
        return current_object_name, object_json

    if not snapshot_object_name:
        if not current_object_name:
            raise ValueError("Select an object before refining.")
        parser = SceneParser(precision=1)
        object_json = parser.generate_object_json([current_object_name])
        scene.gen_refinement_object_snapshot = current_object_name
        scene.gen_refinement_object_json_snapshot = object_json
        return current_object_name, object_json

    if current_object_name and current_object_name != snapshot_object_name:
        raise ValueError("Object changed; run Prompt Refinement again for the selected object.")

    if not snapshot_object_json:
        parser = SceneParser(precision=1)
        snapshot_object_json = parser.generate_object_json([snapshot_object_name])
        scene.gen_refinement_object_json_snapshot = snapshot_object_json

    return snapshot_object_name, snapshot_object_json


def _start_refinement_job(scene, context, reset=False):
    prompt_text = str(scene.gen_prompt or "").strip()
    if not prompt_text or prompt_text == "Write your prompt here...":
        raise ValueError("Please enter a real prompt before refining.")

    existing_job = _get_refinement_job(scene)
    if scene.gen_refinement_loading and existing_job is not None and existing_job["thread"].is_alive():
        raise RuntimeError("Prompt refinement is already running.")

    if reset:
        _set_refinement_history(scene, [])
        scene.gen_current_question = ""
        scene.gen_refined_prompt = ""
        scene.gen_refinement_prompt_snapshot = prompt_text
        scene.gen_refinement_object_snapshot = ""
        scene.gen_refinement_object_json_snapshot = ""

    object_name, object_json = _resolve_refinement_object_context(context, scene, reset=reset)
    history = _get_refinement_history(scene)

    result_queue = queue.Queue()
    worker_thread = threading.Thread(
        target=_refinement_worker,
        args=(result_queue, prompt_text, object_name, object_json, history),
        name="PromptRefinementWorker",
        daemon=True,
    )

    job_state = {
        "thread": worker_thread,
        "result_queue": result_queue,
        "prompt_snapshot": prompt_text,
        "object_snapshot": object_name,
        "history": history,
        "status_base": "Refining prompt",
        "status_dots": 0,
    }

    scene.gen_refinement_loading = True
    scene.gen_refinement_status = "Refining prompt"
    _set_refinement_job(scene, job_state)
    worker_thread.start()
    _register_refinement_timer(scene)

    _tag_redraw()


def _handle_refinement_queue(scene):
    job_state = _get_refinement_job(scene)
    if job_state is None:
        scene.gen_refinement_loading = False
        _tag_redraw()
        return None

    result_queue = job_state["result_queue"]

    try:
        item = result_queue.get_nowait()
    except queue.Empty:
        if not job_state["thread"].is_alive():
            scene.gen_refinement_loading = False
            scene.gen_refinement_status = "Refinement worker stopped"
            _clear_refinement_job(scene)
            _tag_redraw()
            return None

        dots = "." * (job_state.get("status_dots", 0) % 4)
        job_state["status_dots"] = job_state.get("status_dots", 0) + 1
        scene.gen_refinement_status = f"{job_state['status_base']}{dots}"
        _tag_redraw()
        return _TIMER_INTERVAL

    if str(scene.gen_prompt or "").strip() != job_state["prompt_snapshot"]:
        scene.gen_refinement_loading = False
        scene.gen_refinement_status = "Prompt changed; refine again"
        scene.gen_current_question = ""
        scene.gen_refined_prompt = ""
        _clear_refinement_job(scene)
        _tag_redraw()
        return None

    item_type = item["type"]
    history = _get_refinement_history(scene)

    if item_type == "question":
        scene.gen_current_question = item["question"]
        scene.gen_refined_prompt = ""
        scene.gen_refinement_loading = False
        scene.gen_refinement_status = f"Question {len(history) + 1}/{MAX_REFINEMENT_QUESTIONS}"
        _clear_refinement_job(scene)
        _tag_redraw()
        return None

    if item_type == "stop":
        _finish_refinement(scene, history, final_prompt=item.get("final_prompt"))
        _clear_refinement_job(scene)
        return None

    if item_type == "error":
        scene.gen_refinement_loading = False
        scene.gen_refinement_status = f"Refinement failed: {item.get('message')}"
        scene.gen_current_question = ""
        scene.gen_refined_prompt = ""
        _clear_refinement_job(scene)
        _tag_redraw()
        return None

    return _TIMER_INTERVAL


def _generation_worker(request_queue, result_queue, object_name, object_json, prompt, clarification_history):
    try:
        print(f"[Generator] Worker started for object `{object_name}`.")
        print("[Generator] Stage 2: Generating animation plan in background thread.")
        animation_plan = run_llm(
            object_name,
            object_json,
            prompt,
            clarification_history=clarification_history,
        )
        plan_steps = split_animation_plan(animation_plan)
#         animation_plan = [
#         "a man picks up an unseen object to his front left and moves it to an unseen platform on this front right without moving his feet."
# ]
#         plan_steps = [
#         "a man picks up an unseen object to his front left and moves it to an unseen platform on this front right without moving his feet."
#     ]
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
            instruction = request["step"]
            previous_animation = request.get("previous_animation")

            print(f"[Generator] Worker generating keyframes for step {index}.")

            try:
                response = keyframe.invoke_chain(
                    {
                        "object": object_name,
                        "object_json": request["object_json"],
                        "instruction": instruction,
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
                    step=instruction,
                )
                continue

            if response is None:
                print(f"[Generator] Keyframe generation returned no result for step {index}.")
                _queue_worker_error(
                    result_queue,
                    f"Keyframe generation returned no result for step {index}.",
                    item_type="step_error",
                    index=index,
                    step=instruction,
                )
                continue

            result_queue.put(
                {
                    "type": "step_result",
                    "index": index,
                    "step": instruction,
                    "response": response,
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
        job_state["previous_animation"] = response
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

    job_state["current_instruction"] = job_state["steps"][job_state["current_step"]]
    _set_status(job_state, f"Step {index} complete")
    _refresh_scene_output(scene, job_state)


def _dispatch_next_step(scene, job_state):
    if job_state["waiting_for_llm"]:
        return
    if not job_state["plan_ready"]:
        return
    if job_state["current_step"] >= job_state["total_steps"]:
        return

    step_index = job_state["current_step"] + 1
    instruction = job_state["steps"][job_state["current_step"]]
    job_state["current_instruction"] = instruction

    _set_status(job_state, f"Parsing scene for Step {step_index}")
    _refresh_scene_output(scene, job_state)

    try:
        object_json = job_state["parser"].generate_object_json([job_state["object_name"]])
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
            "step": instruction,
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

        if job_state["plan_ready"] and job_state["current_step"] < job_state["total_steps"]:
            _dispatch_next_step(scene, job_state)
            return _TIMER_INTERVAL if _get_generation_job(scene) is not None else None

        return _TIMER_INTERVAL

    item_type = item["type"]

    if item_type == "plan":
        job_state["plan_text"] = item["plan"]
        job_state["steps"] = item["steps"]
        job_state["total_steps"] = len(item["steps"])
        job_state["plan_ready"] = True
        job_state["current_instruction"] = job_state["steps"][0] if job_state["steps"] else ""
        job_state["step_logs"].append(f"Planner returned {job_state['total_steps']} step(s).")
        _set_status(job_state, f"Plan ready ({job_state['total_steps']} steps)")
        _refresh_scene_output(scene, job_state)
        return _TIMER_INTERVAL

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


def _register_refinement_timer(scene):
    def timer_callback():
        return _handle_refinement_queue(scene)

    bpy.app.timers.register(timer_callback, first_interval=_TIMER_INTERVAL)


def _resolve_target_object_name(context, scene):
    if getattr(scene, "gen_mode", ""):
        return scene.gen_mode
    if context.active_object is not None:
        return context.active_object.name
    return ""


class GENERATOR_OT_refine_prompt(bpy.types.Operator):
    bl_label = "Refine Prompt"
    bl_idname = "generator.refine_prompt"

    def execute(self, context):
        scene = context.scene
        try:
            _start_refinement_job(scene, context, reset=True)
        except Exception as error:
            scene.gen_refinement_status = f"Refinement failed: {error}"
            scene.gen_refinement_loading = False
            scene.gen_current_question = ""
            scene.gen_refined_prompt = ""
            _tag_redraw()
            self.report({"ERROR"}, f"Prompt refinement failed: {error}")
            return {"CANCELLED"}

        self.report({"INFO"}, "Prompt refinement started")
        return {"FINISHED"}


class GENERATOR_OT_answer_refinement(bpy.types.Operator):
    bl_label = "Answer Refinement"
    bl_idname = "generator.answer_refinement"

    answer: bpy.props.StringProperty(default="Skip")

    def execute(self, context):
        scene = context.scene
        if scene.gen_refinement_loading:
            self.report({"WARNING"}, "Prompt refinement is already running")
            return {"CANCELLED"}

        question = str(scene.gen_current_question or "").strip()
        if not question:
            self.report({"WARNING"}, "There is no refinement question to answer")
            return {"CANCELLED"}

        if scene.gen_refinement_prompt_snapshot != str(scene.gen_prompt or "").strip():
            scene.gen_refinement_status = "Prompt changed; refine again"
            scene.gen_refinement_loading = False
            scene.gen_current_question = ""
            scene.gen_refined_prompt = ""
            _set_refinement_history(scene, [])
            _tag_redraw()
            self.report({"WARNING"}, "Prompt changed; run Refine Prompt again")
            return {"CANCELLED"}

        current_object_name = _resolve_target_object_name(context, scene)
        snapshot_object_name = str(scene.gen_refinement_object_snapshot or "").strip()
        if snapshot_object_name and current_object_name and current_object_name != snapshot_object_name:
            scene.gen_refinement_status = "Object changed; refine again"
            scene.gen_refinement_loading = False
            scene.gen_current_question = ""
            scene.gen_refined_prompt = ""
            _set_refinement_history(scene, [])
            _tag_redraw()
            self.report({"WARNING"}, "Object changed; run Refine Prompt again")
            return {"CANCELLED"}

        answer = str(self.answer or "Skip").strip().title()
        if answer not in {"Yes", "No", "Skip"}:
            self.report({"WARNING"}, "Answer must be Yes, No, or Skip")
            return {"CANCELLED"}

        history = _get_refinement_history(scene)
        history.append({"question": question, "answer": answer})
        _set_refinement_history(scene, history)
        scene.gen_current_question = ""

        try:
            _start_refinement_job(scene, context, reset=False)
        except Exception as error:
            scene.gen_refinement_status = f"Refinement failed: {error}"
            scene.gen_refinement_loading = False
            scene.gen_current_question = ""
            scene.gen_refined_prompt = ""
            _tag_redraw()
            self.report({"ERROR"}, f"Prompt refinement failed: {error}")
            return {"CANCELLED"}

        self.report({"INFO"}, "Prompt refinement continuing")
        return {"FINISHED"}


class GENERATOR_OT_generate(bpy.types.Operator):
    bl_label = "Generate"
    bl_idname = "generator.generate"

    def execute(self, context):
        from .SceneParser import SceneParser

        scene = context.scene
        original_prompt = str(scene.gen_prompt or "").strip()
        user_prompt = str(scene.gen_refined_prompt or "").strip()
        clarification_history = _get_refinement_history(scene)

        if original_prompt == "Write your prompt here...":
            self.report({"WARNING"}, "Please enter a real prompt")
            return {"CANCELLED"}

        if scene.gen_refinement_loading:
            self.report({"WARNING"}, "Wait for Prompt Refinement to finish")
            return {"CANCELLED"}

        if not user_prompt or scene.gen_refinement_prompt_snapshot != original_prompt:
            self.report({"WARNING"}, "Run Prompt Refinement for the current prompt before generating")
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

        refinement_object_name = str(scene.gen_refinement_object_snapshot or "").strip()
        if not refinement_object_name or refinement_object_name != object_name:
            self.report({"WARNING"}, "Run Prompt Refinement for the selected object before generating")
            return {"CANCELLED"}

        print(f"[Generator] Starting pipeline for object `{object_name}`.")
        print(f"[Generator] Original prompt: {original_prompt}")
        print(f"[Generator] Final refined prompt sent to planner: {user_prompt}")
        print(f"[Generator] Clarification history: {_format_clarification_history(clarification_history)}")

        parser = SceneParser(precision=1)

        try:
            print("[Generator] Stage 1: Parsing initial scene state on main thread.")
            object_json = parser.generate_object_json([object_name])
            print("[Generator] Initial scene parse completed.")
        except Exception as error:
            print(f"[Generator] Initial scene parse failed: {error}")
            self.report({"ERROR"}, f"Scene parse failed: {error}")
            return {"CANCELLED"}

        request_queue = queue.Queue()
        result_queue = queue.Queue()
        worker_thread = threading.Thread(
            target=_generation_worker,
            args=(
                request_queue,
                result_queue,
                object_name,
                object_json,
                user_prompt,
                clarification_history,
            ),
            name="GradGeneratorWorker",
            daemon=True,
        )

        job_state = {
            "scene": scene,
            "object_name": object_name,
            "request_queue": request_queue,
            "result_queue": result_queue,
            "thread": worker_thread,
            "parser": parser,
            "executor": BlenderExecutor(),
            "steps": [],
            "current_step": 0,
            "total_steps": 0,
            "current_instruction": "",
            "status": "Generating plan",
            "status_base": "",
            "status_dots": 0,
            "waiting_for_llm": False,
            "previous_animation": None,
            "plan_text": "",
            "step_logs": [],
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


class GENERATOR_PT_panel(bpy.types.Panel):
    bl_label = "Generator"
    bl_idname = "GENERATOR_PT_panel"
    bl_space_type = "VIEW_3D"
    bl_region_type = "UI"
    bl_category = "Generator"

    def draw(self, context):
        layout = self.layout
        scene = context.scene

        refine_box = layout.box()
        refine_box.label(text="Prompt Refinement", icon="QUESTION")
        refine_box.prop(scene, "gen_prompt", text="")
        refine_box.operator("generator.refine_prompt", icon="FILE_REFRESH")
        refine_box.label(text=f"Status: {scene.gen_refinement_status or 'Not refined'}")
        if scene.gen_refinement_loading:
            refine_box.label(text="Refining prompt...")

        if scene.gen_current_question:
            refine_box.separator(factor=0.5)
            refine_box.label(text="Current Question")
            for line in textwrap.wrap(scene.gen_current_question, width=45):
                refine_box.label(text=line)

            answer_row = refine_box.row(align=True)
            yes_op = answer_row.operator("generator.answer_refinement", text="Yes")
            yes_op.answer = "Yes"
            no_op = answer_row.operator("generator.answer_refinement", text="No")
            no_op.answer = "No"
            skip_op = answer_row.operator("generator.answer_refinement", text="Skip")
            skip_op.answer = "Skip"

        history = _get_refinement_history(scene)
        if history:
            refine_box.separator(factor=0.5)
            refine_box.label(text="Clarification History")
            for item in history:
                history_line = f"Q: {item['question']} A: {item['answer']}"
                for line in textwrap.wrap(history_line, width=45):
                    refine_box.label(text=line)

        if scene.gen_refined_prompt:
            refine_box.separator(factor=0.5)
            refine_box.label(text="Final Refined Prompt")
            for paragraph in scene.gen_refined_prompt.split("\n"):
                if not paragraph.strip():
                    continue
                for line in textwrap.wrap(paragraph, width=45):
                    refine_box.label(text=line)

        box = layout.box()
        box.label(text="Generation", icon="PLAY")
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
    GENERATOR_OT_refine_prompt,
    GENERATOR_OT_answer_refinement,
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
        name="Object",
        description="Select Object",
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

    bpy.types.Scene.gen_current_question = bpy.props.StringProperty(
        name="Current Question",
        description="Current prompt refinement question",
        default="",
    )

    bpy.types.Scene.gen_refinement_history = bpy.props.StringProperty(
        name="Clarification History",
        description="Prompt refinement clarification history",
        default="[]",
    )

    bpy.types.Scene.gen_refinement_status = bpy.props.StringProperty(
        name="Refinement Status",
        description="Prompt refinement status",
        default="Not refined",
    )

    bpy.types.Scene.gen_refinement_loading = bpy.props.BoolProperty(
        name="Refining",
        description="True while prompt refinement is running",
        default=False,
    )

    bpy.types.Scene.gen_refined_prompt = bpy.props.StringProperty(
        name="Final Refined Prompt",
        description="Final prompt sent to the planner after refinement",
        default="",
    )

    bpy.types.Scene.gen_refinement_prompt_snapshot = bpy.props.StringProperty(
        name="Refinement Prompt Snapshot",
        description="Original prompt text used for the current refined prompt",
        default="",
    )

    bpy.types.Scene.gen_refinement_object_snapshot = bpy.props.StringProperty(
        name="Refinement Object Snapshot",
        description="Object name used for the current prompt refinement",
        default="",
    )

    bpy.types.Scene.gen_refinement_object_json_snapshot = bpy.props.StringProperty(
        name="Refinement Object JSON Snapshot",
        description="Object JSON rig hierarchy used for the current prompt refinement",
        default="",
    )


def unregister():
    del bpy.types.Scene.gen_prompt
    del bpy.types.Scene.gen_mode
    del bpy.types.Scene.gen_output
    del bpy.types.Scene.gen_loading
    del bpy.types.Scene.gen_current_question
    del bpy.types.Scene.gen_refinement_history
    del bpy.types.Scene.gen_refinement_status
    del bpy.types.Scene.gen_refinement_loading
    del bpy.types.Scene.gen_refined_prompt
    del bpy.types.Scene.gen_refinement_prompt_snapshot
    del bpy.types.Scene.gen_refinement_object_snapshot
    del bpy.types.Scene.gen_refinement_object_json_snapshot

    for cls in reversed(classes):
        bpy.utils.unregister_class(cls)


if __name__ == "__main__":
    register()
