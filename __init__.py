# This Blender add-on creates a sidebar panel with a text input for prompts,
# a dropdown to select modes (Rig, Animation, Modeling),
# and a button to generate content based on the prompt and selected mode.
# The operator prints the prompt and mode to the console when the button is clicked.


from .planner_agent import *
from .Blender_Executer import *
bl_info = {
    "name": "Generator Panel",
    "author": "Rana",
    "version": (1, 0),
    "blender": (4, 0, 0),
    "category": "3D View",
}

import bpy  # type: ignore
import textwrap
# ---------------------------------------------------
# Properties (Prompt + Dropdown)
# ---------------------------------------------------

def dropdown_items(self, context):
    return [
        
    ]


# ---------------------------------------------------
# Operator (Generate Button)
# ---------------------------------------------------

class GENERATOR_OT_generate(bpy.types.Operator):
    bl_label = "Generate"
    bl_idname = "generator.generate"

    def execute(self, context):
        import ast
        import os
        import re

        from .SceneParser import SceneParser

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

        user_prompt = context.scene.gen_prompt
        mode = context.scene.gen_mode

        if user_prompt == "Write your prompt here...":
            self.report({'WARNING'}, "Please enter a real prompt")
            return {'CANCELLED'}

        if context.active_object is None:
            self.report({'WARNING'}, "Select an object before generating")
            return {'CANCELLED'}

        object_name = "Human Male"
        print(f"[Generator] Starting pipeline for object `{object_name}`.")
        print(f"[Generator] Prompt: {user_prompt}")
        print(f"[Generator] Mode: {mode}")

        parser = SceneParser(precision=1)

        try:
            print("[Generator] Stage 1: Parsing initial scene state.")
            object_json = parser.generate_object_json(["SMPLX-lh-male"])
            print("[Generator] Initial scene parse completed.")
        except Exception as error:
            print(f"[Generator] Initial scene parse failed: {error}")
            self.report({'ERROR'}, f"Scene parse failed: {error}")
            return {'CANCELLED'}

        try:
            print("[Generator] Stage 2: Generating animation plan.")
            animation_plan = run_llm(object_name, object_json, user_prompt)
            context.scene.gen_output = str(animation_plan)
            print("[Generator] Animation plan generated:")
            print(animation_plan)
        except Exception as error:
            print(f"[Generator] Animation planning failed: {error}")
            self.report({'ERROR'}, f"Animation planning failed: {error}")
            return {'CANCELLED'}

        plan_steps = split_animation_plan(animation_plan)
        if not plan_steps:
            print("[Generator] Planner returned no executable steps.")
            self.report({'WARNING'}, "Animation planner returned no steps")
            return {'CANCELLED'}

        print(f"[Generator] Parsed {len(plan_steps)} plan step(s).")
        for index, step in enumerate(plan_steps, start=1):
            print(f"[Generator] Plan step {index}: {step}")

        try:
            print("[Generator] Stage 3: Initializing keyframe agent.")
            KeyFrameAgent = load_keyframe_agent_class()
            keyframe = KeyFrameAgent()
            keyframe.initialize_chain()
            print("[Generator] Keyframe agent initialized.")
        except Exception as error:
            print(f"[Generator] Keyframe agent initialization failed: {error}")
            self.report({'ERROR'}, f"Keyframe agent initialization failed: {error}")
            return {'CANCELLED'}

        previous_animation = None
        executor = BlenderExecutor()
        for index, step in enumerate(plan_steps, start=1):
            print(f"[Generator] Processing step {index}/{len(plan_steps)}.")

            if index > 1:
                try:
                    print(f"[Generator] Refreshing scene state before step {index}.")
                    object_json = parser.generate_object_json(["SMPLX-lh-male"])
                    print(f"[Generator] Scene state refreshed before step {index}.")
                except Exception as error:
                    print(f"[Generator] Scene refresh failed before step {index}: {error}")
                    continue

            response = None

            try:
                print(f"[Generator] Stage 4.{index}: Generating keyframes.")
                response = keyframe.invoke_chain(
                    {
                        "object": object_name,
                        "object_json": object_json,
                        "instruction": step,
                        "previous_animation": previous_animation,
                    }
                )
                print(f"[Generator] Keyframe response for step {index}:")
                print(response)
            except Exception as error:
                print(f"[Generator] Keyframe generation failed for step {index}: {error}")

            if response is None:
                continue

            try:
                print(f"[Generator] Stage 5.{index}: Executing keyframes in Blender.")
                executor.execute_from_text(response)
                print(f"[Generator] Blender execution completed for step {index}.")
            except Exception as error:
                print(f"[Generator] Blender execution failed for step {index}: {error}")

            previous_animation = response
            print(f"[Generator] Updated previous_animation after step {index}.")

            try:
                print(f"[Generator] Refreshing scene state after step {index} execution.")
                object_json = parser.generate_object_json(["SMPLX-lh-male"])
                print(f"[Generator] Scene state refreshed after step {index}.")
                print(f"[Generator] Updated object_json after step {index}: {object_json}")
            except Exception as error:
                print(f"[Generator] Scene refresh failed after step {index}: {error}")

        self.report({'INFO'}, f"Generated with {mode}")
        return {'FINISHED'}

# ---------------------------------------------------
# Panel (Sidebar UI)
# ---------------------------------------------------

class GENERATOR_PT_panel(bpy.types.Panel):
    bl_label = "Generator"
    bl_idname = "GENERATOR_PT_panel"
    bl_space_type = 'VIEW_3D'
    bl_region_type = 'UI'
    bl_category = "Generator"   # Sidebar tab name

    def draw(self, context):
        layout = self.layout
        scene = context.scene

        # --- Prompt Box ---
        box = layout.box()
        box.label(text="Prompt", icon="TEXT")
        box.prop(scene, "gen_prompt", text="")
        box.prop(scene, "gen_mode")
        box.operator("generator.generate", icon="PLAY")
        
        # --- Result Box ---
        result_box = layout.box()
        result_box.label(text="Result", icon="CONSOLE")
        
        # Check if there is any output to display
        if scene.gen_output:
            # 1. Split the LLM output by the newlines (\n) it generates
            paragraphs = scene.gen_output.split('\n')
            
            for paragraph in paragraphs:
                if paragraph.strip(): # Skip empty lines
                    # 2. Wrap each line at ~45 characters so it fits the sidebar width
                    wrapped_lines = textwrap.wrap(paragraph, width=45)
                    
                    # 3. Draw each wrapped line as a new label
                    for line in wrapped_lines:
                        result_box.label(text=line)
                    
                    # Add a tiny bit of vertical space between steps for readability
                    result_box.separator(factor=0.5)
        else:
            result_box.label(text="Waiting for generation...")

# ---------------------------------------------------
# Register
# ---------------------------------------------------

classes = [
    GENERATOR_OT_generate,
    GENERATOR_PT_panel,
]

def register():
    # Register classes first
    for cls in classes:
        bpy.utils.register_class(cls)

    # Then register scene properties
    bpy.types.Scene.gen_prompt = bpy.props.StringProperty(
        name="",
        description="Enter your prompt",
        default="Write your prompt here..."
    )

    bpy.types.Scene.gen_mode = bpy.props.EnumProperty(
        name="Mode",
        description="Select Mode",
        items=dropdown_items
    )

    bpy.types.Scene.gen_output = bpy.props.StringProperty(
        name="Output",
        description="LLM Result",
        default="",
    )

def unregister():
    # Remove properties first
    del bpy.types.Scene.gen_prompt
    del bpy.types.Scene.gen_mode
    del bpy.types.Scene.gen_output

    # Then unregister classes
    for cls in reversed(classes):
        bpy.utils.unregister_class(cls)

if __name__ == "__main__":
    register()
