# This Blender add-on creates a sidebar panel with a text input for prompts,
# a dropdown to select modes (Rig, Animation, Modeling),
# and a button to generate content based on the prompt and selected mode.
# The operator prints the prompt and mode to the console when the button is clicked.


from .planner_agent import run_llm
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
        user_prompt = context.scene.gen_prompt
        mode = context.scene.gen_mode

        if user_prompt == "Write your prompt here...":
            self.report({'WARNING'}, "Please enter a real prompt")
            return {'CANCELLED'}
        object_name = "racoon"
        object_json = (
            "name:metarig,position:(0.00,0.00,0.00),rotation:(-0.7,0.0,0.0,0.7),"
            "children:[name:spine,position:(0.00,0.00,0.00),rotation:(0.7,0.0,0.0,0.7),"
            "children:[name:pelvis.L,position:(0.00,0.00,0.00),rotation:(-0.2,0.6,0.7,0.4),"
            "name:pelvis.R,position:(0.00,0.00,0.00),rotation:(0.2,0.6,0.7,-0.4),"
            "name:spine.001,position:(0.00,0.00,0.00),rotation:(0.0,0.0,0.0,1.0),"
            "children:[name:spine.002,position:(0.00,0.00,0.00),rotation:(0.0,0.0,0.0,1.0),"
            "children:[name:spine.003,position:(0.00,0.00,0.00),rotation:(-0.1,0.0,0.0,1.0),"
            "children:[name:breast.L,position:(0.00,0.00,0.00),rotation:(0.0,0.8,0.6,0.0),"
            "name:breast.R,position:(0.00,0.00,0.00),rotation:(0.0,0.8,0.6,0.0),"
            "name:shoulder.L,position:(0.00,0.00,0.00),rotation:(-0.7,0.2,0.4,0.6),"
            "children:[name:upper_arm.L,position:(0.00,0.00,0.00),rotation:(0.2,-0.7,0.4,0.5),"
            "children:[name:forearm.L,position:(0.00,0.00,0.00),rotation:(0.5,0.0,0.0,0.8),"
            "children:[name:hand.L,position:(0.00,0.00,0.00),rotation:(0.1,0.0,-0.1,1.0)]]],"
            "name:shoulder.R,position:(0.00,0.00,0.00),rotation:(-0.7,-0.2,-0.4,0.6),"
            "children:[name:upper_arm.R,position:(0.00,0.00,0.00),rotation:(-0.1,0.9,-0.3,0.4),"
            "children:[name:forearm.R,position:(0.00,0.00,0.00),rotation:(-0.1,0.2,-0.6,0.8),"
            "children:[name:hand.R,position:(0.00,0.00,0.00),rotation:(0.1,0.1,-0.2,1.0),"
            "children:[name:hand.R.001,position:(0.00,0.00,0.00),rotation:(0.3,0.0,0.7,0.6),"
            "children:[name:Spork_low,position:(0.00,0.00,0.00),rotation:(0.0,0.7,-0.7,0.1)"
            "]]]]],"
            "name:spine.006,position:(0.00,0.00,0.00),rotation:(-0.1,0.0,0.0,1.0),"
            "children:[name:ear.L,position:(0.00,0.01,0.00),rotation:(-0.1,-0.1,0.2,1.0),"
            "name:ear.R,position:(0.00,0.01,0.00),rotation:(-0.1,0.1,-0.5,0.9)"
            "]]]]]],"
            "name:tail,position:(0.00,0.00,0.00),rotation:(0.8,0.4,-0.1,-0.3),"
            "children:[name:tail.001,position:(0.00,0.00,0.00),rotation:(-0.2,0.1,-0.2,1.0),"
            "children:[name:tail.002,position:(0.00,0.00,0.00),rotation:(0.0,0.5,-0.5,0.7),"
            "children:[name:tail.003,position:(0.00,0.00,0.00),rotation:(-0.5,0.0,0.0,0.8)"
            "]]],"
            "name:thigh.L,position:(0.00,0.00,0.00),rotation:(1.0,0.1,-0.3,0.0),"
            "children:[name:shin.L,position:(0.00,0.00,0.00),rotation:(0.2,0.3,-0.1,0.9),"
            "children:[name:foot.L,position:(0.00,0.00,0.00),rotation:(-0.5,0.1,0.2,0.8),"
            "children:[name:heel.02.L,position:(0.00,0.00,0.00),rotation:(-0.6,0.6,-0.2,-0.5),"
            "name:toe.L,position:(0.00,0.00,0.00),rotation:(-0.3,0.8,-0.4,-0.3)"
            "]]],"
            "name:thigh.R,position:(0.00,0.00,0.00),rotation:(1.0,-0.1,0.3,0.0),"
            "children:[name:shin.R,position:(0.00,0.00,0.00),rotation:(0.2,-0.3,0.1,0.9),"
            "children:[name:foot.R,position:(0.00,0.00,0.00),rotation:(-0.5,-0.1,-0.2,0.8),"
            "children:[name:heel.02.R,position:(0.00,0.00,0.00),rotation:(0.6,0.6,-0.2,0.5),"
            "name:toe.R,position:(0.00,0.00,0.00),rotation:(0.3,0.8,-0.4,0.3)"
            "]]]]]]"
        ),
        run_result = run_llm(object_name, object_json, user_prompt)
        context.scene.gen_output = run_result
        print("LLM Response:", run_result)
        print("Prompt:", user_prompt)
        print("Mode:", mode)

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
