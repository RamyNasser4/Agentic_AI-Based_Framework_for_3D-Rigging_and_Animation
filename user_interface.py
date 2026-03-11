# This Blender add-on creates a sidebar panel with a text input for prompts,
# a dropdown to select modes (Rig, Animation, Modeling),
# and a button to generate content based on the prompt and selected mode.
# The operator prints the prompt and mode to the console when the button is clicked.

bl_info = {
    "name": "Generator Panel",
    "author": "Rana",
    "version": (1, 0),
    "blender": (4, 0, 0),
    "category": "3D View",
}

import bpy  # type: ignore

# ---------------------------------------------------
# Properties (Prompt + Dropdown)
# ---------------------------------------------------

def dropdown_items(self, context):
    return [
        ('RIG', "Rig", "Rigging Mode"),
        ('ANIM', "Animation", "Animation Mode"),
        ('MODEL', "Modeling", "Modeling Mode"),
    ]

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

# ---------------------------------------------------
# Operator (Generate Button)
# ---------------------------------------------------

class GENERATOR_OT_generate(bpy.types.Operator):
    bl_label = "Generate"
    bl_idname = "generator.generate"

    def execute(self, context):
        prompt = context.scene.gen_prompt
        mode = context.scene.gen_mode

        if prompt == "Write your prompt here...":
            self.report({'WARNING'}, "Please enter a real prompt")
            return {'CANCELLED'}

        print("Prompt:", prompt)
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

        box = layout.box()
        box.label(text="Prompt", icon="TEXT")
        box.prop(scene, "gen_prompt", text="")
        box.prop(scene, "gen_mode")
        box.operator("generator.generate", icon="PLAY")

# ---------------------------------------------------
# Register
# ---------------------------------------------------

classes = [
    GENERATOR_OT_generate,
    GENERATOR_PT_panel,
]

def register():
    for cls in classes:
        bpy.utils.register_class(cls)

def unregister():
    for cls in reversed(classes):
        bpy.utils.unregister_class(cls)

if __name__ == "__main__":
    register()
