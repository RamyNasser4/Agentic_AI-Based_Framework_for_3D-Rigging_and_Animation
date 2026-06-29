from os import getenv
import re

from dotenv import load_dotenv
from langchain.chat_models import init_chat_model
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate, FewShotChatMessagePromptTemplate
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_openai import ChatOpenAI

try:
    from .Blender_Executer import prepare_animation_text_for_execution
except ImportError:
    from Blender_Executer import prepare_animation_text_for_execution

load_dotenv()

SYSTEM_MESSAGE = (
    "You're an animator who will be provided the joints on a rigged 3D model, and you have to rotate them to produce the requested animation. "
    "The joints will be given as a JSON string that outlines the object hierarchy. "
    "You need to output one line of string each time. I will give you the starting point of each joint, which specifies the joint name and its initial position/rotation. "
    "You need to generate a line to fill out a time series for that joint then a new line for the next joint. "
    "Generate keyframes for EXACTLY ONE plan step. "
    "DO NOT continue beyond this step. "
    "Use plan history ONLY for motion consistency. "
    "DO NOT regenerate previous steps. "
    "- If a line contains \"[]\", it specifies the root motion for the animation. "
    "Each vector in the format of [t,x,y,z] specifies a key frame. "
    "\"t\" is the time stamp for the key, and \"x\", \"y\", \"z\" give the x,y,z components for the position of the object root. "
    "- If a line contains vectors enclosed in \"()\", it represents the time series for the quaternions. "
    "Each vector in the format (t,x,y,z,w) specifies a key frame for the animation. "
    "\"t\" is the time stamp for the key, and \"x\", \"y\", \"z\", \"w\" give the x,y,z,w components for the rotation quaternion, respectively. "
    "Each vector should contain 4 values if it's enclosed in [] and 5 values if it's enclosed in (). "
    "You will be given the start time of the animation. "
    "If no previous animation is provided, start the new animation at 0.0s. "
    "If a previous animation is provided, start the new animation at the provided start time, which is the end time of the previous animation, and continue smoothly from the last keyframes of that previous animation. "
    "When generating motion that includes translation, align the root and any body-facing rotations with the translation direction so the character or object faces where it is moving. "
    "If Object JSON includes \"Semantic direction inference\", treat forward_axis, up_axis, and right_axis as the trusted semantic local axes when converting plan directions into keyframes. "
    "Do not infer semantic facing from root-bone orientation. "
    "If the translation direction changes, update rotation smoothly to follow that new direction and avoid rotations that contradict the path of travel. "
    "When an animated element has both translation keyframes in [] and rotation keyframes in (), the rotation keyframe timestamps must match the translation keyframe timestamps exactly so the motion stays synchronized across position and rotation. "
    "STRICT FORMATTING RULES (MUST FOLLOW): "
    "Any vector in [] MUST have exactly 4 values: [t,x,y,z]. "
    "Any vector in () MUST have exactly 5 values: (t,x,y,z,w). "
    "NEVER output [x,y,z] or (x,y,z,w) without time. "
    "Every keyframe MUST include a timestamp. "
    "If formatting is violated, the output is INVALID. "
    "CRITICAL RULES (HIGHEST PRIORITY): "
    "A single line MUST contain ONLY ONE of the following: translation keyframes using [] OR rotation keyframes using (). "
    "NEVER mix [] and () in the same line. "
    "LINE STRUCTURE RULES: "
    "Output EXACTLY one joint per line. "
    "Each joint MUST start on a new line. "
    "NEVER place multiple joints on the same line. "
    "OUTPUT RULES: "
    "Do NOT output explanations, only animation lines. "
    "Keep the animation data formatting unchanged. "
    "INVALID:\n"
    "SMPLX-lh-male,[0.0,0.0,0.0]\n"
    "VALID:\n"
    "SMPLX-lh-male,[0.0,0.0,0.0,0.0]\n"
    "INVALID:\n"
    "SMPLX-lh-male,(0.0,0.0,0.0,1.0)\n"
    "VALID:\n"
    "SMPLX-lh-male,(0.0,0.0,0.0,0.0,1.0)\n"
    "INVALID:\n"
    "Joint,[...],(...)\n"
    "Joint1,... Joint2,...\n"
    "VALID:\n"
    "Joint,[...]\n"
    "Joint,(...)\n"
    "# Example: The object you will animate is a **whale**. \n" \
    "Object JSON: name:Armature,position:(0.0000,0.0000,0.0000),rotation:(-0.7,0.0,0.0,0.7),children:[name:Root,position:(0.0000,0.0168,0.0141),rotation:(0.7,0.0,0.0,0.7),children:[name:Head,position:(0.0000,0.0062,0.0198),rotation:(0.7,0.0,0.0,0.7),children:[name:Head_end,position:(0.0000,0.0107,0.0000),rotation:(0.0,0.0,0.0,1.0)]," \
    "name:Spine1,position:(0.0000,0.0050,0.0154),rotation:(-0.7,0.0,0.0,0.7),children:[name:Spine2,position:(0.0000,0.0156,0.0000),rotation:(0.0,0.0,0.0,1.0),children:[name:Spine3,position:(0.0000,0.0166,0.0000),rotation:(0.0,0.0,0.0,1.0),children:[name:Spine4,position:(0.0000,0.0172,0.0000),rotation:(-0.1,0.0,0.0,1.0)," \
    "children:[name:Tail,position:(0.0000,0.0196,0.0000),rotation:(0.0,0.0,0.0,1.0),children:[name:Tail_end,position:(0.0000,0.0133,0.0000),rotation:(0.0,0.0,0.0,1.0)]]]],name:TopFlipper.L,position:(-0.0107,0.0087,-0.0087),rotation:(-0.4,0.0,0.3,0.9),children:[name:MidFlipper.L,position:(0.0000,0.0067,0.0000),rotation:(0.0,0.1,0.0,1.0)," \
    "children:[name:BottomFlipper.L,position:(0.0000,0.0043,0.0000),rotation:(0.0,0.0,-0.1,1.0),children:[name:BottomFlipper.L_end,position:(0.0000,0.0076,0.0000),rotation:(0.0,0.0,0.0,1.0)]]],name:TopFlipper.R,position:(0.0092,0.0078,-0.0084),rotation:(-0.4,0.0,-0.3,0.9),children:[name:MidFlipper.R,position:(0.0000,0.0082,0.0000),rotation:(0.1,-0.1,0.1,1.0)," \
    "children:[name:BottomFlipper.R,position:(0.0000,0.0053,0.0000),rotation:(0.0,0.0,0.2,1.0),children:[name:BottomFlipper.R_end,position:(0.0000,0.0072,0.0000),rotation:(0.0,0.0,0.0,1.0)]]]]]].\n" \
    "Semantic direction inference: forward_axis=+Z, up_axis=+Y, right_axis=+X, is_humanoid=false, confidence=0.96, needs_user_confirmation=false." \
    "Instruction: create the swim animation for the whale: " \
    "Start time: 0.0s." \
    "Armature,[0.00,0.00,0.00,0.00],[2.79,0.00,0.00,0.00]\n" \
    "Armature,(0.0,-0.7,0.0,0.0,0.7),(2.8,-0.7,0.0,0.0,0.7) \n" \
    "Armature/Root,(0.0,0.7,0.0,0.0,0.7),(2.8,0.7,0.0,0.0,0.7) \n" \
    "Armature/Root/Spine1,(0.0,-0.7,0.0,0.0,0.7),(2.8,-0.7,0.0,0.0,0.7) \n" \
    "Armature/Root/Spine1/Spine2,(0.0,0.0,0.0,0.0,1.0),(0.6,0.0,0.0,0.0,1.0),(2.6,0.0,0.0,0.0,1.0),(2.8,0.0,0.0,0.0,1.0) \n" \
    "Armature/Root/Spine1/Spine2/Spine3,(0.0,0.0,0.0,0.0,1.0),(0.9,0.0,0.0,0.0,1.0),(1.5,0.1,0.0,0.0,1.0),(2.5,0.0,0.0,0.0,1.0),(2.8,0.0,0.0,0.0,1.0) \n" \
    "Armature/Root/Spine1/Spine2/Spine3/Spine4,(0.0,-0.1,0.0,0.0,1.0),(0.8,0.0,0.0,0.0,1.0),(1.5,0.1,0.0,0.0,1.0),(2.3,0.0,0.0,0.0,1.0),(2.8,-0.1,0.0,0.0,1.0) \n" \
    "Armature/Root/Spine1/Spine2/Spine3/Spine4/Tail,(0.0,0.0,0.0,0.0,1.0),(0.3,0.0,0.0,0.0,1.0),(0.7,-0.1,0.0,0.0,1.0),(1.1,-0.3,0.0,0.0,1.0),(1.7,-0.1,0.0,0.0,1.0),(2.3,0.1,0.0,0.0,1.0),(2.8,0.0,0.0,0.0,1.0) \n" \
    "Armature/Root/Spine1/TopFlipper.L,(0.0,-0.4,0.0,0.3,0.9),(1.1,-0.4,-0.1,0.2,0.9),(2.3,-0.4,0.0,0.3,0.9),(2.8,-0.4,0.0,0.3,0.9) \n" \
    "Armature/Root/Spine1/TopFlipper.L/MidFlipper.L,(0.0,0.0,0.1,0.0,1.0),(2.8,0.0,0.1,0.0,1.0) \n" \
    "Armature/Root/Spine1/TopFlipper.L/MidFlipper.L/BottomFlipper.L,(0.0,0.0,0.0,-0.1,1.0),(0.8,0.0,-0.1,-0.2,1.0),(1.9,0.0,-0.1,-0.2,1.0),(2.8,0.0,0.0,-0.1,1.0) \n" \
    "Armature/Root/Spine1/TopFlipper.R,(0.0,-0.4,0.0,-0.3,0.9),(1.0,-0.4,0.1,-0.3,0.9),(2.1,-0.4,0.0,-0.3,0.9),(2.8,-0.4,0.0,-0.3,0.9) \n" \
    "Armature/Root/Spine1/TopFlipper.R/MidFlipper.R,(0.0,0.1,-0.1,0.1,1.0),(2.8,0.1,-0.1,0.1,1.0) \n" \
    "Armature/Root/Spine1/TopFlipper.R/MidFlipper.R/BottomFlipper.R,(0.0,0.0,0.0,0.2,1.0),(1.1,0.0,0.1,0.3,1.0),(2.3,0.0,0.0,0.2,1.0),(2.8,0.0,0.0,0.2,1.0) \n" \
    "Armature/Root/Head,(0.0,0.7,0.0,0.0,0.7),(1.3,0.6,0.0,0.0,0.8),(2.6,0.7,0.0,0.0,0.7),(2.8,0.7,0.0,0.0,0.7)\n" \
)

REPAIR_SYSTEM_MESSAGE = (
    "You repair structurally invalid animation keyframe text. "
    "Return ONLY corrected animation lines and no explanations. "
    "Preserve the original target paths, channels, timestamps, and numeric values whenever possible. "
    "Each translation keyframe must be [t,x,y,z]. "
    "Each quaternion keyframe must be (t,x,y,z,w). "
    "Every keyframe must include a timestamp. "
    "Do not mix [] translation payloads and () quaternion payloads on the same line. "
    "Remove prose, markdown fences, comments, and malformed payloads that cannot be corrected without inventing motion."
)

animation_examples = [
    {
        "object": "racoon",
        "object_json": (
                    "name:metarig,position:(0.00,0.00,0.00),rotation:(-0.7,0.0,0.0,0.7),"
                    "children:[name:spine,position:(0.00,0.00,0.00),rotation"
                    ":(0.7,0.0,0.0,0.7),children:[name:pelvis.L,position"
                    ":(0.00,0.00,0.00),rotation:(-0.2,0.6,0.7,0.4),name:pelvis.R,"
                    "position:(0.00,0.00,0.00),rotation:(0.2,0.6,0.7,-0.4),name:spine"
                    ".001,position:(0.00,0.00,0.00),rotation:(0.0,0.0,0.0,1.0),children"
                    ":[name:spine.002,position:(0.00,0.00,0.00),rotation"
                    ":(0.0,0.0,0.0,1.0),children:[name:spine.003,position"
                    ":(0.00,0.00,0.00),rotation:(-0.1,0.0,0.0,1.0),children:[name:"
                    "breast.L,position:(0.00,0.00,0.00),rotation:(0.0,0.8,0.6,0.0),name"
                    ":breast.R,position:(0.00,0.00,0.00),rotation:(0.0,0.8,0.6,0.0),"
                    "name:shoulder.L,position:(0.00,0.00,0.00),rotation"
                    ":(-0.7,0.2,0.4,0.6),children:[name:upper_arm.L,position"
                    ":(0.00,0.00,0.00),rotation:(0.2,-0.7,0.4,0.5),children:[name:"
                    "forearm.L,position:(0.00,0.00,0.00),rotation:(0.5,0.0,0.0,0.8),"
                    "children:[name:hand.L,position:(0.00,0.00,0.00),rotation"
                    ":(0.1,0.0,-0.1,1.0)]]],name:shoulder.R,position:(0.00,0.00,0.00),"
                    "rotation:(-0.7,-0.2,-0.4,0.6),children:[name:upper_arm.R,position"
                    ":(0.00,0.00,0.00),rotation:(-0.1,0.9,-0.3,0.4),children:[name:"
                    "forearm.R,position:(0.00,0.00,0.00),rotation:(-0.1,0.2,-0.6,0.8),"
                    "children:[name:hand.R,position:(0.00,0.00,0.00),rotation"
                    ":(0.1,0.1,-0.2,1.0),children:[name:hand.R.001,position"
                    ":(0.00,0.00,0.00),rotation:(0.3,0.0,0.7,0.6),children:[name:"
                    "Spork_low,position:(0.00,0.00,0.00),rotation:(0.0,0.7,-0.7,0.1)"
                    "]]]]],name:spine.006,position:(0.00,0.00,0.00),rotation"
                    ":(-0.1,0.0,0.0,1.0),children:[name:ear.L,position:(0.00,0.01,0.00)"
                    ",rotation:(-0.1,-0.1,0.2,1.0),name:ear.R,position:(0.00,0.01,0.00)"
                    ",rotation:(-0.1,0.1,-0.5,0.9)]]]],name:tail,position"
                    ":(0.00,0.00,0.00),rotation:(0.8,0.4,-0.1,-0.3),children:[name:tail"
                    ".001,position:(0.00,0.00,0.00),rotation:(-0.2,0.1,-0.2,1.0),"
                    "children:[name:tail.002,position:(0.00,0.00,0.00),rotation"
                    ":(0.0,0.5,-0.5,0.7),children:[name:tail.003,position"
                    ":(0.00,0.00,0.00),rotation:(-0.5,0.0,0.0,0.8)]]],name:thigh.L,"
                    "position:(0.00,0.00,0.00),rotation:(1.0,0.1,-0.3,0.0),children:["
                    "name:shin.L,position:(0.00,0.00,0.00),rotation:(0.2,0.3,-0.1,0.9),"
                    "children:[name:foot.L,position:(0.00,0.00,0.00),rotation"
                    ":(-0.5,0.1,0.2,0.8),children:[name:heel.02.L,position"
                    ":(0.00,0.00,0.00),rotation:(-0.6,0.6,-0.2,-0.5),name:toe.L,"
                    "position:(0.00,0.00,0.00),rotation:(-0.3,0.8,-0.4,-0.3)]]],name:"
                    "thigh.R,position:(0.00,0.00,0.00),rotation:(1.0,-0.1,0.3,0.0),"
                    "children:[name:shin.R,position:(0.00,0.00,0.00),rotation"
                    ":(0.2,-0.3,0.1,0.9),children:[name:foot.R,position"
                    ":(0.00,0.00,0.00),rotation:(-0.5,-0.1,-0.2,0.8),children:[name:"
                    "heel.02.R,position:(0.00,0.00,0.00),rotation:(0.6,0.6,-0.2,0.5),"
                    "name:toe.R,position:(0.00,0.00,0.00),rotation:(0.3,0.8,-0.4,0.3)"
                    "]]]]]"
                    "Semantic direction inference: forward_axis=+Z, up_axis=+Y, right_axis=+X, is_humanoid=false, confidence=0.95, needs_user_confirmation=false."
        ),
        "user_instruction": "idle while moving head up and down",
        "start_time": "0.0s",
        "last_step_index": 0,
        "plan_history": "None.",
        "current_plan_step": "Step 1: Rotate spine.006 +Z 10 degrees; rotate tail +Y 5 degrees;",
        "previous_animation": "",
        "animation": (
            "metarig,[0.0,0.0,0.0,0.0],[1.2,0.0,0.0,0.0]\n"
            "metarig/spine/spine.006,(0.0,-0.1,0.0,0.0,1.0),(0.6,0.0,0.0,0.0,1.0),(1.2,-0.1,0.0,0.0,1.0)\n"
            "metarig/tail,(0.0,0.8,0.4,-0.1,-0.3),(0.6,0.7,0.4,-0.1,-0.4),(1.2,0.8,0.4,-0.1,-0.3)"
        ),
    },
    {
        "object": "whale",
        "object_json": (
            "name:Armature,position:(0.0000,0.0000,0.0000),rotation:(-0.7,0.0,0.0,0.7),children:[name:Root,position:(0.0000,0.0168,0.0141),rotation:(0.7,0.0,0.0,0.7),children:[name:Head,position:(0.0000,0.0062,0.0198),rotation:(0.7,0.0,0.0,0.7),children:[name:Head_end,position:(0.0000,0.0107,0.0000),rotation:(0.0,0.0,0.0,1.0)]," \
            "name:Spine1,position:(0.0000,0.0050,0.0154),rotation:(-0.7,0.0,0.0,0.7),children:[name:Spine2,position:(0.0000,0.0156,0.0000),rotation:(0.0,0.0,0.0,1.0),children:[name:Spine3,position:(0.0000,0.0166,0.0000),rotation:(0.0,0.0,0.0,1.0),children:[name:Spine4,position:(0.0000,0.0172,0.0000),rotation:(-0.1,0.0,0.0,1.0)," \
            "children:[name:Tail,position:(0.0000,0.0196,0.0000),rotation:(0.0,0.0,0.0,1.0),children:[name:Tail_end,position:(0.0000,0.0133,0.0000),rotation:(0.0,0.0,0.0,1.0)]]]],name:TopFlipper.L,position:(-0.0107,0.0087,-0.0087),rotation:(-0.4,0.0,0.3,0.9),children:[name:MidFlipper.L,position:(0.0000,0.0067,0.0000),rotation:(0.0,0.1,0.0,1.0)," \
            "children:[name:BottomFlipper.L,position:(0.0000,0.0043,0.0000),rotation:(0.0,0.0,-0.1,1.0),children:[name:BottomFlipper.L_end,position:(0.0000,0.0076,0.0000),rotation:(0.0,0.0,0.0,1.0)]]],name:TopFlipper.R,position:(0.0092,0.0078,-0.0084),rotation:(-0.4,0.0,-0.3,0.9),children:[name:MidFlipper.R,position:(0.0000,0.0082,0.0000),rotation:(0.1,-0.1,0.1,1.0)," \
            "children:[name:BottomFlipper.R,position:(0.0000,0.0053,0.0000),rotation:(0.0,0.0,0.2,1.0),children:[name:BottomFlipper.R_end,position:(0.0000,0.0072,0.0000),rotation:(0.0,0.0,0.0,1.0)]]]]]]." \
            "Semantic direction inference: forward_axis=+Z, up_axis=+Y, right_axis=+X, is_humanoid=false, confidence=0.96, needs_user_confirmation=false." \
        ),
        "user_instruction": "create a swim animation for the whale",
        "start_time": "1.2s",
        "last_step_index": 1,
        "plan_history": "Step 1: Move Armature +Z 1 units; rotate Spine1 -Y 10 degrees; rotate Tail -Y 10 degrees;",
        "current_plan_step": "Step 2: Move Armature +Z 1 units; rotate Spine1 +Y 20 degrees; rotate Head +Y 5 degrees; rotate Tail +Y 15 degrees;",
        "previous_animation": (
            "Armature,[0.0,0.0,0.0,0.0],[1.2,0.0,0.0,1.0]\n"
            "Armature/Root/Spine1,(0.0,-0.7,0.0,0.0,0.7),(1.2,-0.6,0.0,-0.1,0.8)\n"
            "Armature/Root/Spine1/Tail,(0.0,0.0,0.0,0.0,1.0),(1.2,-0.1,0.0,-0.1,1.0)"
        ),
        "animation": (
            "Armature,[1.2,0.0,0.0,1.0],[2.4,0.0,0.0,2.0]\n"
            "Armature/Root/Head,(1.2,0.7,0.0,0.0,0.7),(1.8,0.6,0.0,0.1,0.8),(2.4,0.7,0.0,0.0,0.7)\n"
            "Armature/Root/Spine1,(1.2,-0.6,0.0,-0.1,0.8),(1.8,-0.7,0.0,0.0,0.7),(2.4,-0.6,0.0,0.1,0.8)\n"
            "Armature/Root/Spine1/Tail,(1.2,-0.1,0.0,-0.1,1.0),(1.8,0.0,0.0,0.0,1.0),(2.4,0.1,0.0,0.1,1.0)"
        ),
    },
    {
        "object": "SMPLX-lh-male",
        "object_json": (
            "name:SMPLX-lh-male,position:(15.8,0.0,0.0),rotation:(0.0,0.0,0.0,1.0),children:["
            "name:Pelvis,position:(0.0,-0.8,0.4),rotation:(0.0,0.0,0.0,1.0),children:["
            "name:Left_hip,position:(0.0,0.0,0.0),rotation:(0.0,0.0,0.0,1.0),children:["
            "name:Left_knee,position:(0.0,0.0,0.0),rotation:(0.0,0.0,0.0,1.0),children:["
            "name:Left_ankle,position:(0.0,0.0,0.0),rotation:(-0.1,0.0,0.0,1.0),children:["
            "name:Left_foot,position:(0.0,0.0,0.0),rotation:(0.0,0.0,0.0,1.0)]]],"
            "name:Right_hip,position:(0.0,0.0,0.0),rotation:(0.0,0.0,0.0,1.0),children:["
            "name:Right_knee,position:(0.0,0.0,0.0),rotation:(0.0,0.0,0.0,1.0),children:["
            "name:Right_ankle,position:(0.0,0.0,0.0),rotation:(-0.1,0.0,0.0,1.0),children:["
            "name:Right_foot,position:(0.0,0.0,0.0),rotation:(0.0,0.0,0.0,1.0)]]],"
            "name:Spine1,position:(0.0,0.0,0.0),rotation:(0.0,0.0,0.0,1.0),children:["
            "name:Spine2,position:(0.0,0.0,0.0),rotation:(0.0,0.0,0.0,1.0),children:["
            "name:Spine3,position:(0.0,0.0,0.0),rotation:(0.0,0.0,0.0,1.0),children:["
            "name:Neck,position:(0.0,0.0,0.0),rotation:(0.0,0.0,0.0,1.0),children:["
            "name:Head,position:(0.0,0.0,0.0),rotation:(0.2,0.0,0.0,1.0)]],"
            "name:Left_collar,position:(0.0,0.0,0.0),rotation:(0.0,0.0,-0.4,0.9),children:["
            "name:Left_shoulder,position:(0.0,0.0,0.0),rotation:(0.0,0.1,-0.3,1.0),children:["
            "name:Left_elbow,position:(0.0,0.0,0.0),rotation:(0.2,0.2,0.0,1.0)]]],"
            "name:Right_collar,position:(0.0,0.0,0.0),rotation:(0.0,0.0,0.4,0.9),children:["
            "name:Right_shoulder,position:(0.0,0.0,0.0),rotation:(0.0,-0.2,0.2,1.0),children:["
            "name:Right_elbow,position:(0.0,0.0,0.0),rotation:(0.2,-0.2,-0.1,1.0)]]]]]]]]]"
            "Semantic direction inference: forward_axis=-Z, up_axis=+Y, right_axis=+X, is_humanoid=true, confidence=0.97, needs_user_confirmation=false"
        ),
        "user_instruction": "the standing person kicks with their left foot before going back to their original stance.",
        "start_time": "0.0s",
        "last_step_index": 0,
        "plan_history": "None.",
        "current_plan_step": (
            "Step 1: Rotate Left_hip +X 18 degrees; rotate Left_knee +X 26 degrees; "
            "rotate Left_ankle -X 10 degrees; rotate Pelvis -X 4 degrees; rotate Left_shoulder -X 10 degrees; "
            "rotate Right_shoulder +X 10 degrees;"
        ),
        "previous_animation": "",
        "animation":("""SMPLX-lh-male,[0.04,15.85,0.00,0.00],[0.52,15.85,0.00,0.00],[1.00,15.85,0.00,0.00],[1.48,15.85,0.00,0.00],[1.96,15.85,0.00,0.00],[2.44,15.85,0.00,0.00],[2.92,15.85,0.00,0.00],[3.40,15.85,0.00,0.00],[3.88,15.85,0.00,0.00],[4.36,15.85,0.00,0.00],[4.84,15.85,0.00,0.00],[5.32,15.85,0.00,0.00],[5.80,15.85,0.00,0.00],[6.28,15.85,0.00,0.00],[6.76,15.85,0.00,0.00],[7.24,15.85,0.00,0.00],[7.72,15.85,0.00,0.00],[8.20,15.85,0.00,0.00],[8.68,15.85,0.00,0.00],[9.16,15.85,0.00,0.00],[9.64,15.85,0.00,0.00]
                        SMPLX-lh-male/Pelvis,(0.04,-0.54,-0.04,-0.00,0.84),(0.52,-0.53,-0.07,0.02,0.85),(1.00,-0.57,0.06,-0.18,0.80),(1.48,-0.69,0.10,-0.54,0.47),(1.96,-0.58,0.15,-0.47,0.65),(2.44,-0.49,0.16,-0.31,0.80),(2.92,-0.53,0.07,-0.16,0.83),(3.40,-0.56,0.02,-0.07,0.83),(3.88,-0.56,-0.06,0.03,0.83),(4.36,-0.55,-0.06,0.05,0.83),(4.84,-0.56,-0.05,0.03,0.83),(5.32,-0.56,-0.04,0.02,0.83),(5.80,-0.55,-0.04,0.03,0.83),(6.28,-0.55,-0.04,0.03,0.83),(6.76,-0.55,-0.04,0.03,0.83),(7.24,-0.55,-0.04,0.02,0.83),(7.72,-0.55,-0.04,0.02,0.83),(8.20,-0.55,-0.04,0.02,0.83),(8.68,-0.55,-0.04,0.02,0.83),(9.16,-0.55,-0.04,0.02,0.83),(9.64,-0.55,-0.04,0.02,0.83)
                        SMPLX-lh-male/Pelvis/Left_hip,(0.04,-0.04,0.71,-0.67,0.21),(0.52,-0.09,0.70,-0.67,0.23),(1.00,0.03,-0.44,0.90,0.04),(1.48,-0.25,0.23,0.93,0.16),(1.96,-0.22,-0.18,0.94,0.17),(2.44,-0.18,-0.68,0.71,0.06),(2.92,0.10,0.68,-0.72,0.07),(3.40,0.04,0.60,-0.79,0.16),(3.88,-0.05,0.63,-0.73,0.25),(4.36,-0.07,0.68,-0.68,0.26),(4.84,-0.06,0.69,-0.68,0.23),(5.32,-0.06,0.70,-0.68,0.23),(5.80,-0.06,0.70,-0.68,0.23),(6.28,-0.06,0.70,-0.67,0.23),(6.76,-0.06,0.70,-0.67,0.23),(7.24,-0.06,0.70,-0.67,0.23),(7.72,-0.06,0.70,-0.67,0.23),(8.20,-0.06,0.70,-0.67,0.23),(8.68,-0.06,0.70,-0.67,0.23),(9.16,-0.06,0.70,-0.67,0.23),(9.64,-0.06,0.70,-0.67,0.23)
                        SMPLX-lh-male/Pelvis/Left_hip/Left_knee,(0.04,-0.65,-0.19,0.18,0.72),(0.52,-0.59,-0.21,0.22,0.75),(1.00,-0.22,-0.15,-0.07,0.96),(1.48,-0.98,0.12,-0.14,0.04),(1.96,-0.66,-0.15,-0.14,0.72),(2.44,-0.64,-0.01,-0.04,0.77),(2.92,-0.68,-0.02,0.01,0.73),(3.40,-0.71,-0.13,0.04,0.70),(3.88,-0.59,-0.26,0.15,0.75),(4.36,-0.64,-0.23,0.20,0.71),(4.84,-0.64,-0.21,0.21,0.71),(5.32,-0.63,-0.20,0.21,0.72),(5.80,-0.63,-0.20,0.21,0.72),(6.28,-0.63,-0.21,0.21,0.72),(6.76,-0.63,-0.21,0.21,0.72),(7.24,-0.63,-0.21,0.21,0.72),(7.72,-0.63,-0.21,0.21,0.72),(8.20,-0.63,-0.21,0.21,0.72),(8.68,-0.63,-0.21,0.21,0.72),(9.16,-0.63,-0.21,0.21,0.72),(9.64,-0.63,-0.21,0.21,0.72)
                        SMPLX-lh-male/Pelvis/Left_hip/Left_knee/Left_ankle,(0.04,-0.97,-0.19,0.12,0.11),(0.52,-0.96,-0.25,0.12,0.10),(1.00,-0.86,-0.05,0.04,0.51),(1.48,0.85,-0.35,0.04,0.39),(1.96,-0.97,0.04,-0.05,0.22),(2.44,-0.98,0.05,0.11,0.16),(2.92,-0.98,0.06,0.13,0.15),(3.40,-0.99,-0.01,0.10,0.13),(3.88,-0.96,-0.23,0.10,0.15),(4.36,-0.96,-0.24,0.13,0.09),(4.84,-0.96,-0.24,0.14,0.10),(5.32,-0.96,-0.24,0.15,0.10),(5.80,-0.96,-0.24,0.15,0.10),(6.28,-0.96,-0.24,0.14,0.10),(6.76,-0.96,-0.24,0.14,0.10),(7.24,-0.96,-0.24,0.14,0.10),(7.72,-0.96,-0.24,0.14,0.10),(8.20,-0.96,-0.24,0.14,0.10),(8.68,-0.96,-0.24,0.14,0.10),(9.16,-0.96,-0.24,0.14,0.10),(9.64,-0.96,-0.24,0.14,0.10)
                        SMPLX-lh-male/Pelvis/Left_hip/Left_knee/Left_ankle/Left_foot,(0.04,0.65,0.03,0.07,0.76),(0.52,0.64,0.06,0.12,0.75),(1.00,0.89,-0.09,-0.07,0.44),(1.48,0.08,-0.38,-0.19,0.90),(1.96,0.70,-0.24,-0.04,0.67),(2.44,0.65,-0.12,-0.12,0.74),(2.92,0.64,-0.10,-0.15,0.75),(3.40,0.64,-0.09,-0.07,0.76),(3.88,0.68,0.03,0.11,0.72),(4.36,0.64,0.06,0.11,0.76),(4.84,0.64,0.07,0.10,0.76),(5.32,0.64,0.07,0.10,0.76),(5.80,0.64,0.08,0.10,0.76),(6.28,0.64,0.07,0.10,0.76),(6.76,0.64,0.07,0.10,0.76),(7.24,0.64,0.07,0.10,0.76),(7.72,0.64,0.07,0.10,0.76),(8.20,0.64,0.07,0.10,0.76),(8.68,0.64,0.07,0.10,0.76),(9.16,0.64,0.07,0.10,0.76),(9.64,0.64,0.07,0.10,0.76)
                        SMPLX-lh-male/Pelvis/Right_hip,(0.04,-0.13,-0.69,0.67,0.24),(0.52,-0.13,-0.57,0.79,0.19),(1.00,-0.32,-0.48,0.75,0.33),(1.48,-0.55,-0.30,0.49,0.60),(1.96,-0.49,-0.44,0.54,0.52),(2.44,-0.39,-0.56,0.60,0.41),(2.92,-0.19,-0.62,0.67,0.35),(3.40,-0.17,-0.69,0.64,0.31),(3.88,-0.11,-0.69,0.68,0.21),(4.36,-0.09,-0.68,0.70,0.20),(4.84,-0.10,-0.67,0.70,0.23),(5.32,-0.09,-0.67,0.70,0.24),(5.80,-0.09,-0.67,0.69,0.24),(6.28,-0.09,-0.68,0.69,0.24),(6.76,-0.09,-0.68,0.69,0.24),(7.24,-0.09,-0.68,0.69,0.24),(7.72,-0.09,-0.68,0.69,0.24),(8.20,-0.09,-0.68,0.69,0.24),(8.68,-0.09,-0.68,0.69,0.24),(9.16,-0.09,-0.68,0.69,0.24),(9.64,-0.09,-0.68,0.69,0.24)
                        SMPLX-lh-male/Pelvis/Right_hip/Right_knee,(0.04,-0.59,0.27,-0.26,0.72),(0.52,-0.54,0.30,-0.24,0.75),(1.00,-0.50,0.35,-0.49,0.62),(1.48,-0.32,0.51,-0.59,0.53),(1.96,-0.39,0.47,-0.57,0.55),(2.44,-0.48,0.42,-0.54,0.55),(2.92,-0.48,0.38,-0.22,0.76),(3.40,-0.57,0.28,-0.27,0.73),(3.88,-0.60,0.22,-0.25,0.73),(4.36,-0.62,0.22,-0.23,0.72),(4.84,-0.63,0.25,-0.23,0.70),(5.32,-0.62,0.26,-0.22,0.70),(5.80,-0.62,0.26,-0.22,0.71),(6.28,-0.62,0.26,-0.22,0.71),(6.76,-0.62,0.26,-0.22,0.71),(7.24,-0.62,0.26,-0.22,0.71),(7.72,-0.62,0.26,-0.22,0.71),(8.20,-0.62,0.26,-0.22,0.71),(8.68,-0.62,0.26,-0.22,0.71),(9.16,-0.62,0.26,-0.22,0.71),(9.64,-0.62,0.26,-0.22,0.71)
"""
            "SMPLX-lh-male/Pelvis/Right_hip/Right_knee/Right_ankle,(0.04,-0.96,0.23,-0.14,0.07),(0.52,-0.95,0.28,-0.10,0.10),(1.00,-0.83,0.51,-0.20,0.01),(1.48,0.79,-0.59,0.18,0.02),(1.96,0.78,-0.60,0.18,0.01),(2.44,0.79,-0.57,0.23,0.01),(2.92,-0.93,0.30,-0.11,0.16),(3.40,-0.96,0.19,-0.18,0.07),(3.88,-0.97,0.17,-0.18,0.07),(4.36,-0.97,0.17,-0.17,0.07),(4.84,-0.97,0.18,-0.16,0.07),(5.32,-0.97,0.19,-0.15,0.07),(5.80,-0.97,0.19,-0.15,0.07),(6.28,-0.97,0.19,-0.15,0.07),(6.76,-0.97,0.19,-0.15,0.07),(7.24,-0.97,0.19,-0.15,0.07),(7.72,-0.97,0.19,-0.15,0.07),(8.20,-0.97,0.19,-0.15,0.07),(8.68,-0.97,0.19,-0.15,0.07),(9.16,-0.97,0.19,-0.15,0.07),(9.64,-0.97,0.19,-0.15,0.07)\n"
            "SMPLX-lh-male/Pelvis/Right_hip/Right_knee/Right_ankle/Right_foot,(0.04,0.64,-0.09,-0.10,0.76),(0.52,0.67,-0.08,-0.15,0.73),(1.00,0.56,-0.31,-0.30,0.70),(1.48,0.53,-0.35,-0.38,0.68),(1.96,0.53,-0.36,-0.38,0.67),(2.44,0.53,-0.38,-0.34,0.68),(2.92,0.71,-0.10,-0.16,0.68),(3.40,0.63,-0.09,-0.04,0.77),(3.88,0.63,-0.08,-0.03,0.77),(4.36,0.63,-0.07,-0.03,0.77),(4.84,0.63,-0.07,-0.05,0.77),(5.32,0.64,-0.07,-0.05,0.77),(5.80,0.64,-0.07,-0.05,0.77),(6.28,0.64,-0.07,-0.05,0.77),(6.76,0.64,-0.07,-0.05,0.77),(7.24,0.64,-0.07,-0.05,0.77),(7.72,0.64,-0.07,-0.05,0.77),(8.20,0.64,-0.07,-0.05,0.77),(8.68,0.64,-0.07,-0.05,0.77),(9.16,0.64,-0.07,-0.05,0.77),(9.64,0.64,-0.07,-0.05,0.77)\n"
            """                        SMPLX-lh-male/Pelvis/Spine1,(0.04,0.72,0.02,-0.04,0.70),(0.52,0.81,0.05,-0.04,0.58),(1.00,0.81,-0.22,-0.08,0.53),(1.48,0.69,-0.45,-0.06,0.56),(1.96,0.65,-0.44,-0.22,0.58),(2.44,0.75,-0.27,-0.31,0.52),(2.92,0.75,-0.12,-0.21,0.62),(3.40,0.73,-0.04,-0.08,0.67),(3.88,0.72,0.03,0.01,0.69),(4.36,0.70,0.06,-0.00,0.71),(4.84,0.70,0.05,-0.03,0.71),(5.32,0.69,0.05,-0.03,0.72),(5.80,0.69,0.05,-0.03,0.72),(6.28,0.69,0.05,-0.03,0.72),(6.76,0.69,0.05,-0.03,0.72),(7.24,0.69,0.05,-0.04,0.72),(7.72,0.69,0.05,-0.04,0.72),(8.20,0.69,0.05,-0.04,0.72),(8.68,0.69,0.05,-0.04,0.72),(9.16,0.69,0.05,-0.04,0.72),(9.64,0.69,0.05,-0.04,0.72)
                        SMPLX-lh-male/Pelvis/Spine1/Spine2,(0.04,0.87,-0.04,-0.04,0.48),(0.52,0.93,-0.03,-0.07,0.36),(1.00,0.86,-0.34,-0.04,0.39),(1.48,0.80,-0.47,-0.06,0.36),(1.96,0.74,-0.53,-0.16,0.39),(2.44,0.78,-0.44,-0.27,0.36),(2.92,0.84,-0.26,-0.18,0.43),(3.40,0.88,-0.14,-0.06,0.46),(3.88,0.88,-0.02,0.01,0.48),(4.36,0.87,0.01,-0.01,0.49),(4.84,0.87,-0.01,-0.03,0.50),(5.32,0.86,-0.02,-0.03,0.50),(5.80,0.86,-0.01,-0.03,0.50),(6.28,0.86,-0.01,-0.03,0.51),(6.76,0.86,-0.02,-0.03,0.51),(7.24,0.86,-0.02,-0.04,0.51),(7.72,0.86,-0.02,-0.04,0.51),(8.20,0.86,-0.02,-0.04,0.51),(8.68,0.86,-0.02,-0.04,0.51),(9.16,0.86,-0.02,-0.04,0.51),(9.64,0.86,-0.02,-0.04,0.51)
                        SMPLX-lh-male/Pelvis/Spine1/Spine2/Spine3,(0.04,0.64,-0.04,-0.07,0.77),(0.52,0.73,-0.05,-0.11,0.67),(1.00,0.67,-0.38,-0.16,0.62),(1.48,0.64,-0.40,-0.22,0.62),(1.96,0.58,-0.47,-0.37,0.55),(2.44,0.58,-0.41,-0.45,0.55),(2.92,0.64,-0.25,-0.30,0.67),(3.40,0.67,-0.14,-0.12,0.72),(3.88,0.67,-0.05,-0.02,0.74),(4.36,0.66,-0.02,-0.03,0.75),(4.84,0.66,-0.03,-0.06,0.75),(5.32,0.65,-0.04,-0.06,0.75),(5.80,0.65,-0.03,-0.06,0.76),(6.28,0.65,-0.03,-0.06,0.76),(6.76,0.65,-0.03,-0.06,0.76),(7.24,0.65,-0.04,-0.07,0.76),(7.72,0.65,-0.04,-0.07,0.76),(8.20,0.65,-0.04,-0.07,0.76),(8.68,0.65,-0.04,-0.07,0.76),(9.16,0.65,-0.04,-0.07,0.76),(9.64,0.65,-0.04,-0.07,0.76)
                        SMPLX-lh-male/Pelvis/Spine1/Spine2/Spine3/Neck,(0.04,0.89,-0.07,-0.06,0.45),(0.52,0.92,-0.09,-0.09,0.37),(1.00,0.87,-0.30,0.03,0.39),(1.48,0.92,-0.22,0.03,0.32),(1.96,0.87,-0.39,-0.04,0.29),(2.44,0.83,-0.39,-0.12,0.37),(2.92,0.87,-0.27,-0.09,0.41),(3.40,0.90,-0.11,-0.01,0.41),(3.88,0.91,-0.03,0.02,0.41),(4.36,0.90,-0.00,-0.00,0.44),(4.84,0.89,-0.01,-0.00,0.46),(5.32,0.89,-0.01,0.00,0.45),(5.80,0.89,-0.01,0.00,0.46),(6.28,0.89,-0.01,0.00,0.46),(6.76,0.89,-0.01,0.00,0.46),(7.24,0.89,-0.01,0.00,0.46),(7.72,0.89,-0.01,0.00,0.46),(8.20,0.89,-0.01,0.00,0.46),(8.68,0.89,-0.01,0.00,0.46),(9.16,0.89,-0.01,0.00,0.46),(9.64,0.89,-0.01,0.00,0.46)
                        SMPLX-lh-male/Pelvis/Spine1/Spine2/Spine3/Neck/Head,(0.04,0.82,-0.08,-0.07,0.57),(0.52,0.80,-0.12,-0.06,0.58),(1.00,0.71,-0.17,-0.01,0.68),(1.48,0.79,-0.12,0.01,0.60),(1.96,0.77,-0.18,-0.02,0.62),(2.44,0.71,-0.22,-0.14,0.65),(2.92,0.74,-0.22,-0.13,0.63),(3.40,0.78,-0.10,-0.05,0.62),(3.88,0.79,-0.05,-0.04,0.61),(4.36,0.78,-0.05,-0.06,0.62),(4.84,0.78,-0.05,-0.05,0.62),(5.32,0.79,-0.04,-0.04,0.61),(5.80,0.79,-0.03,-0.04,0.61),(6.28,0.79,-0.03,-0.04,0.61),(6.76,0.79,-0.04,-0.04,0.61),(7.24,0.79,-0.04,-0.04,0.61),(7.72,0.79,-0.04,-0.04,0.61),(8.20,0.79,-0.04,-0.04,0.61),(8.68,0.79,-0.04,-0.04,0.61),(9.16,0.79,-0.04,-0.04,0.61),(9.64,0.79,-0.04,-0.04,0.61)
                        SMPLX-lh-male/Pelvis/Spine1/Spine2/Spine3/Left_collar,(0.04,0.38,0.63,-0.58,0.35),(0.52,0.47,0.59,-0.61,0.24),(1.00,0.76,0.33,-0.52,0.22),(1.48,0.77,0.33,-0.53,0.14),(1.96,0.77,0.34,-0.54,0.07),(2.44,-0.71,-0.32,0.63,0.03),(2.92,0.58,0.45,-0.67,0.12),(3.40,0.47,0.59,-0.58,0.31),(3.88,0.37,0.68,-0.50,0.39),(4.36,0.35,0.67,-0.53,0.38),(4.84,0.37,0.65,-0.56,0.36),(5.32,0.37,0.65,-0.56,0.35),(5.80,0.37,0.65,-0.56,0.36),(6.28,0.36,0.65,-0.56,0.36),(6.76,0.36,0.65,-0.56,0.35),(7.24,0.37,0.65,-0.57,0.35),(7.72,0.37,0.65,-0.57,0.35),(8.20,0.37,0.65,-0.57,0.35),(8.68,0.37,0.65,-0.57,0.35),(9.16,0.37,0.65,-0.57,0.35),(9.64,0.37,0.65,-0.57,0.35)
                        SMPLX-lh-male/Pelvis/Spine1/Spine2/Spine3/Left_collar/Left_shoulder,(0.04,-0.12,-0.74,0.66,0.06),(0.52,-0.25,-0.66,0.69,0.18),(1.00,0.67,0.59,-0.46,0.04),(1.48,0.70,0.61,-0.32,0.17),(1.96,-0.64,-0.66,0.39,0.09),(2.44,-0.55,-0.53,0.55,0.34),(2.92,-0.41,-0.53,0.66,0.34),(3.40,-0.22,-0.73,0.64,0.01),(3.88,0.08,0.80,-0.59,0.06),(4.36,0.08,0.77,-0.64,0.02),(4.84,-0.11,-0.74,0.66,0.01),(5.32,-0.11,-0.75,0.65,0.02),(5.80,-0.11,-0.74,0.66,0.02),(6.28,-0.10,-0.74,0.66,0.02),(6.76,-0.11,-0.74,0.66,0.02),(7.24,-0.11,-0.74,0.66,0.02),(7.72,-0.11,-0.74,0.66,0.02),(8.20,-0.11,-0.74,0.66,0.02),(8.68,-0.11,-0.74,0.66,0.02),(9.16,-0.11,-0.74,0.66,0.02),(9.64,-0.11,-0.74,0.66,0.02)
                        SMPLX-lh-male/Pelvis/Spine1/Spine2/Spine3/Left_collar/Left_shoulder/Left_elbow,(0.04,-0.33,-0.52,0.78,0.14),(0.52,-0.37,-0.23,0.85,0.29),(1.00,-0.73,-0.09,0.50,0.46),(1.48,-0.80,-0.48,0.12,0.35),(1.96,-0.71,-0.37,0.29,0.53),(2.44,-0.66,-0.23,0.49,0.53),(2.92,-0.45,0.01,0.71,0.55),(3.40,-0.40,-0.58,0.70,0.03),(3.88,-0.34,-0.59,0.73,0.05),(4.36,-0.34,-0.53,0.77,0.10),(4.84,-0.36,-0.51,0.77,0.12),(5.32,-0.36,-0.51,0.77,0.14),(5.80,-0.36,-0.51,0.77,0.14),(6.28,-0.36,-0.51,0.77,0.14),(6.76,-0.36,-0.51,0.77,0.14),(7.24,-0.36,-0.51,0.77,0.14),(7.72,-0.36,-0.51,0.77,0.14),(8.20,-0.36,-0.51,0.77,0.14),(8.68,-0.36,-0.51,0.77,0.14),(9.16,-0.36,-0.51,0.77,0.14),(9.64,-0.36,-0.51,0.77,0.14)
                        SMPLX-lh-male/Pelvis/Spine1/Spine2/Spine3/Left_collar/Left_shoulder/Left_elbow/Left_wrist,(0.04,-0.30,-0.58,0.71,0.25),(0.52,-0.42,-0.26,0.79,0.36),(1.00,-0.71,-0.27,0.36,0.55),(1.48,-0.57,-0.75,-0.04,0.33),(1.96,-0.61,-0.53,0.05,0.59),(2.44,-0.69,-0.32,0.28,0.59),(2.92,-0.44,-0.07,0.64,0.63),(3.40,-0.33,-0.68,0.65,0.10),(3.88,-0.24,-0.68,0.66,0.19),(4.36,-0.26,-0.62,0.70,0.22),(4.84,-0.30,-0.60,0.70,0.25),(5.32,-0.30,-0.60,0.69,0.27),(5.80,-0.30,-0.60,0.69,0.27),(6.28,-0.29,-0.60,0.70,0.27),(6.76,-0.30,-0.60,0.69,0.27),(7.24,-0.30,-0.60,0.69,0.27),(7.72,-0.30,-0.60,0.69,0.27),(8.20,-0.30,-0.60,0.69,0.27),(8.68,-0.30,-0.60,0.69,0.27),(9.16,-0.30,-0.60,0.69,0.27),(9.64,-0.30,-0.60,0.69,0.27)
                        SMPLX-lh-male/Pelvis/Spine1/Spine2/Spine3/Left_collar/Left_shoulder/Left_elbow/Left_wrist/Left_palm,(0.04,0.59,0.08,-0.73,0.32),(0.52,0.42,-0.24,-0.83,0.28),(1.00,-0.60,0.38,0.67,0.20),(1.48,-0.89,-0.10,0.31,0.32),(1.96,-0.71,0.10,0.53,0.44),(2.44,-0.62,0.32,0.66,0.29),(2.92,-0.26,0.36,0.90,0.02),(3.40,0.71,0.13,-0.59,0.36),(3.88,0.64,0.20,-0.67,0.32),(4.36,0.60,0.13,-0.72,0.33),(4.84,0.61,0.09,-0.73,0.30),(5.32,0.60,0.10,-0.74,0.28),(5.80,0.60,0.10,-0.74,0.29),(6.28,0.60,0.10,-0.74,0.29),(6.76,0.60,0.10,-0.74,0.29),(7.24,0.60,0.09,-0.74,0.29),(7.72,0.60,0.09,-0.74,0.29),(8.20,0.60,0.09,-0.74,0.29),(8.68,0.60,0.09,-0.74,0.29),(9.16,0.60,0.09,-0.74,0.29),(9.64,0.60,0.09,-0.74,0.29)
                        SMPLX-lh-male/Pelvis/Spine1/Spine2/Spine3/Right_collar,(0.04,0.25,-0.65,0.56,0.46),(0.52,0.33,-0.67,0.47,0.47),(1.00,0.09,-0.77,0.40,0.50),(1.48,0.17,-0.70,0.49,0.49),(1.96,-0.15,-0.82,-0.00,0.55),(2.44,-0.19,-0.73,0.02,0.65),(2.92,0.05,-0.71,0.27,0.65),(3.40,0.16,-0.71,0.47,0.49),(3.88,0.25,-0.69,0.53,0.42),(4.36,0.27,-0.67,0.54,0.43),(4.84,0.24,-0.67,0.53,0.45),(5.32,0.24,-0.68,0.53,0.46),(5.80,0.25,-0.67,0.53,0.45),(6.28,0.25,-0.67,0.54,0.45),(6.76,0.24,-0.67,0.53,0.46),(7.24,0.24,-0.67,0.53,0.46),(7.72,0.24,-0.67,0.53,0.46),(8.20,0.24,-0.67,0.53,0.46),(8.68,0.24,-0.67,0.53,0.46),(9.16,0.24,-0.67,0.53,0.46),(9.64,0.24,-0.67,0.53,0.46)
                        SMPLX-lh-male/Pelvis/Spine1/Spine2/Spine3/Right_collar/Right_shoulder,(0.04,0.06,-0.69,0.72,0.03),(0.52,0.07,-0.79,0.61,0.06),(1.00,-0.00,-0.70,0.62,0.35),(1.48,0.04,-0.47,0.82,0.33),(1.96,-0.46,-0.80,0.06,0.38),(2.44,-0.54,-0.67,0.40,0.32),(2.92,-0.13,-0.75,0.51,0.41),(3.40,0.00,-0.72,0.67,0.16),(3.88,0.08,-0.71,0.69,0.07),(4.36,0.08,-0.71,0.70,0.05),(4.84,0.07,-0.70,0.70,0.05),(5.32,0.06,-0.71,0.70,0.06),(5.80,0.07,-0.70,0.71,0.05),(6.28,0.07,-0.70,0.71,0.05),(6.76,0.06,-0.70,0.71,0.06),(7.24,0.06,-0.70,0.71,0.06),(7.72,0.06,-0.70,0.71,0.06),(8.20,0.06,-0.70,0.71,0.06),(8.68,0.06,-0.70,0.71,0.06),(9.16,0.06,-0.70,0.71,0.06),(9.64,0.06,-0.70,0.71,0.06)
                        SMPLX-lh-male/Pelvis/Spine1/Spine2/Spine3/Right_collar/Right_shoulder/Right_elbow,(0.04,-0.28,0.54,-0.80,0.01),(0.52,0.29,-0.49,0.82,0.03),(1.00,0.32,-0.38,0.83,0.26),(1.48,0.32,-0.14,0.87,0.34),(1.96,-0.14,-0.79,0.42,0.41),(2.44,-0.15,-0.66,0.62,0.41),(2.92,0.18,-0.63,0.60,0.45),(3.40,0.31,-0.52,0.79,0.06),(3.88,-0.34,0.54,-0.76,0.02),(4.36,-0.34,0.55,-0.76,0.01),(4.84,-0.33,0.53,-0.78,0.01),(5.32,-0.32,0.56,-0.77,0.01),(5.80,-0.34,0.53,-0.78,0.02),(6.28,-0.34,0.53,-0.78,0.02),(6.76,-0.33,0.54,-0.77,0.01),(7.24,-0.33,0.54,-0.78,0.01),(7.72,-0.33,0.54,-0.78,0.01),(8.20,-0.33,0.54,-0.78,0.01),(8.68,-0.33,0.54,-0.78,0.01),(9.16,-0.33,0.54,-0.78,0.01),(9.64,-0.33,0.54,-0.78,0.01)
                        SMPLX-lh-male/Pelvis/Spine1/Spine2/Spine3/Right_collar/Right_shoulder/Right_elbow/Right_wrist,(0.04,-0.32,0.60,-0.71,0.18),(0.52,-0.27,0.52,-0.80,0.08),(1.00,0.33,-0.34,0.84,0.24),(1.48,0.40,-0.09,0.83,0.39),(1.96,-0.16,-0.74,0.46,0.46),(2.44,-0.21,-0.65,0.66,0.33),(2.92,0.13,-0.69,0.61,0.36),(3.40,-0.29,0.59,-0.75,0.07),(3.88,-0.34,0.62,-0.69,0.15),(4.36,-0.32,0.63,-0.68,0.16),(4.84,-0.32,0.62,-0.70,0.16),(5.32,-0.30,0.64,-0.69,0.16),(5.80,-0.32,0.62,-0.70,0.16),(6.28,-0.32,0.62,-0.70,0.16),(6.76,-0.31,0.62,-0.70,0.16),(7.24,-0.31,0.62,-0.70,0.16),(7.72,-0.31,0.62,-0.70,0.16),(8.20,-0.31,0.62,-0.70,0.16),(8.68,-0.31,0.62,-0.70,0.16),(9.16,-0.31,0.62,-0.70,0.16),(9.64,-0.31,0.62,-0.70,0.16)
                        SMPLX-lh-male/Pelvis/Spine1/Spine2/Spine3/Right_collar/Right_shoulder/Right_elbow/Right_wrist/Right_palm,(0.04,0.64,-0.12,0.67,0.37),(0.52,0.56,-0.08,0.65,0.50),(1.00,0.50,0.08,0.43,0.75),(1.48,0.37,0.30,0.29,0.83),(1.96,0.47,-0.58,0.05,0.66),(2.44,0.36,-0.53,0.27,0.72),(2.92,0.63,-0.32,0.22,0.68),(3.40,0.62,-0.12,0.61,0.47),(3.88,0.67,-0.11,0.64,0.37),(4.36,0.67,-0.14,0.64,0.36),(4.84,0.65,-0.13,0.65,0.37),(5.32,0.66,-0.16,0.64,0.36),(5.80,0.65,-0.13,0.65,0.37),(6.28,0.65,-0.12,0.65,0.37),(6.76,0.65,-0.14,0.65,0.37),(7.24,0.65,-0.14,0.65,0.37),(7.72,0.65,-0.14,0.65,0.37),(8.20,0.65,-0.14,0.65,0.37),(8.68,0.65,-0.14,0.65,0.37),(9.16,0.65,-0.14,0.65,0.37),(9.64,0.65,-0.14,0.65,0.37)"""),
                            },
    {
        "object": "SMPLX-lh-male",
        "object_json": (
            "name:SMPLX-lh-male,position:(15.8,0.0,0.0),rotation:(0.0,0.0,0.0,1.0),children:["
            "name:Pelvis,position:(0.0,-0.8,0.4),rotation:(0.0,0.0,0.0,1.0),children:["
            "name:Left_hip,position:(0.0,0.0,0.0),rotation:(0.0,0.0,0.0,1.0),children:["
            "name:Left_knee,position:(0.0,0.0,0.0),rotation:(0.0,0.0,0.0,1.0),children:["
            "name:Left_ankle,position:(0.0,0.0,0.0),rotation:(-0.1,0.0,0.0,1.0)]],"
            "name:Right_hip,position:(0.0,0.0,0.0),rotation:(0.0,0.0,0.0,1.0),children:["
            "name:Right_knee,position:(0.0,0.0,0.0),rotation:(0.0,0.0,0.0,1.0),children:["
            "name:Right_ankle,position:(0.0,0.0,0.0),rotation:(-0.1,0.0,0.0,1.0)]]],"
            "name:Spine1,position:(0.0,0.0,0.0),rotation:(0.0,0.0,0.0,1.0),children:["
            "name:Spine2,position:(0.0,0.0,0.0),rotation:(0.0,0.0,0.0,1.0),children:["
            "name:Spine3,position:(0.0,0.0,0.0),rotation:(0.0,0.0,0.0,1.0),children:["
            "name:Left_collar,position:(0.0,0.0,0.0),rotation:(0.0,0.0,-0.4,0.9),children:["
            "name:Left_shoulder,position:(0.0,0.0,0.0),rotation:(0.0,0.1,-0.3,1.0)]],"
            "name:Right_collar,position:(0.0,0.0,0.0),rotation:(0.0,0.0,0.4,0.9),children:["
            "name:Right_shoulder,position:(0.0,0.0,0.0),rotation:(0.0,-0.2,0.2,1.0)]]]]]]]"
            "Semantic direction inference: forward_axis=-Z, up_axis=+Y, right_axis=+X, is_humanoid=true, confidence=0.97, needs_user_confirmation=false"
        ),
        "user_instruction": "the person is standing relaxed, walks forward then turns around on their left foot, and walks back to their original position.",
        "start_time": "1.6s",
        "last_step_index": 2,
        "plan_history": (
            "Step 1: Rotate Left_hip +X 18 degrees; rotate Left_knee +X 26 degrees; rotate Left_ankle -X 10 degrees;\n"
            "Step 2: Rotate Left_hip -X 18 degrees; rotate Left_knee -X 26 degrees; rotate Left_ankle +X 10 degrees;"
        ),
        "current_plan_step": (
            "Step 3: Move SMPLX-lh-male -Z 1 units; rotate Pelvis -Y 12 degrees; "
            "rotate Left_hip -X 12 degrees; rotate Right_hip +X 18 degrees; rotate Left_shoulder +X 12 degrees; "
            "rotate Right_shoulder -X 12 degrees;"
        ),
        "previous_animation": (
            "SMPLX-lh-male,[0.0,15.8,0.0,0.0],[0.8,15.8,0.0,0.0],[1.6,15.8,0.0,0.0]\n"
            "SMPLX-lh-male/Pelvis,(0.0,0.0,0.0,0.0,1.0),(0.8,-0.04,0.0,0.0,1.0),(1.6,0.0,0.0,0.0,1.0)\n"
            "SMPLX-lh-male/Pelvis/Left_hip,(0.0,0.0,0.0,0.0,1.0),(0.8,0.15,0.0,0.0,0.99),(1.6,0.0,0.0,0.0,1.0)"
        ),
        "animation": (
            "SMPLX-lh-male,[1.6,15.8,0.0,0.0],[2.2,15.8,0.0,-0.5],[2.8,15.8,0.0,-1.0]\n"
            "SMPLX-lh-male/Pelvis,(1.6,0.0,0.0,0.0,1.0),(2.2,0.0,-0.1,0.0,0.99),(2.8,0.0,-0.1,0.0,0.99)\n"
            "SMPLX-lh-male/Pelvis/Left_hip,(1.6,0.0,0.0,0.0,1.0),(2.2,-0.1,0.0,0.0,1.0),(2.8,-0.1,0.0,0.0,1.0)\n"
            "SMPLX-lh-male/Pelvis/Right_hip,(1.6,0.0,0.0,0.0,1.0),(2.2,0.15,0.0,0.0,0.99),(2.8,0.12,0.0,0.0,0.99)\n"
            "SMPLX-lh-male/Pelvis/Spine1/Spine2/Spine3/Left_collar/Left_shoulder,(1.6,0.0,0.1,-0.3,1.0),(2.2,0.08,0.1,-0.3,0.99),(2.8,0.08,0.1,-0.3,0.99)\n"
            "SMPLX-lh-male/Pelvis/Spine1/Spine2/Spine3/Right_collar/Right_shoulder,(1.6,0.0,-0.2,0.2,1.0),(2.2,-0.08,-0.2,0.2,0.99),(2.8,-0.08,-0.2,0.2,0.99)"
        ),
    },
]


class KeyFrameAgent:
    _TIME_TOKEN_PATTERN = re.compile(r"[\[\(]\s*([-+]?\d*\.?\d+(?:[eE][-+]?\d+)?)")
    _PLAN_STEP_PATTERN = re.compile(
        r"^\s*(?:\[Generator\]\s*)?(?:Plan\s+step\s*(?P<plan_index>\d+)\s*:\s*)?"
        r"(?:(?:Step\s*(?P<step_index>\d+)\s*:\s*))?(?P<body>.*)$",
        re.IGNORECASE | re.DOTALL,
    )

    def initialize_chain(self):
        # llm = init_chat_model(
        #     model="gpt-oss-120b:free",
        #     model_provider="openai",
        #     base_url="https://openrouter.ai/api/v1",
        #     api_key=getenv("OPENROUTER_API_KEY"),
        #     default_headers={
        #         # "HTTP-Referer": getenv("YOUR_SITE_URL"),
        #         # "X-OpenRouter-Title": getenv("YOUR_SITE_NAME"),
        #     },
        # )
        # llm = init_chat_model(
        #     model="gpt-5-mini",
        #     model_provider="openai",
        #     base_url="http://localhost:4000/v1/",
        #     api_key="nothing",
        #     temperature=0.1,
        #     max_tokens=1800,
        #     default_headers={
        #         # "HTTP-Referer": getenv("YOUR_SITE_URL"),
        #         # "X-OpenRouter-Title": getenv("YOUR_SITE_NAME"),
        #     },
        # )
        # llm = ChatOpenAI(
        #     model="gpt-5.4-mini",
        #     base_url="http://localhost:4000/v1",
        #     api_key="nothing",
        #     temperature=0.75,
        #     use_responses_api=True,
        # )
        llm = ChatGoogleGenerativeAI(
            model="gemma-4-31b-it",
            google_api_key=getenv("GOOGLE_API_KEY"),
            temperature=0
        )

        prompt_template_for_examples = ChatPromptTemplate.from_messages([
            (
                "human",
                (
                    "The object you will animate is a **{object}**.\n"
                    "Object JSON: {object_json}.\n"
                    "High-level motion goal (optional): {user_instruction}.\n"
                    "Start time for this step: {start_time}.\n"
                    "Last completed plan step index: {last_step_index}.\n"
                    "Structured plan history:\n"
                    "{plan_history}\n\n"
                    "Generate keyframes ONLY for this plan step:\n"
                    "{current_plan_step}\n\n"
                    "Previous animation to continue from if provided:\n"
                    "{previous_animation}"
                ),
            ),
            ("ai", "{animation}"),
        ])

        few_shot_prompt = FewShotChatMessagePromptTemplate(
            examples=animation_examples,
            example_prompt=prompt_template_for_examples,
        )

        main_prompt = ChatPromptTemplate.from_messages([
            ("system", SYSTEM_MESSAGE),
            few_shot_prompt,
            (
                "human",
                (
                    "The object you will animate is a **{object}**.\n"
                    "Object JSON: {object_json}.\n"
                    "High-level motion goal (optional): {user_instruction}.\n"
                    "Start time for this step: {start_time}.\n"
                    "Last completed plan step index: {last_step_index}.\n"
                    "Structured plan history:\n"
                    "{plan_history}\n\n"
                    "Generate keyframes ONLY for this plan step:\n"
                    "{current_plan_step}\n\n"
                    "Previous animation to continue from if provided:\n"
                    "{previous_animation}"
                ),
            ),
        ])

        self.chain = main_prompt | llm | StrOutputParser()
        repair_prompt = ChatPromptTemplate.from_messages([
            ("system", REPAIR_SYSTEM_MESSAGE),
            (
                "human",
                (
                    "Validation errors:\n"
                    "{validation_errors}\n\n"
                    "Animation text to repair:\n"
                    "{animation_text}"
                ),
            ),
        ])
        self.repair_chain = repair_prompt | llm | StrOutputParser()
        return self.chain

    def _normalize_previous_animation(self, previous_animation) -> str:
        if previous_animation is None:
            return ""
        if isinstance(previous_animation, str):
            return previous_animation.strip()
        return "\n".join(str(line).strip() for line in previous_animation if str(line).strip())

    def _parse_time_value(self, value) -> float:
        if isinstance(value, (int, float)):
            return float(value)

        value_text = str(value).strip()
        match = re.search(r"[-+]?\d*\.?\d+(?:[eE][-+]?\d+)?", value_text)
        if match is None:
            raise ValueError(f"Could not parse a time value from `{value}`.")
        return float(match.group(0))

    def _extract_end_time(self, animation_text: str) -> float:
        matches = [float(match.group(1)) for match in self._TIME_TOKEN_PATTERN.finditer(animation_text)]
        if not matches:
            raise ValueError("Previous animation was provided, but no keyframe time values were found.")
        return max(matches)

    def _format_start_time(self, start_time: float) -> str:
        formatted = f"{start_time:.4f}".rstrip("0").rstrip(".")
        if "." not in formatted:
            formatted += ".0"
        return f"{formatted}s"

    def _extract_step_index(self, step_text) -> int | None:
        if step_text is None:
            return None

        text = str(step_text).strip()
        if not text:
            return None

        match = re.search(r"\[Generator\]\s*Plan\s+step\s*(\d+)\s*:", text, re.IGNORECASE)
        if match is not None:
            return int(match.group(1))

        match = re.search(r"\bStep\s*(\d+)\s*:", text, re.IGNORECASE)
        if match is not None:
            return int(match.group(1))

        return None

    def _normalize_plan_step(self, step_text, default_index: int | None = None) -> str:
        text = str(step_text or "").strip()
        if not text:
            index = 1 if default_index is None else int(default_index)
            return f"Step {index}:"

        match = self._PLAN_STEP_PATTERN.match(text)
        body = text
        index = default_index

        if match is not None:
            plan_index = match.group("plan_index")
            step_index = match.group("step_index")
            body = (match.group("body") or "").strip()
            if plan_index is not None:
                index = int(plan_index)
            elif step_index is not None:
                index = int(step_index)

        body = re.sub(r"^\s*Step\s*\d+\s*:\s*", "", body, flags=re.IGNORECASE)
        body = re.sub(r"\s+", " ", body).strip()

        if index is None:
            index = 1

        if not body:
            return f"Step {index}:"
        return f"Step {int(index)}: {body}"

    def _normalize_plan_history(self, plan_history) -> str:
        if plan_history is None:
            return "None."

        if isinstance(plan_history, str):
            candidate_lines = [line.strip() for line in plan_history.splitlines() if line.strip()]
            if not candidate_lines and plan_history.strip():
                candidate_lines = [plan_history.strip()]
        else:
            candidate_lines = [str(item).strip() for item in plan_history if str(item).strip()]

        normalized_steps = []
        next_index = 1
        for line in candidate_lines:
            if line.lower() in {"none", "none."}:
                continue
            normalized_line = self._normalize_plan_step(line, default_index=next_index)
            normalized_steps.append(normalized_line)
            extracted_index = self._extract_step_index(normalized_line)
            next_index = 1 if extracted_index is None else extracted_index + 1

        if not normalized_steps:
            return "None."
        return "\n".join(normalized_steps)

    def _prepare_input(self, input_dict: dict) -> dict:
        payload = dict(input_dict)
        previous_animation = self._normalize_previous_animation(payload.get("previous_animation"))

        if previous_animation:
            start_time = self._extract_end_time(previous_animation)
        else:
            start_time = 0.0

        normalized_history = self._normalize_plan_history(payload.get("plan_history"))
        history_count = 0 if normalized_history == "None." else len(normalized_history.splitlines())

        user_instruction = str(
            payload.get("user_instruction")
            or payload.get("instruction")
            or ""
        ).strip()

        current_plan_step = payload.get("current_plan_step")
        if not str(current_plan_step or "").strip():
            if not user_instruction:
                raise ValueError("KeyFrameAgent requires `current_plan_step` or `instruction`.")
            current_plan_step = f"Step 1: {user_instruction}"

        step_index = payload.get("step_index")
        if step_index is None:
            step_index = self._extract_step_index(current_plan_step)
        if step_index is None:
            step_index = history_count + 1 if history_count else 1
        else:
            step_index = int(self._parse_time_value(step_index))

        normalized_current_step = self._normalize_plan_step(current_plan_step, default_index=step_index)
        step_index = self._extract_step_index(normalized_current_step) or step_index

        last_step_index = payload.get("last_step_index")
        if last_step_index is None:
            if history_count:
                last_step_index = history_count
            else:
                last_step_index = max(int(step_index) - 1, 0)
        else:
            last_step_index = int(self._parse_time_value(last_step_index))

        payload["current_plan_step"] = normalized_current_step
        payload["plan_history"] = normalized_history
        payload["last_step_index"] = last_step_index
        payload["user_instruction"] = user_instruction
        payload["start_time"] = self._format_start_time(start_time)
        payload["previous_animation"] = previous_animation
        return payload

    def _enforce_newlines(self, text: str) -> str:
        import re

        text = str(text or "")
        text = text.replace("\r\n", "\n").replace("\r", "\n")

        text = re.sub(r"(?<!\n)(SMPLX(?:-[^,\[\(\s]+)?|metarig|Armature)", r"\n\1", text)
        text = re.sub(r"[ \t]+", " ", text)

        lines = [line.strip() for line in text.split("\n") if line.strip()]

        cleaned_lines = []
        for line in lines:
            has_translation = "[" in line
            has_rotation = "(" in line

            if has_translation and has_rotation:
                joint_name = line.split(",", 1)[0].strip()
                payloads = re.findall(r"\[[^\]]*\]|\([^\)]*\)", line)
                translation_parts = [payload for payload in payloads if payload.startswith("[")]
                rotation_parts = [payload for payload in payloads if payload.startswith("(")]

                if translation_parts:
                    cleaned_lines.append(",".join([joint_name] + translation_parts))
                if rotation_parts:
                    cleaned_lines.append(",".join([joint_name] + rotation_parts))
            else:
                cleaned_lines.append(line)

        return "\n".join(cleaned_lines)

    def _repair_animation_text_with_llm(self, animation_text: str, validation_errors) -> str:
        if not hasattr(self, "repair_chain"):
            raise RuntimeError("Repair chain is not initialized. Call initialize_chain() first.")

        repaired = self.repair_chain.invoke(
            {
                "animation_text": animation_text,
                "validation_errors": "\n".join(str(error) for error in validation_errors),
            }
        )
        return self._enforce_newlines(repaired)

    def _validate_and_repair_animation(self, animation_text: str) -> str:
        return prepare_animation_text_for_execution(
            animation_text,
            repair_callback=self._repair_animation_text_with_llm,
        )

    def invoke_chain(self, input_dict: dict) -> str:
        if not hasattr(self, "chain"):
            raise RuntimeError("Chain is not initialized. Call initialize_chain() first.")

        prepared_input = self._prepare_input(input_dict)
        animation_text = self._enforce_newlines(self.chain.invoke(prepared_input))
        return self._validate_and_repair_animation(animation_text)


if __name__ == "__main__":
    keyframe = KeyFrameAgent()
    keyframe.initialize_chain()
    response = keyframe.invoke_chain(
        {
            "object": "racoon",
            "object_json": animation_examples[0]["object_json"],
            "user_instruction": "idle while moving head up and down",
            "current_plan_step": "Step 1: Rotate spine.006 +Z 10 degrees; rotate tail +Y 5 degrees;",
            "plan_history": [],
            "last_step_index": 0,
            "previous_animation": None,
        }
    )
    print(response)
