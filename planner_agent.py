# Testing the Gemini API with a simple prompt to explain how AI works.
from os import getenv
from langchain_core.prompts import ChatPromptTemplate, FewShotChatMessagePromptTemplate
from langchain.chat_models import init_chat_model
from dotenv import load_dotenv

load_dotenv()

# Updated Planner System Prompt to match the required step-by-step, joint-specific output
PLANNER_SYSTEM_PROMPT = """You are an animation planner. Given a user's request, the object JSON hierarchy, and root directions, you will produce a clear, sequential plan detailing how to move the necessary joints to perform the given motion.

# Guidelines to follow
- Read the user's request carefully and produce a step-by-step plan.
- Output must be a series of numbered steps (e.g., Step 1:, Step 2:).
- Each step must be on a single line, followed by a newline for the next step.
- Each step consists of an action sentence for each necessary joint, separated by semicolons (e.g., "Move root forward; bend left_knee; rotate spine1.").
- Explicitly use the exact joint names provided in the object JSON.
- Use direction vectors (e.g., (0, 1, 0)) when describing root translational or rotational movement based on the provided root directions.
- Do not output any introductory or concluding text. Generate output ONLY.
"""

# Updated few shots with the specific step-by-step formatting
few_shots = [
    {
        "object": "racoon",
        "object_json": (
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
        "user_prompt": "Animate the raccoon standing still while nodding its head up and down.",
        "plan": (
            "Step 1: Keep metarig stationary; stabilize spine, pelvis.L, and pelvis.R; maintain tail in default rotation.\n"
            "Step 2: Rotate spine.006 forward to lower the head; apply slight forward wobble to ear.L and ear.R; apply subtle downward rotation to tail.001.\n"
            "Step 3: Rotate spine.006 backward to raise the head; apply slight backward wobble to ear.L and ear.R; reverse subtle rotation on tail.001.\n"
            "Step 4: Repeat rotation of spine.006 forward; continue passive motion on ear.L and ear.R; maintain idle sway on tail.\n"
            "Step 5: Return spine.006 to upright position; settle ear.L and ear.R; keep thigh.L, shin.L, thigh.R, and shin.R stationary."
        ),
    },
    {
        "object": "whale",
        "object_json":(
            "name:Armature,position:(0.0000,0.0000,0.0000),rotation:(-0.7,0.0,0.0,0.7),children:[name:Root,position:(0.0000,0.0168,0.0141),rotation:(0.7,0.0,0.0,0.7),children:[name:Head,position:(0.0000,0.0062,0.0198),rotation:(0.7,0.0,0.0,0.7),children:[name:Head_end,position:(0.0000,0.0107,0.0000),rotation:(0.0,0.0,0.0,1.0)]," 
            "name:Spine1,position:(0.0000,0.0050,0.0154),rotation:(-0.7,0.0,0.0,0.7),children:[name:Spine2,position:(0.0000,0.0156,0.0000),rotation:(0.0,0.0,0.0,1.0),children:[name:Spine3,position:(0.0000,0.0166,0.0000),rotation:(0.0,0.0,0.0,1.0),children:[name:Spine4,position:(0.0000,0.0172,0.0000),rotation:(-0.1,0.0,0.0,1.0)," 
            "children:[name:Tail,position:(0.0000,0.0196,0.0000),rotation:(0.0,0.0,0.0,1.0),children:[name:Tail_end,position:(0.0000,0.0133,0.0000),rotation:(0.0,0.0,0.0,1.0)]]]],name:TopFlipper.L,position:(-0.0107,0.0087,-0.0087),rotation:(-0.4,0.0,0.3,0.9),children:[name:MidFlipper.L,position:(0.0000,0.0067,0.0000),rotation:(0.0,0.1,0.0,1.0)," 
            "children:[name:BottomFlipper.L,position:(0.0000,0.0043,0.0000),rotation:(0.0,0.0,-0.1,1.0),children:[name:BottomFlipper.L_end,position:(0.0000,0.0076,0.0000),rotation:(0.0,0.0,0.0,1.0)]]],name:TopFlipper.R,position:(0.0092,0.0078,-0.0084),rotation:(-0.4,0.0,-0.3,0.9),children:[name:MidFlipper.R,position:(0.0000,0.0082,0.0000),rotation:(0.1,-0.1,0.1,1.0)," 
            "children:[name:BottomFlipper.R,position:(0.0000,0.0053,0.0000),rotation:(0.0,0.0,0.2,1.0),children:[name:BottomFlipper.R_end,position:(0.0000,0.0072,0.0000),rotation:(0.0,0.0,0.0,1.0)]]]]]]. " 
            "Root forward direction: (0.00, 1.00, 0.00); right direction: (1.00, 0.00, 0.00); up direction: (0.00, 0.00,-1.00). "
        ),
        "user_prompt": "Create a swim animation for the whale.",
        "plan": (
            "Step 1: Move Armature forward along (0, 1, 0); keep Root stable; prepare Spine1 and Spine2 for downward wave rotation.\n"
            "Step 2: Move Armature forward along (0, 1, 0); rotate Spine1 and Spine2 downward; rotate Head slightly upward; rotate TopFlipper.L and TopFlipper.R backward.\n"
            "Step 3: Move Armature forward along (0, 1, 0); rotate Spine3 and Spine4 downward; rotate Spine1 and Spine2 upward; rotate Tail downward.\n"
            "Step 4: Move Armature forward along (0, 1, 0); rotate Spine3 and Spine4 upward; rotate Tail upward; rotate TopFlipper.L and TopFlipper.R forward.\n"
            "Step 5: Continue continuous Armature forward motion along (0, 1, 0); cycle spine bones and Tail in cascading wave; cycle TopFlipper.L and TopFlipper.R to balance motion."
        )
    },
    {
        "object": "character",
        "object_json": "",
        "user_prompt": "Animate a character jumping over an obstacle.",
        "plan": (
            "Step 1: Move root forward along (0, 1, 0); bend left_knee and right_knee; lower pelvis; rotate spine1 forward; swing left_shoulder and right_shoulder backward.\n"
            "Step 2: Move root forward and upward along (0, 1, 1); extend left_knee and right_knee rapidly; raise pelvis; straighten spine1; swing left_shoulder and right_shoulder forward.\n"
            "Step 3: Move root forward along (0, 1, 0) at peak vertical height; tuck left_knee and right_knee slightly upward; stabilize pelvis; keep head facing forward.\n"
            "Step 4: Move root forward and downward along (0, 1, -1); extend left_knee and right_knee to prepare for landing; brace spine1 and spine2.\n"
            "Step 5: Move root forward to ground level; bend left_knee and right_knee deeply to absorb impact; lower pelvis; drop left_shoulder and right_shoulder.\n"
            "Step 6: Center pelvis over feet; extend left_knee and right_knee to standing position; align spine1 upright."
        )
    },
    {
        "object": "flag",
        "object_json": "",
        "user_prompt": "Animate a flag waving in the wind.",
        "plan": (
            "Step 1: Keep pole_root stationary; initiate slight rotation on flag_bone_1 along the wind direction.\n"
            "Step 2: Hold pole_root stationary; rotate flag_bone_1 further; initiate offset rotation on flag_bone_2.\n"
            "Step 3: Rotate flag_bone_1 back toward center; rotate flag_bone_2 further; initiate offset rotation on flag_bone_3.\n"
            "Step 4: Cycle flag_bone_1, flag_bone_2, flag_bone_3 in alternating wave pattern to simulate continuous wind ripples."
        )
    },
    {
        "object": "character",
        "object_json": "",
        "user_prompt": "Make a character pick up an object from a table.",
        "plan": (
            "Step 1: Keep root stationary; rotate pelvis slightly forward; bend spine1 and spine2 forward; keep left_arm relaxed.\n"
            "Step 2: Continue bending spine1 forward; rotate right_shoulder forward and upward; extend right_elbow toward object; open right_thumb and right_index.\n"
            "Step 3: Hold spine1 position; fully extend right_elbow; close right_thumb and right_index around object.\n"
            "Step 4: Rotate right_shoulder backward; bend right_elbow to lift object; rotate spine1 and spine2 backward to upright position.\n"
            "Step 5: Center pelvis; align spine1 upright; keep right_elbow bent holding object securely."
        )
    },
    {
        "object": "rocket",
        "object_json": "",
        "user_prompt": "Animate a rocket launching into the sky.",
        "plan": (
            "Step 1: Keep rocket_root stationary; trigger initial ignition particle effect.\n"
            "Step 2: Move rocket_root slightly downward to simulate squash and anticipation; maintain stable vertical orientation.\n"
            "Step 3: Move rocket_root rapidly upward along (0, 0, 1); keep rotation strictly vertical.\n"
            "Step 4: Continue moving rocket_root upward along (0, 0, 1) with increasing speed; rotate rocket_root slightly toward (0, 1, 0) for trajectory arc."
        )
    },
]

FREE_MODELS = [
"openrouter/free",
"stepfun/step-3.5-flash:free",
"arcee-ai/trinity-large-preview:free",
"liquid/lfm-2.5-1.2b-thinking:free",
"liquid/lfm-2.5-1.2b-instruct:free",
"nvidia/nemotron-3-nano-30b-a3b:free",
"arcee-ai/trinity-mini:free",
"nvidia/nemotron-nano-12b-v2-vl:free",
"qwen/qwen3-vl-30b-a3b-thinking",
"qwen/qwen3-vl-235b-a22b-thinking",
"qwen/qwen3-next-80b-a3b-instruct:free",
"nvidia/nemotron-nano-9b-v2:free",
"openai/gpt-oss-120b:free",
"openai/gpt-oss-20b:free",
"z-ai/glm-4.5-air:free",
"qwen/qwen3-coder:free",
"cognitivecomputations/dolphin-mistral-24b-venice-edition:free",
"google/gemma-3n-e2b-it:free",
"google/gemma-3n-e4b-it:free",
"qwen/qwen3-4b:free",
"mistralai/mistral-small-3.1-24b-instruct:free",
"google/gemma-3-4b-it:free",
"google/gemma-3-12b-it:free",
"google/gemma-3-27b-it:free",
"meta-llama/llama-3.3-70b-instruct:free",
"meta-llama/llama-3.2-3b-instruct:free",
"nousresearch/hermes-3-llama-3.1-405b:free",
]

def get_llm(model: str):
    return init_chat_model(
        model=model,
        model_provider="openai",
        base_url="https://openrouter.ai/api/v1",
        api_key=getenv("OPENROUTER_API_KEY"),
        temperature=0,
    )

example_prompt = ChatPromptTemplate.from_messages([
    ("human", "Object: **{object}**. Object JSON: {object_json}. Request: {user_prompt}."),
    ("ai", "{plan}"),
])

few_shot_prompt = FewShotChatMessagePromptTemplate(
    examples=few_shots,
    example_prompt=example_prompt,
)

# Main prompt with few_shot_prompt injected between system and human
prompt_template = ChatPromptTemplate.from_messages([
    ("system", PLANNER_SYSTEM_PROMPT),
    few_shot_prompt,
    ("human", (
        "The object you will make the animation plan for is **{object}**. "
        "Object JSON: {object_json}. "
        "The user's request is: {user_prompt}."
    )),
])

# Auto-fallback logic
def run_llm(object_name, object_json, user_prompt):
    for model in FREE_MODELS:
        try:
            llm = get_llm(model)
            chain = prompt_template | llm

            response = chain.invoke({
                "object": object_name,
                "object_json": object_json,
                "user_prompt": user_prompt,
            })

            print(f"Working model: {model}")
            return response.content

        except Exception as e:
            print(f"{model} failed: {str(e)[:60]}")
            continue

    return "All models failed."