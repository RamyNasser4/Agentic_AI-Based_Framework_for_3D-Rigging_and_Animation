# Testing the Gemini API with a simple prompt to explain how AI works.
from os import getenv
from langchain_core.prompts import ChatPromptTemplate, FewShotChatMessagePromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain.chat_models import init_chat_model
from dotenv import load_dotenv

load_dotenv()

# Updated Planner System Prompt to match the required step-by-step, joint-specific output
PLANNER_SYSTEM_PROMPT = """You are an animation planner. Given a user's request, the object JSON hierarchy, and root directions, you will produce a clear, sequential plan detailing how to move the necessary joints to perform the given motion.

# Guidelines to follow
- Read the user's request carefully and produce a step-by-step plan.
- Output must be a series of numbered steps (e.g., Step 1:, Step 2:).
- Each step must be on a single line, followed by a newline for the next step.
- Each step consists of action phrases for each necessary joint, separated by semicolons.
- Every joint action must use simple directional language with exact numeric values (e.g., "rotate left_hip backward 30 degrees", "move root forward").
- Only include joints that are actively moving in that step. Never mention a joint if its value is 0 or unchanged.
- Explicitly use the exact joint names provided in the object JSON.
- Do not use vague language like "slightly", "a bit", or "gently" — always use exact numeric values for rotations.
- Do not output any introductory or concluding text. Generate output ONLY.
- Do not use adverbs or descriptive keywords like 'rapidly', 'smoothly', 'quickly', 'slowly', or 'continuously' — use only directional language and exact numeric values.

# Output format example
Step 1: Move root forward; rotate left_hip backward 30 degrees; rotate right_hip forward 30 degrees;
Step 2: Move root forward; rotate right_hip backward 30 degrees; rotate left_hip forward 30 degrees;
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
            "Step 1: Rotate spine.006 forward 30 degrees; rotate ear.L forward 10 degrees; rotate ear.R forward 10 degrees; rotate tail.001 downward 5 degrees.\n"
            "Step 2: Rotate spine.006 backward 30 degrees; rotate ear.L backward 10 degrees; rotate ear.R backward 10 degrees; rotate tail.001 upward 5 degrees.\n"
            "Step 3: Rotate spine.006 forward 30 degrees; rotate ear.L forward 10 degrees; rotate ear.R forward 10 degrees; rotate tail.001 downward 5 degrees.\n"
            "Step 4: Rotate spine.006 backward 30 degrees; rotate ear.L backward 10 degrees; rotate ear.R backward 10 degrees; rotate tail.001 upward 5 degrees.\n"
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
            "Step 1: Move Armature forward; rotate Spine1 downward 20 degrees; rotate Spine2 downward 15 degrees; rotate Head upward 10 degrees; rotate TopFlipper.L backward 20 degrees; rotate TopFlipper.R backward 20 degrees.\n"
            "Step 2: Move Armature forward; rotate Spine3 downward 20 degrees; rotate Spine4 downward 15 degrees; rotate Spine1 upward 20 degrees; rotate Spine2 upward 15 degrees; rotate Tail downward 25 degrees.\n"
            "Step 3: Move Armature forward; rotate Spine3 upward 20 degrees; rotate Spine4 upward 15 degrees; rotate Tail upward 25 degrees; rotate TopFlipper.L forward 20 degrees; rotate TopFlipper.R forward 20 degrees.\n"
            "Step 4: Move Armature forward; rotate Spine1 downward 20 degrees; rotate Spine2 downward 15 degrees; rotate Spine3 upward 20 degrees; rotate Tail downward 25 degrees; rotate TopFlipper.L backward 20 degrees; rotate TopFlipper.R backward 20 degrees.\n"
        )
    },
    {
        "object": "character",
        "object_json": "",
        "user_prompt": "Animate a character jumping over an obstacle.",
        "plan": (
            "Step 1: Move root forward; rotate left_knee forward 40 degrees; rotate right_knee forward 40 degrees; move pelvis downward; rotate spine1 forward 15 degrees; rotate left_shoulder backward 20 degrees; rotate right_shoulder backward 20 degrees.\n"
            "Step 2: Move root forward and upward; rotate left_knee backward 40 degrees; rotate right_knee backward 40 degrees; move pelvis upward; rotate spine1 backward 15 degrees; rotate left_shoulder forward 20 degrees; rotate right_shoulder forward 20 degrees.\n"
            "Step 3: Move root forward; rotate left_knee upward 20 degrees; rotate right_knee upward 20 degrees.\n"
            "Step 4: Move root forward and downward; rotate left_knee backward 30 degrees; rotate right_knee backward 30 degrees; rotate spine1 forward 10 degrees; rotate spine2 forward 10 degrees.\n"
            "Step 5: Move root downward; rotate left_knee forward 50 degrees; rotate right_knee forward 50 degrees; move pelvis downward; rotate left_shoulder downward 15 degrees; rotate right_shoulder downward 15 degrees.\n"
            "Step 6: Rotate left_knee backward 50 degrees; rotate right_knee backward 50 degrees; move pelvis upward; rotate spine1 upward 10 degrees.\n"
        )
    },
    {
        "object": "flag",
        "object_json": "",
        "user_prompt": "Animate a flag waving in the wind.",
        "plan": (
            "Step 1: Rotate flag_bone_1 sideward 10 degrees.\n"
            "Step 2: Rotate flag_bone_1 sideward 15 degrees; rotate flag_bone_2 sideward 10 degrees.\n"
            "Step 3: Rotate flag_bone_1 backward 10 degrees; rotate flag_bone_2 sideward 15 degrees; rotate flag_bone_3 sideward 10 degrees.\n"
            "Step 4: Rotate flag_bone_1 sideward 15 degrees; rotate flag_bone_2 backward 10 degrees; rotate flag_bone_3 sideward 15 degrees.\n"
        )
    },
    {
        "object": "character",
        "object_json": "",
        "user_prompt": "Make a character pick up an object from a table.",
        "plan": (
            "Step 1: Rotate pelvis forward 10 degrees; rotate spine1 forward 20 degrees; rotate spine2 forward 15 degrees.\n"
            "Step 2: Rotate spine1 forward 30 degrees; rotate right_shoulder forward 40 degrees; rotate right_shoulder upward 20 degrees; rotate right_elbow forward 30 degrees; rotate right_thumb outward 20 degrees; rotate right_index outward 20 degrees.\n"
            "Step 3: Rotate right_elbow forward 50 degrees; rotate right_thumb inward 20 degrees; rotate right_index inward 20 degrees.\n"
            "Step 4: Rotate right_shoulder backward 40 degrees; rotate right_elbow backward 30 degrees; rotate spine1 backward 30 degrees; rotate spine2 backward 15 degrees.\n"
            "Step 5: Rotate pelvis backward 10 degrees; rotate right_elbow inward 30 degrees.\n"
        )
    },
    {
        "object": "rocket",
        "object_json": "",
        "user_prompt": "Animate a rocket launching into the sky.",
        "plan": (
            "Step 1: Move rocket_root downward 5 degrees.\n"
            "Step 2: Move rocket_root upward 20 degrees.\n"
            "Step 3: Move rocket_root upward 40 degrees; rotate rocket_root forward 10 degrees.\n"
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
    # return init_chat_model(
    #     model=model,
    #     model_provider="openai",
    #     base_url="https://openrouter.ai/api/v1",
    #     api_key=getenv("OPENROUTER_API_KEY"),
    #     temperature=0,
    # )
    return init_chat_model(
                model="gpt-5-mini",
                model_provider="openai",
                base_url="http://localhost:4000/v1/",
                api_key="nothing",
                default_headers={
                    # "HTTP-Referer": getenv("YOUR_SITE_URL"),
                    # "X-OpenRouter-Title": getenv("YOUR_SITE_NAME"),
                },
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

class PlannerAgent:
    def __init__(self, models=None):
        self.models = models or FREE_MODELS
        self.chain = None
        self.working_model = None

    def _build_chain(self, model: str):
        llm = get_llm(model)
        return prompt_template | llm | StrOutputParser()

    def initialize_chain(self, model: str | None = None):
        if model is None:
            self.chain = None
            self.working_model = None
            return

        self.chain = self._build_chain(model)
        self.working_model = model

    def invoke_chain(self, input_dict: dict) -> str:
        if self.chain is not None:
            try:
                return self.chain.invoke(input_dict)
            except Exception as error:
                print(f"{self.working_model} failed: {str(error)[:60]}")
                self.chain = None
                self.working_model = None

        for model in self.models:
            try:
                self.chain = self._build_chain(model)
                self.working_model = model
                response = self.chain.invoke(input_dict)
                print(f"Working model: {model}")
                return response
            except Exception as error:
                print(f"{model} failed: {str(error)[:60]}")
                self.chain = None
                self.working_model = None
                continue

        return "All models failed."

    @staticmethod
    def get_example_object_json(object_name: str) -> str:
        for example in few_shots:
            if example["object"].lower() == object_name.lower():
                return example["object_json"]
        raise ValueError(f"No example object_json found for '{object_name}'.")


def run_llm(object_name, object_json, user_prompt):
    planner = PlannerAgent()
    return planner.invoke_chain({
        "object": object_name,
        "object_json": object_json,
        "user_prompt": user_prompt,
    })


if __name__ == "__main__":
    planner = PlannerAgent()
    planner.initialize_chain()

    whale_object_json = planner.get_example_object_json("whale")
    whale_plan = planner.invoke_chain({
        "object": "whale",
        "object_json": whale_object_json,
        "user_prompt": "Create tilt tail animation",
    })

    print(whale_plan)
