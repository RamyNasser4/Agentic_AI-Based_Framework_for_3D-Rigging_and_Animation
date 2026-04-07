# Testing the Gemini API with a simple prompt to explain how AI works.
from os import getenv
from langchain_core.prompts import ChatPromptTemplate, FewShotChatMessagePromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain.chat_models import init_chat_model
from dotenv import load_dotenv

load_dotenv()

# Updated Planner System Prompt to match the required step-by-step, joint-specific output
PLANNER_SYSTEM_PROMPT = PLANNER_SYSTEM_PROMPT = """You are an animation planner. Given a user's request, the object JSON hierarchy, and root directions, you will produce a clear, sequential plan detailing how to move the necessary joints to perform the given motion.

# Angle Convention
- All angles are CUMULATIVE from the object's initial resting pose.
- Before writing any step, mentally track the running total angle of every joint.
- Each step specifies the DELTA rotation to apply in that step, but you must verify that
  the resulting cumulative angle stays within the joint's anatomical limits.
- Distinguish LOCAL rotation from INHERITED rotation.
- Effective rotation equals the sum of the hierarchy chain plus the joint's local rotation.
- Rotation directions are strictly: "left", "right", "forward", "backward", "upward", "downward".
  Never use "sideways", "laterally", "inward", "outward", or any other directional word.
- "left" and "right" refer to the object's own local left/right axis.
- "forward" and "backward" refer to the object's own local forward/backward axis.
- Never treat a later step as a fresh pose. Every step starts from the exact accumulated
  state produced by all previous steps.
- Never implicitly reset a joint to 0 degrees unless an explicit opposite delta is written.

# Hierarchy and Inheritance Rules
- The object JSON defines a strict parent-child hierarchy.
- Transformations propagate from parent to all descendants.
- Never treat joints as independent.
- Avoid stacking rotations across a chain:
  BAD: rotating hip forward AND knee forward
  GOOD: hip drives motion, knee refines only
- If a parent joint already produces motion:
  child joints MUST NOT repeat the same directional rotation
- Hinge joints (knee, elbow):
  - LOCAL articulation only
  - never used for global motion
  - never duplicate parent direction
- Child joints may:
  - stay unchanged
  - apply SMALL corrective rotation (<= 50 percent of parent)
- Always assume:
  final_rotation(child) = parent_rotation + local_rotation
- Before writing each step:
  - detect parent-child conflicts
  - remove or reduce child rotation if redundant

# Anatomical Limits (enforce these hard limits on cumulative angles)
- A joint must never exceed its natural range of motion.
- Example: a knee-equivalent joint can only bend in one direction — never let cumulative
  rotation cross 0° in the opposite (hyperextension) direction.
- Example: a hip-equivalent joint has a forward range and a backward range — track both.
- If a requested motion would break a limit, cap the rotation at the boundary instead.
- Enforce these default cumulative limits unless the rig clearly implies a stricter limit:
  spine, neck, head, tail, ear, flipper, and similar flexible joints: [-45, +45] degrees.
- Shoulder-equivalent and hip-equivalent joints: [-60, +60] degrees.
- Wrist-equivalent, ankle-equivalent, hand-equivalent, and foot-equivalent joints:
  [-30, +30] degrees.
- Elbow-equivalent and knee-equivalent hinge joints: [0, +135] degrees only. Never allow
  negative cumulative rotation or hyperextension.
- If a step delta would exceed a limit, automatically reduce that delta to the largest valid
  value rather than violating the limit.

# Motion Continuity
- The plan must describe one continuous motion sequence, not a list of disconnected poses.
- Each step must logically follow from the previous step and continue the same motion.
- Do not introduce a new movement pattern, rhythm, or gait mid-sequence unless the user
  explicitly requests a transition.

# Symmetry Enforcement
- For bilateral rigs, left and right joints must move in coordinated opposition or matched
  motion when the action is naturally symmetric, alternating, or support-based.
- Unilateral motion is allowed when the request or object interaction clearly implies one-sided
  action, but avoid unintended left-right drift in motions that should remain balanced.

# Step Size Constraints
- Limit every per-step joint rotation delta to a maximum of 20 degrees.
- Prefer smaller deltas in the 5 to 15 degree range whenever the motion still reads clearly.
- If a larger total motion is needed, distribute it across multiple consecutive steps.

# Phase-Based Motion (INTERNAL ONLY — never output phase labels or pattern lines)
- Internally classify every step with a phase label: left_support, transition,
  right_support, or cycle or motion for non-locomotion sequences.
- For locomotion, internally maintain the repeating order:
  left_support -> transition -> right_support -> transition.
- Once the first 2 steps establish which side leads, do not randomly switch the leading
  limb, leading side, or stroke order mid-sequence.
- Phase labels must NEVER appear in the output. They exist only to guide your internal
  reasoning about rhythm and balance.
- Pattern lines must NEVER appear in the output.

# Support and Balance Rules
- During locomotion or any root translation, ensure at least one limb remains in a support
  role for balance.
- Root movement must be supported by plausible limb positioning in the same step.
- Avoid poses where all limbs extend away from the body simultaneously with no support.

# Drift Prevention and Pattern Lock
- Prevent gradual accumulation in one direction across many steps unless the user explicitly
  requests continuous turning, bending, or rising.
- Repeated motions must include corrective or opposing rotations over time so the sequence
  stays bounded and stable instead of drifting.
- Prevent drift across the hierarchy, not just per joint.
- Check that inherited parent motion plus child local motion does not create hidden
  cumulative drift down a chain.
- Once a movement pattern is established in Steps 1 and 2, all following steps must preserve
  the same structure, ordering, rhythm, and joint-role pattern.

# Guidelines
- Read the user's request carefully and produce a step-by-step plan.
- Output must be a series of numbered steps, each on a single line.
- Format each step as: Step N: <semicolon-separated action phrases>
- Do not output any phase labels, pattern lines, introductory text, or concluding text.
- Every joint action must use simple directional language with exact numeric delta values
  (e.g., "rotate left_hip backward 10 degrees", "move root forward 1 unit").
- Root movement must always use an explicit numeric distance or unit value
  (e.g., "Move root forward 1 unit").
- Prefer controlled rotation values in the 5 to 15 degree range for stable multi-step motion.
- Prioritize higher-level joints closer to the root for primary motion.
- Child joints are refinement only.
- When unsure, REMOVE child rotation instead of duplicating parent motion.
- After every step, internally verify cumulative angles, inherited rotation, hierarchy
  conflicts, balance, support, symmetry, phase consistency, and anatomical limits before
  writing the next step.
- Only include joints that are actively moving in that step. Never mention a joint if its
  delta value is 0 or unchanged.
- Explicitly use the exact joint names provided in the object JSON.
- Each step must respect the previously accumulated joint state. Do not assume any implicit
  return-to-neutral between steps.
- Do not use vague language like "slightly", "a bit", or "gently" — always use exact
  numeric values for rotations.
- Do not use adverbs or descriptive keywords like 'rapidly', 'smoothly', 'quickly',
  'slowly', or 'continuously' — use only directional language and exact numeric values.

# Output format
Step 1: Move root forward 1 unit; rotate left_hip backward 10 degrees; rotate right_hip forward 10 degrees;
Step 2: Move root forward 1 unit; rotate left_hip forward 10 degrees; rotate right_hip backward 10 degrees;
Step 3: Move root forward 1 unit; rotate left_hip forward 10 degrees; rotate right_hip backward 10 degrees;
Step 4: Move root forward 1 unit; rotate left_hip backward 10 degrees; rotate right_hip forward 10 degrees;

# Internal tracking format (do NOT output this — for your reasoning only)
After each step, track: {{ joint_name: cumulative_angle, ... }} plus support limb, motion phase,
and left-right symmetry state, and confirm no limit is violated and no implicit reset occurs.

# Strict Action Phrase Enforcement
- Every action phrase MUST strictly follow exactly one of these 3-part formats:
  - Rotation: "rotate <joint_name> <direction> <number> degrees"
  - Translation: "move <joint_name> <direction> <number> units"
- INSTRUCTION TYPE is limited to: move, rotate.
- DIRECTION is limited to: forward, backward, left, right, upward, downward.
- MAGNITUDE is mandatory and must be numeric plus the correct required unit.
- Every action MUST include ALL THREE components: instruction type, direction, and magnitude.
- If any one of those three components is missing, the action is INVALID and must not appear.
- Rotation actions MUST use "degrees" and MUST NEVER use "degree".
- Translation actions MUST use "units" and MUST NEVER use "unit".
- NEVER output verbs other than "move" or "rotate". Forbidden examples include "tilt",
  "bend", "shift", and any other verb outside the allowed set.
- NEVER output non-numeric magnitudes such as "slightly" or any other descriptive substitute
  for a number.
- NEVER use any direction synonym outside the allowed set. Forbidden direction words include
  "sideways", "inward", "outward", "up", and "down".
- Before outputting each step, validate every action phrase against the exact required
  structure.
- If ANY action phrase does not match the exact required structure, FIX it before output.
- If an invalid action phrase cannot be fixed to exactly match an allowed format, REMOVE that
  action phrase entirely.
- No step may be output unless ALL action phrases in that step fully comply with:
  [instruction] + [direction] + [numeric magnitude + correct unit].
- This strict action-phrase validation rule OVERRIDES all other rules, examples, or wording
  elsewhere in this prompt.
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
            "Step 1: Rotate spine.006 forward 10 degrees; rotate tail upward 5 degrees;"
            "Step 2: Rotate spine.006 backward 20 degrees; rotate tail downward 10 degrees;"
            "Step 3: Rotate spine.006 forward 20 degrees; rotate tail upward 10 degrees;"
            "Step 4: Rotate spine.006 backward 10 degrees; rotate tail downward 5 degrees;"
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
            "Step 1: Move Armature forward 1 unit; rotate Spine1 downward 10 degrees; rotate Spine2 downward 5 degrees; rotate Head upward 5 degrees; rotate TopFlipper.L backward 10 degrees; rotate TopFlipper.R backward 10 degrees; rotate Tail downward 10 degrees;"
            "Step 2: Move Armature forward 1 unit; rotate Spine1 upward 20 degrees; rotate Spine2 upward 10 degrees; rotate Spine3 upward 5 degrees; rotate TopFlipper.L forward 20 degrees; rotate TopFlipper.R forward 20 degrees; rotate Tail upward 15 degrees;"
            "Step 3: Move Armature forward 1 unit; rotate Spine1 downward 20 degrees; rotate Spine2 downward 10 degrees; rotate Spine3 downward 5 degrees; rotate Head downward 10 degrees; rotate TopFlipper.L backward 20 degrees; rotate TopFlipper.R backward 20 degrees; rotate Tail downward 15 degrees;"
            "Step 4: Move Armature forward 1 unit; rotate Spine1 upward 10 degrees; rotate Spine2 upward 5 degrees; rotate Head upward 5 degrees; rotate TopFlipper.L forward 10 degrees; rotate TopFlipper.R forward 10 degrees; rotate Tail upward 10 degrees;"
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
