# Testing the Gemini API with a simple prompt to explain how AI works.
import math
import re
from os import getenv
from langchain_core.prompts import ChatPromptTemplate, FewShotChatMessagePromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain.chat_models import init_chat_model
from langchain_openai import ChatOpenAI
from dotenv import load_dotenv
from langchain_openai import ChatOpenAI

load_dotenv()

PLANNER_MODEL = "gemma-4-31b-it"

# Updated Planner System Prompt to match the required step-by-step, joint-specific axis output
PLANNER_SYSTEM_PROMPT = """You are an animation planner. Given a user's request, the object JSON hierarchy, and local axis basis, you will produce a clear, sequential plan detailing how to move the necessary joints to perform the given motion.

# Angle Convention
- All angles are CUMULATIVE from the object's initial resting pose.
- Before writing any step, mentally track the running total angle of every joint.
- Each step specifies the DELTA rotation to apply in that step, but you must verify that
  the resulting cumulative angle stays within the joint's anatomical limits.
- Distinguish LOCAL rotation from INHERITED rotation.
- Effective rotation equals the sum of the hierarchy chain plus the joint's local rotation.
- Directions must use axis notation: +X, -X, +Y, -Y, +Z, -Z
- These correspond to the object's local axes.
- Never treat a later step as a fresh pose. Every step starts from the exact accumulated
  state produced by all previous steps.
- Never implicitly reset a joint to 0 degrees unless an explicit opposite delta is written.

# Hierarchy and Inheritance Rules
- The object JSON defines a strict parent-child hierarchy.
- Transformations propagate from parent to all descendants.
- Never treat joints as independent.
- Avoid stacking rotations across a chain:
  BAD: rotating hip +Z AND knee +Z
  GOOD: hip drives motion, knee refines only
- If a parent joint already produces motion:
  child joints MUST NOT repeat the same axis rotation
- Hinge joints (knee, elbow):
  - LOCAL articulation only
  - never used for global motion
  - never duplicate parent axis
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
- Example: a hip-equivalent joint can have both positive and negative range on a local axis — track both.
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
- For bilateral rigs, paired mirror joints must move in coordinated opposition or matched
  motion when the action is naturally symmetric, alternating, or support-based.
- Unilateral motion is allowed when the request or object interaction clearly implies one-sided
  action, but avoid unintended mirror drift in motions that should remain balanced.

# Step Size Constraints
- Limit every per-step joint rotation delta to a maximum of 20 degrees.
- Prefer smaller deltas in the 5 to 15 degree range whenever the motion still reads clearly.
- If a larger total motion is needed, distribute it across multiple consecutive steps.

# Phase-Based Motion (INTERNAL ONLY — never output phase labels or pattern lines)
- Internally classify every step with a phase label: support_a, transition,
  support_b, or cycle or motion for non-locomotion sequences.
- For locomotion, internally maintain the repeating order:
  support_a -> transition -> support_b -> transition.
- Once the first 2 steps establish which support side leads, do not randomly switch the leading
  limb, support side, or stroke order mid-sequence.
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
- Every joint action must use axis notation with exact numeric delta values
  (e.g., "rotate left_hip -Z 10 degrees", "move root +Z 1 units").
- Root movement must always use an explicit numeric distance or unit value
  (e.g., "move root +Z 1 units").
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
Step 1: move root +Z 1 units; rotate left_hip -Z 10 degrees; rotate right_hip +Z 10 degrees;
Step 2: move root +Z 1 units; rotate left_hip +Z 10 degrees; rotate right_hip -Z 10 degrees;
Step 3: move root +Z 1 units; rotate left_hip +Z 10 degrees; rotate right_hip -Z 10 degrees;
Step 4: move root +Z 1 units; rotate left_hip -Z 10 degrees; rotate right_hip +Z 10 degrees;

# Internal tracking format (do NOT output this — for your reasoning only)
After each step, track: {{ joint_name: cumulative_angle, ... }} plus support limb, motion phase,
and mirror symmetry state, and confirm no limit is violated and no implicit reset occurs.

# Strict Action Phrase Enforcement
- Every action phrase MUST strictly follow exactly one of these 3-part formats:
  - Rotation: "rotate <joint_name> <direction> <number> degrees"
  - Translation: "move <joint_name> <direction> <number> units"
- INSTRUCTION TYPE is limited to: move, rotate.
- DIRECTION is limited to: +X, -X, +Y, -Y, +Z, -Z
- MAGNITUDE is mandatory and must be numeric plus the correct required unit.
- Every action MUST include ALL THREE components: instruction type, direction, and magnitude.
- If any one of those three components is missing, the action is INVALID and must not appear.
- Rotation actions MUST use "degrees" and MUST NEVER use "degree".
- Translation actions MUST use "units" and MUST NEVER use "unit".
- NEVER output verbs other than "move" or "rotate". Forbidden examples include "tilt",
  "bend", "shift", and any other verb outside the allowed set.
- NEVER output non-numeric magnitudes such as "slightly" or any other descriptive substitute
  for a number.
- NEVER use any direction token outside the allowed set.
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

# Updated few shots with the specific step-by-step axis formatting
few_shots = [
    {
        "object": "racoon",
        "object_json": (
           """name:metarig,position:(0.0,0.0,0.0),rotation:(0.0,0.0,0.0,1.0),children:[name:spine,position:(0.0,0.0,0.0),rotation:(0.0,0.0,0.0,1.0),children:[name:spine.001,position:(0.0,0.0,0.0),rotation:(0.0,0.0,0.0,1.0),children:[name:spine.002,position:(0.0,0.0,0.0),rotation:(0.0,0.0,0.0,1.0),children:[name:spine.003,position:(0.0,0.0,0.0),rotation:(0.0,0.0,0.0,1.0),children:[name:spine.006,position:(0.0,0.0,0.0),rotation:(0.0,0.0,0.0,1.0),children:[name:ear.L,position:(0.0,0.0,0.0),rotation:(0.0,0.0,0.0,1.0),name:ear.R,position:(0.0,0.0,0.0),rotation:(0.0,0.0,0.3,0.9)],name:shoulder.L,position:(0.0,0.0,0.0),rotation:(0.0,0.0,0.0,1.0),children:[name:upper_arm.L,position:(0.0,0.0,0.0),rotation:(0.0,0.0,0.0,1.0),children:[name:forearm.L,position:(0.0,0.0,0.0),rotation:(0.4,0.0,0.0,0.9),children:[name:hand.L,position:(0.0,0.0,0.0),rotation:(0.0,0.0,0.0,1.0)]]],name:shoulder.R,position:(0.0,0.0,0.0),rotation:(0.0,0.0,0.0,1.0),children:[name:upper_arm.R,position:(0.0,0.0,0.0),rotation:(-0.2,0.0,0.3,0.9),children:[name:forearm.R,position:(0.0,0.0,0.0),rotation:(-0.2,-0.1,0.6,0.7),children:[name:hand.R,position:(0.0,0.0,0.0),rotation:(0.0,0.0,0.2,1.0),children:[name:hand.R.001,position:(0.0,-0.2,0.0),rotation:(0.0,0.0,-0.1,1.0)]]]],name:breast.L,position:(0.0,0.0,0.0),rotation:(0.0,0.0,0.0,1.0),name:breast.R,position:(0.0,0.0,0.0),rotation:(0.0,0.0,0.0,1.0)]]],name:pelvis.L,position:(0.0,0.0,0.0),rotation:(0.0,0.0,0.0,1.0),name:pelvis.R,position:(0.0,0.0,0.0),rotation:(0.0,0.0,0.0,1.0),name:thigh.L,position:(0.0,0.0,0.0),rotation:(0.0,0.0,0.0,1.0),children:[name:shin.L,position:(0.0,0.0,0.0),rotation:(0.0,0.0,0.0,1.0),children:[name:foot.L,position:(0.0,0.0,0.0),rotation:(0.0,0.0,0.0,1.0),children:[name:toe.L,position:(0.0,0.0,0.0),rotation:(0.0,0.0,0.0,1.0),name:heel.02.L,position:(0.0,0.0,0.0),rotation:(0.0,0.0,0.0,1.0)]]],name:thigh.R,position:(0.0,0.0,0.0),rotation:(0.0,0.0,0.0,1.0),children:[name:shin.R,position:(0.0,0.0,0.0),rotation:(0.0,0.0,0.0,1.0),children:[name:foot.R,position:(0.0,0.0,0.0),rotation:(0.0,0.0,0.0,1.0),children:[name:toe.R,position:(0.0,0.0,0.0),rotation:(0.0,0.0,0.0,1.0),name:heel.02.R,position:(0.0,0.0,0.0),rotation:(0.0,0.0,0.0,1.0)]]],name:tail,position:(0.0,0.0,0.0),rotation:(-0.2,0.3,0.2,0.9),children:[name:tail.001,position:(0.0,0.0,0.0),rotation:(-0.3,0.0,0.2,0.9),children:[name:tail.002,position:(0.0,0.0,0.0),rotation:(0.0,-0.5,0.5,0.7),children:[name:tail.003,position:(0.0,0.0,0.0),rotation:(-0.6,0.0,0.0,0.8)]]]]]
Root axis +Z: (0.0, 0.0, 1.0); axis +X: (1.0, 0.0, 0.0); axis +Y: (0.0, -1.0, 0.0)"""
        ),
        "user_prompt": "Animate the raccoon standing still while nodding its head up and down.",
        "plan": (
            "Step 1: Rotate spine.006 +Z 10 degrees; rotate tail +Y 5 degrees;"
            "Step 2: Rotate spine.006 -Z 10 degrees; rotate tail -Y 10 degrees;"
            "Step 3: Rotate spine.006 +Z 10 degrees; rotate tail +Y 10 degrees;"
            "Step 4: Rotate spine.006 -Z 10 degrees; rotate tail -Y 5 degrees;"
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
            "Root axis +Z: (0.00, 1.00, 0.00); axis +X: (1.00, 0.00, 0.00); axis +Y: (0.00, 0.00,-1.00). "
        ),
        "user_prompt": "Create a swim animation for the whale.",
        "plan": (
            "Step 1: Move Armature +Z 1 units; rotate Spine1 -Y 10 degrees; rotate Spine2 -Y 5 degrees; rotate Head +Y 5 degrees; rotate TopFlipper.L -Z 10 degrees; rotate TopFlipper.R -Z 10 degrees; rotate Tail -Y 10 degrees;"
            "Step 2: Move Armature +Z 1 units; rotate Spine1 +Y 20 degrees; rotate Spine2 +Y 10 degrees; rotate Spine3 +Y 5 degrees; rotate TopFlipper.L +Z 20 degrees; rotate TopFlipper.R +Z 20 degrees; rotate Tail +Y 15 degrees;"
            "Step 3: Move Armature +Z 1 units; rotate Spine1 -Y 20 degrees; rotate Spine2 -Y 10 degrees; rotate Spine3 -Y 5 degrees; rotate Head -Y 10 degrees; rotate TopFlipper.L -Z 20 degrees; rotate TopFlipper.R -Z 20 degrees; rotate Tail -Y 15 degrees;"
            "Step 4: Move Armature +Z 1 units; rotate Spine1 +Y 10 degrees; rotate Spine2 +Y 5 degrees; rotate Head +Y 5 degrees; rotate TopFlipper.L +Z 10 degrees; rotate TopFlipper.R +Z 10 degrees; rotate Tail +Y 10 degrees;"
        )
    },
    {
    "object": "human male",
    "object_json": ("name:bvh_output000075,position:(0.0,0.0,0.0),rotation:(0.0,0.0,0.0,1.0),children:[name:Pelvis,position:(0.0,-0.8,0.4),rotation:(0.0,0.0,0.0,1.0),children:[name:Left_hip,position:(0.0,0.0,0.0),rotation:(0.0,0.0,0.0,1.0),children:[name:Left_knee,position:(0.0,0.0,0.0),rotation:(0.0,0.0,0.1,1.0),children:[name:Left_ankle,position:(0.0,0.0,0.0),rotation:(-0.1,0.0,0.0,1.0),children:[name:Left_foot,position:(0.0,0.0,0.0),rotation:(0.0,0.0,0.0,1.0)]]],name:Right_hip,position:(0.0,0.0,0.0),rotation:(0.0,0.0,0.0,1.0),children:[name:Right_knee,position:(0.0,0.0,0.0),rotation:(0.0,0.0,0.0,1.0),children:[name:Right_ankle,position:(0.0,0.0,0.0),rotation:(-0.2,0.0,0.0,1.0),children:[name:Right_foot,position:(0.0,0.0,0.0),rotation:(0.0,0.0,0.0,1.0)]]],name:Spine1,position:(0.0,0.0,0.0),rotation:(0.0,0.0,0.0,1.0),children:[name:Spine2,position:(0.0,0.0,0.0),rotation:(0.0,0.0,0.0,1.0),children:[name:Spine3,position:(0.0,0.0,0.0),rotation:(0.0,0.0,0.0,1.0),children:[name:Neck,position:(0.0,0.0,0.0),rotation:(0.0,0.0,0.0,1.0),children:[name:Head,position:(0.0,0.0,0.0),rotation:(0.1,0.0,0.0,1.0)],name:Left_collar,position:(0.0,0.0,0.0),rotation:(0.0,0.0,-0.3,0.9),children:[name:Left_shoulder,position:(0.0,0.0,0.0),rotation:(0.0,0.1,-0.4,0.9),children:[name:Left_elbow,position:(0.0,0.0,0.0),rotation:(0.3,0.0,0.1,1.0),children:[name:Left_wrist,position:(0.0,0.0,0.0),rotation:(0.0,0.2,0.0,1.0),children:[name:Left_palm,position:(0.0,0.0,0.0),rotation:(0.0,0.0,0.0,1.0)]]]],name:Right_collar,position:(0.0,0.0,0.0),rotation:(0.0,0.0,0.4,0.9),children:[name:Right_shoulder,position:(0.0,0.0,0.0),rotation:(0.0,-0.1,0.3,0.9),children:[name:Right_elbow,position:(0.0,0.0,0.0),rotation:(0.2,0.0,-0.1,1.0),children:[name:Right_wrist,position:(0.0,0.0,0.0),rotation:(0.0,-0.2,0.0,1.0),children:[name:Right_palm,position:(0.0,0.0,0.0),rotation:(0.0,0.0,0.0,1.0)]]]]]]]]]"
                    "Root forward direction: (0.0, 0.5, -0.9); right direction: (1.0, 0.0, 0.0); up direction: (0.0, 0.9, 0.5)"),
    "user_prompt": "a man walking forward slowly.",
    "plan": (
        "Step 1: Move bvh_output000075 -Z 0 units; rotate Pelvis +X 0 degrees; rotate Spine1 -X 3 degrees -Z 4 degrees; rotate Left_hip +X 0 degrees; rotate Right_hip +X 0 degrees; rotate Left_knee +X 0 degrees; rotate Right_knee +X 0 degrees; rotate Left_ankle -X 6 degrees; rotate Right_ankle -X 12 degrees; rotate Left_shoulder -Z 20 degrees; rotate Right_shoulder +Z 20 degrees; rotate Left_elbow +X 0 degrees; rotate Right_elbow +X 0 degrees;"
        "Step 2: Move bvh_output000075 -Z 1 units; rotate Pelvis +X 5 degrees -Y 3 degrees; rotate Spine1 +X 8 degrees -Y 6 degrees +Z 1 degrees; rotate Left_hip -X 20 degrees -Z 5 degrees; rotate Right_hip +X 25 degrees +Z 5 degrees; rotate Left_knee +X 35 degrees; rotate Right_knee -X 10 degrees; rotate Left_ankle -X 14 degrees; rotate Right_ankle +X 2 degrees; rotate Left_shoulder -X 25 degrees +Z 5 degrees; rotate Right_shoulder +X 20 degrees -Z 5 degrees; rotate Left_elbow +X 20 degrees; rotate Right_elbow -X 15 degrees;"
        "Step 3: Move bvh_output000075 -Z 1 units; rotate Pelvis -X 6 degrees +Y 8 degrees; rotate Spine1 -X 6 degrees +Y 11 degrees +Z 1 degrees; rotate Left_hip +X 50 degrees +Z 10 degrees; rotate Right_hip -X 45 degrees -Z 10 degrees; rotate Left_knee -X 40 degrees; rotate Right_knee +X 55 degrees; rotate Left_ankle +X 5 degrees; rotate Right_ankle -X 10 degrees; rotate Left_shoulder +X 50 degrees -Z 30 degrees; rotate Right_shoulder -X 45 degrees +Z 30 degrees; rotate Left_elbow -X 40 degrees; rotate Right_elbow +X 35 degrees;"
        "Step 4: Move bvh_output000075 -Z 1 units; rotate Pelvis +X 4 degrees -Y 12 degrees; rotate Spine1 +X 4 degrees -Y 12 degrees +Z 0 degrees; rotate Left_hip -X 25 degrees -Z 10 degrees; rotate Right_hip +X 25 degrees +Z 10 degrees; rotate Left_knee +X 45 degrees; rotate Right_knee -X 50 degrees; rotate Left_ankle -X 5 degrees; rotate Right_ankle +X 5 degrees; rotate Left_shoulder -X 45 degrees +Z 15 degrees; rotate Right_shoulder +X 35 degrees -Z 15 degrees; rotate Left_elbow +X 38 degrees; rotate Right_elbow -X 53 degrees;"
    )
    },
    {
    "object": "human male",
    "object_json": ("name:bvh_output000075,position:(0.0,0.0,0.0),rotation:(0.0,0.0,0.0,1.0),children:[name:Pelvis,position:(0.0,-0.8,0.4),rotation:(0.0,0.0,0.0,1.0),children:[name:Left_hip,position:(0.0,0.0,0.0),rotation:(0.0,0.0,0.0,1.0),children:[name:Left_knee,position:(0.0,0.0,0.0),rotation:(0.0,0.0,0.1,1.0),children:[name:Left_ankle,position:(0.0,0.0,0.0),rotation:(-0.1,0.0,0.0,1.0),children:[name:Left_foot,position:(0.0,0.0,0.0),rotation:(0.0,0.0,0.0,1.0)]]],name:Right_hip,position:(0.0,0.0,0.0),rotation:(0.0,0.0,0.0,1.0),children:[name:Right_knee,position:(0.0,0.0,0.0),rotation:(0.0,0.0,0.0,1.0),children:[name:Right_ankle,position:(0.0,0.0,0.0),rotation:(-0.2,0.0,0.0,1.0),children:[name:Right_foot,position:(0.0,0.0,0.0),rotation:(0.0,0.0,0.0,1.0)]]],name:Spine1,position:(0.0,0.0,0.0),rotation:(0.0,0.0,0.0,1.0),children:[name:Spine2,position:(0.0,0.0,0.0),rotation:(0.0,0.0,0.0,1.0),children:[name:Spine3,position:(0.0,0.0,0.0),rotation:(0.0,0.0,0.0,1.0),children:[name:Neck,position:(0.0,0.0,0.0),rotation:(0.0,0.0,0.0,1.0),children:[name:Head,position:(0.0,0.0,0.0),rotation:(0.1,0.0,0.0,1.0)],name:Left_collar,position:(0.0,0.0,0.0),rotation:(0.0,0.0,-0.3,0.9),children:[name:Left_shoulder,position:(0.0,0.0,0.0),rotation:(0.0,0.1,-0.4,0.9),children:[name:Left_elbow,position:(0.0,0.0,0.0),rotation:(0.3,0.0,0.1,1.0),children:[name:Left_wrist,position:(0.0,0.0,0.0),rotation:(0.0,0.2,0.0,1.0),children:[name:Left_palm,position:(0.0,0.0,0.0),rotation:(0.0,0.0,0.0,1.0)]]]],name:Right_collar,position:(0.0,0.0,0.0),rotation:(0.0,0.0,0.4,0.9),children:[name:Right_shoulder,position:(0.0,0.0,0.0),rotation:(0.0,-0.1,0.3,0.9),children:[name:Right_elbow,position:(0.0,0.0,0.0),rotation:(0.2,0.0,-0.1,1.0),children:[name:Right_wrist,position:(0.0,0.0,0.0),rotation:(0.0,-0.2,0.0,1.0),children:[name:Right_palm,position:(0.0,0.0,0.0),rotation:(0.0,0.0,0.0,1.0)]]]]]]]]]"
                    "Root forward direction: (0.0, 0.5, -0.9); right direction: (1.0, 0.0, 0.0); up direction: (0.0, 0.9, 0.5)"),
    "user_prompt": "the person is standing relaxed, walks forward then turns around on their left foot, and walks back to their original position.",
    "plan": (
        "Step 1: Move bvh_output000100 +X 0 units +Y 0 units +Z 0 units; rotate Pelvis -X 3 degrees +Y 6 degrees -Z 2 degrees; rotate Spine1 -X 3 degrees -Z 4 degrees; rotate Left_hip +X 0 degrees; rotate Right_hip +X 0 degrees; rotate Left_knee +X 0 degrees; rotate Right_knee +X 0 degrees; rotate Left_ankle -X 6 degrees; rotate Right_ankle -X 6 degrees; rotate Left_shoulder -Z 15 degrees; rotate Right_shoulder +Z 15 degrees;"
        "Step 2: Move bvh_output000100 +X 6 units -Z 18 units; rotate Pelvis -X 1 degrees -Y 5 degrees +Z 4 degrees; rotate Spine1 +X 2 degrees -Y 3 degrees +Z 3 degrees; rotate Left_hip -X 20 degrees -Z 5 degrees; rotate Right_hip +X 25 degrees +Z 5 degrees; rotate Left_knee +X 30 degrees; rotate Right_knee -X 10 degrees; rotate Left_ankle -X 10 degrees; rotate Right_ankle +X 5 degrees; rotate Left_shoulder -X 20 degrees +Z 5 degrees; rotate Right_shoulder +X 18 degrees -Z 5 degrees; rotate Left_elbow +X 15 degrees; rotate Right_elbow -X 12 degrees;"
        "Step 3: Move bvh_output000100 -X 6 units -Z 23 units; rotate Pelvis -X 1 degrees -Y 1 degrees -Z 1 degrees; rotate Spine1 -X 1 degrees +Y 1 degrees -Z 1 degrees; rotate Left_hip +X 35 degrees +Z 8 degrees; rotate Right_hip -X 30 degrees -Z 8 degrees; rotate Left_knee -X 15 degrees; rotate Right_knee +X 40 degrees; rotate Left_ankle +X 5 degrees; rotate Right_ankle -X 10 degrees; rotate Left_shoulder +X 30 degrees -Z 10 degrees; rotate Right_shoulder -X 25 degrees +Z 10 degrees; rotate Left_elbow -X 20 degrees; rotate Right_elbow +X 18 degrees;"
        "Step 4: Move bvh_output000100 -X 5 units -Z 46 units; rotate Pelvis -X 2 degrees +Y 10 degrees -Z 4 degrees; rotate Spine1 -X 2 degrees +Y 8 degrees -Z 3 degrees; rotate Left_hip -X 25 degrees -Z 5 degrees; rotate Right_hip +X 20 degrees +Z 5 degrees; rotate Left_knee +X 35 degrees; rotate Right_knee -X 15 degrees; rotate Left_ankle -X 15 degrees; rotate Right_ankle +X 8 degrees; rotate Left_shoulder -X 22 degrees +Z 5 degrees; rotate Right_shoulder +X 18 degrees -Z 5 degrees; rotate Left_elbow +X 18 degrees; rotate Right_elbow -X 15 degrees;"
        "Step 5: Move bvh_output000100 +X 6 units -Z 21 units; rotate Pelvis -X 3 degrees -Y 23 degrees +Z 9 degrees; rotate Spine1 -X 3 degrees -Y 15 degrees +Z 6 degrees; rotate Left_hip +X 10 degrees +Z 3 degrees; rotate Right_hip -X 5 degrees -Z 3 degrees; rotate Left_knee +X 5 degrees; rotate Right_knee +X 5 degrees; rotate Left_ankle -X 5 degrees; rotate Right_ankle -X 5 degrees; rotate Left_shoulder +X 5 degrees -Z 5 degrees; rotate Right_shoulder -X 5 degrees +Z 5 degrees; rotate Left_elbow -X 5 degrees; rotate Right_elbow +X 5 degrees;"
        "Step 6: Move bvh_output000100 +X 6 units -Z 10 units; rotate Pelvis +X 130 degrees -Y 16 degrees +Z 176 degrees; rotate Spine1 +X 125 degrees -Y 12 degrees +Z 170 degrees; rotate Left_hip +X 15 degrees -Z 5 degrees; rotate Right_hip -X 10 degrees +Z 5 degrees; rotate Left_knee +X 10 degrees; rotate Right_knee +X 5 degrees; rotate Left_ankle -X 8 degrees; rotate Right_ankle -X 5 degrees; rotate Left_shoulder +X 10 degrees +Z 5 degrees; rotate Right_shoulder -X 8 degrees -Z 5 degrees;"
        "Step 7: Move bvh_output000100 +X 6 units +Z 14 units; rotate Pelvis -X 1 degrees +Y 12 degrees +Z 5 degrees; rotate Spine1 -X 1 degrees +Y 8 degrees +Z 3 degrees; rotate Left_hip -X 20 degrees +Z 5 degrees; rotate Right_hip +X 25 degrees -Z 5 degrees; rotate Left_knee +X 30 degrees; rotate Right_knee -X 10 degrees; rotate Left_ankle -X 10 degrees; rotate Right_ankle +X 5 degrees; rotate Left_shoulder -X 18 degrees -Z 5 degrees; rotate Right_shoulder +X 15 degrees +Z 5 degrees; rotate Left_elbow +X 14 degrees; rotate Right_elbow -X 12 degrees;"
        "Step 8: Move bvh_output000100 -X 8 units +Z 25 units; rotate Pelvis -X 1 degrees -Y 1 degrees -Z 1 degrees; rotate Spine1 -X 1 degrees +Y 1 degrees -Z 1 degrees; rotate Left_hip +X 32 degrees -Z 8 degrees; rotate Right_hip -X 28 degrees +Z 8 degrees; rotate Left_knee -X 15 degrees; rotate Right_knee +X 38 degrees; rotate Left_ankle +X 5 degrees; rotate Right_ankle -X 10 degrees; rotate Left_shoulder +X 28 degrees +Z 8 degrees; rotate Right_shoulder -X 22 degrees -Z 8 degrees; rotate Left_elbow -X 18 degrees; rotate Right_elbow +X 16 degrees;"
        "Step 9: Move bvh_output000100 -X 8 units +Z 25 units; rotate Pelvis -X 1 degrees +Y 4 degrees -Z 1 degrees; rotate Spine1 -X 1 degrees +Y 3 degrees -Z 1 degrees; rotate Left_hip -X 22 degrees +Z 5 degrees; rotate Right_hip +X 18 degrees -Z 5 degrees; rotate Left_knee +X 28 degrees; rotate Right_knee -X 12 degrees; rotate Left_ankle -X 12 degrees; rotate Right_ankle +X 6 degrees; rotate Left_shoulder -X 20 degrees -Z 5 degrees; rotate Right_shoulder +X 16 degrees +Z 5 degrees; rotate Left_elbow +X 16 degrees; rotate Right_elbow -X 14 degrees;"
        "Step 10: Move bvh_output000100 +X 5 units +Z 19 units; rotate Pelvis -X 1 degrees +Y 4 degrees +Z 2 degrees; rotate Spine1 -X 1 degrees +Y 2 degrees +Z 1 degrees; rotate Left_hip +X 5 degrees -Z 3 degrees; rotate Right_hip -X 5 degrees +Z 3 degrees; rotate Left_knee -X 10 degrees; rotate Right_knee -X 15 degrees; rotate Left_ankle +X 3 degrees; rotate Right_ankle +X 3 degrees; rotate Left_shoulder +X 5 degrees +Z 3 degrees; rotate Right_shoulder -X 5 degrees -Z 3 degrees; rotate Left_elbow -X 8 degrees; rotate Right_elbow +X 8 degrees;"
    )
    }
]

FREE_MODELS = [PLANNER_MODEL]

def get_llm(model: str):
    # return init_chat_model(
    #     model=model,
    #     model_provider="openai",
    #     base_url="https://openrouter.ai/api/v1",
    #     api_key=getenv("OPENROUTER_API_KEY"),
    #     temperature=0,
    # )
    # return init_chat_model(
    #             model="gpt-5-mini",
    #             model_provider="openai",
    #             base_url="http://localhost:4000/v1/",
    #             api_key="nothing",
    #             temperature=0.6,
    #             max_tokens=1200, 
    #             default_headers={
    #                 # "HTTP-Referer": getenv("YOUR_SITE_URL"),
    #                 # "X-OpenRouter-Title": getenv("YOUR_SITE_NAME"),
    #             },
    #         )
      from langchain_google_genai import ChatGoogleGenerativeAI

      api_key = getenv("GOOGLE_API_KEY")
      if not api_key:
          raise RuntimeError("Missing GOOGLE_API_KEY environment variable.")

      return ChatGoogleGenerativeAI(
                  model=model or PLANNER_MODEL,
                  google_api_key=api_key,
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
        "The user's final refined request is: {user_prompt}. "
        "Clarification history, provided only as supporting context because the final refined request is authoritative: "
        "{clarification_history}."
    )),
])


class QuaternionConverter:
    _STEP_PATTERN = re.compile(
        r"Step\s+(\d+)\s*:\s*(.*?)(?=Step\s+\d+\s*:|\Z)",
        re.IGNORECASE | re.DOTALL,
    )
    _ROTATE_PATTERN = re.compile(
        r"^(?:rotate|Rotate)\s+(.+?)\s+"
        r"([+-][XYZ])\s+"
        r"([-+]?\d*\.?\d+(?:[eE][-+]?\d+)?)\s+degrees$",
    )
    _MOVE_PATTERN = re.compile(
        r"^(?:move|Move)\s+.+?\s+"
        r"([+-][XYZ])\s+"
        r"([-+]?\d*\.?\d+(?:[eE][-+]?\d+)?)\s+units$",
    )

    def __init__(
        self,
        forward_axis: tuple[float, float, float] = (0.0, 0.0, 1.0),
        right_axis: tuple[float, float, float] = (1.0, 0.0, 0.0),
        up_axis: tuple[float, float, float] = (0.0, 1.0, 0.0),
    ):
        self.forward_axis = self._normalize_axis(forward_axis, "forward_axis")
        self.right_axis = self._normalize_axis(right_axis, "right_axis")
        self.up_axis = self._normalize_axis(up_axis, "up_axis")
        self.current_quaternions: dict[str, tuple[float, float, float, float]] = {}

    def _normalize_axis(
        self,
        axis: tuple[float, float, float],
        axis_name: str,
    ) -> tuple[float, float, float]:
        if len(axis) != 3:
            raise ValueError(f"{axis_name} must contain exactly 3 values.")

        x, y, z = (float(value) for value in axis)
        magnitude = math.sqrt((x * x) + (y * y) + (z * z))
        if magnitude == 0.0:
            raise ValueError(f"{axis_name} must be a non-zero vector.")

        return (x / magnitude, y / magnitude, z / magnitude)

    def parse_plan(self, text: str) -> list[dict]:
        if not isinstance(text, str) or not text.strip():
            raise ValueError("Plan text must be a non-empty string.")

        steps = []
        for match in self._STEP_PATTERN.finditer(text.strip()):
            step_number = int(match.group(1))
            action_block = match.group(2).strip()
            if not action_block:
                raise ValueError(f"Step {step_number} is malformed.")

            actions = []
            for raw_action in action_block.split(";"):
                action = raw_action.strip()
                if not action:
                    continue

                rotate_match = self._ROTATE_PATTERN.fullmatch(action)
                if rotate_match:
                    joint_name = rotate_match.group(1).strip()
                    direction = rotate_match.group(2)
                    degrees = float(rotate_match.group(3))
                    actions.append(
                        {
                            "joint_name": joint_name,
                            "direction": direction,
                            "degrees": degrees,
                        }
                    )
                    continue

                if self._MOVE_PATTERN.fullmatch(action):
                    continue

                raise ValueError(f"Malformed action in Step {step_number}: {action}")

            steps.append({"step_number": step_number, "actions": actions})

        if not steps:
            raise ValueError("No valid steps were found in the plan text.")

        return steps

    def direction_to_axis(self, direction: str) -> tuple[float, float, float]:
        if not re.fullmatch(r"[+-][XYZ]", direction):
            raise ValueError(f"Invalid direction: {direction}")

        axis_map = {
            "X": self.right_axis,
            "Y": self.up_axis,
            "Z": self.forward_axis,
        }
        sign = 1 if direction[0] == "+" else -1
        axis_letter = direction[1]
        base_axis = axis_map[axis_letter]
        return tuple(sign * value for value in base_axis)

    def axis_angle_to_quaternion(
        self,
        axis: tuple[float, float, float],
        angle_rad: float,
    ) -> tuple[float, float, float, float]:
        x, y, z = self._normalize_axis(axis, "axis")
        half_angle = angle_rad / 2.0
        sin_half = math.sin(half_angle)
        cos_half = math.cos(half_angle)
        return (x * sin_half, y * sin_half, z * sin_half, cos_half)

    def multiply_quaternions(
        self,
        q1: tuple[float, float, float, float],
        q2: tuple[float, float, float, float],
    ) -> tuple[float, float, float, float]:
        x1, y1, z1, w1 = q1
        x2, y2, z2, w2 = q2

        return (
            (w1 * x2) + (x1 * w2) + (y1 * z2) - (z1 * y2),
            (w1 * y2) - (x1 * z2) + (y1 * w2) + (z1 * x2),
            (w1 * z2) + (x1 * y2) - (y1 * x2) + (z1 * w2),
            (w1 * w2) - (x1 * x2) - (y1 * y2) - (z1 * z2),
        )

    def normalize_quaternion(
        self,
        q: tuple[float, float, float, float],
    ) -> tuple[float, float, float, float]:
        x, y, z, w = q
        magnitude = math.sqrt((x * x) + (y * y) + (z * z) + (w * w))
        if magnitude == 0.0:
            raise ValueError("Quaternion magnitude cannot be zero.")
        return (x / magnitude, y / magnitude, z / magnitude, w / magnitude)

    def apply_step(self, step: dict) -> list[tuple[str, tuple[float, float, float, float]]]:
        if "step_number" not in step or "actions" not in step:
            raise ValueError("Malformed step structure.")

        applied_rotations = []
        for action in step["actions"]:
            joint_name = action["joint_name"]
            direction = action["direction"]
            degrees = action["degrees"]

            axis = self.direction_to_axis(direction)
            angle_rad = math.radians(degrees)
            q_new = self.axis_angle_to_quaternion(axis, angle_rad)
            q_previous = self.current_quaternions.get(joint_name, (0.0, 0.0, 0.0, 1.0))
            q_total = self.multiply_quaternions(q_new, q_previous)
            q_total = self.normalize_quaternion(q_total)

            self.current_quaternions[joint_name] = q_total
            applied_rotations.append((joint_name, q_total))

        return applied_rotations

    def process(self, plan_text: str) -> str:
        self.current_quaternions = {}
        steps = self.parse_plan(plan_text)
        output_lines = []

        for step in steps:
            output_lines.append(f"Step {step['step_number']}:")
            for joint_name, quaternion in self.apply_step(step):
                qx, qy, qz, qw = quaternion
                output_lines.append(
                    f"{joint_name}, ({qx:.2f}, {qy:.2f}, {qz:.2f}, {qw:.2f})"
                )

        return "\n".join(output_lines)

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


def run_llm(object_name, object_json, user_prompt, clarification_history=None):
    planner = PlannerAgent()
    return planner.invoke_chain({
        "object": object_name,
        "object_json": object_json,
        "user_prompt": user_prompt,
        "clarification_history": _format_clarification_history(clarification_history),
    })


def run_pipeline(object_name, object_json, user_prompt, converter=None):
    plan_text = run_llm(object_name, object_json, user_prompt)
    converter = converter or QuaternionConverter()

    try:
        quaternion_output = converter.process(plan_text)
    except ValueError as error:
        raise ValueError(f"Failed to convert generated plan to quaternions: {error}") from error

    return plan_text, quaternion_output


def _format_clarification_history(clarification_history) -> str:
    if not clarification_history:
        return "None."

    if isinstance(clarification_history, str):
        text = clarification_history.strip()
        return text or "None."

    lines = []
    for index, item in enumerate(clarification_history, start=1):
        if isinstance(item, dict):
            question = str(item.get("question") or "").strip()
            answer = str(item.get("answer") or "").strip()
        else:
            question = str(item).strip()
            answer = ""

        if question and answer:
            lines.append(f"{index}. Q: {question} A: {answer}")
        elif question:
            lines.append(f"{index}. {question}")

    return "\n".join(lines) if lines else "None."


if __name__ == "__main__":
    planner = PlannerAgent()
    whale_object_json = planner.get_example_object_json("whale")
    whale_plan, whale_quaternions = run_pipeline(
        "whale",
        whale_object_json,
        "Create tilt tail animation",
    )

    print("Raw plan:")
    print(whale_plan)
    print("\nQuaternion output:")
    print(whale_quaternions)
