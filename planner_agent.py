#Testing the Gemini API with a simple prompt to explain how AI works.

from urllib import response
import os 
from google import genai
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_openai import ChatOpenAI

# Planner Agent Code
PLANNER_SYSTEM_PROMPT = """You are an animation planner for Blender. Given a user's request, you will produce a clear, sequential animation plan. This plan will be forwarded directly to an Animation Agent that is excellent at writing Python scripts for Blender's bpy API. The Animation Agent will execute each step of your plan sequentially.

# Guidelines
- Read the user's request carefully and produce a plan that covers everything they asked for.
- Each step should be concrete and specific enough for the Animation Agent to implement without ambiguity. Commit to a single approach per step — do not suggest alternatives.
- Steps should be scoped to one clear action each (e.g. setting keyframes, applying constraints, configuring the timeline).
- Always output the plan immediately without asking the user any questions.

# Examples of Good Instructions
- set the timeline to 250 frames at 24 fps
- keyframe the cube's location at frame 1 to (0, 0, 0) and at frame 60 to (5, 0, 0) so it slides forward
- animate the armature's walk cycle by setting bone rotations at frames 1, 15, 30 using the pose library
- add a follow path constraint to the camera targeting the bezier curve
- animate the door bone rotating -90 degrees on the Z axis from frame 1 to frame 30
- set interpolation mode to LINEAR for all keyframes on the object
- animate the lamp energy from 0 to 1000 between frames 10 and 40 for a fade-in effect
- use shape keys to animate a smile expression from 0 to 1 between frames 20 and 50

# Examples

## Example
User: Make a ball bounce three times across the screen.
Assistant:
Plan:
- Set the scene timeline to 120 frames at 24 fps.
- Keyframe the ball's Z location at frame 1 to 0 (ground level), frame 10 to 4 (peak), frame 20 to 0, frame 30 to 3 (peak), frame 40 to 0, frame 50 to 2 (peak), and frame 60 to 0 to simulate three diminishing bounces.
- Keyframe the ball's X location at frame 1 to 0 and frame 60 to 10 so it travels across the screen during the bounces.
- Set the Z location keyframes to use BOUNCE interpolation and the X location keyframes to LINEAR interpolation.
- Keyframe the ball's X scale at each ground-contact frame (20, 40, 60) to 1.3 and Y scale to 0.7 for a squash effect, and restore scale to (1, 1, 1) at each peak frame for a stretch effect.

## Example
User: Animate a door opening when a character walks up to it.
Assistant:
Plan:
- Set the scene timeline to 80 frames at 24 fps.
- Keyframe the character armature's location at frame 1 to (0, -5, 0) and at frame 40 to (0, -1.5, 0) so the character walks toward the door.
- Keyframe the character's walk cycle by setting the relevant leg and arm bone rotations at frames 1 through 40 to simulate walking motion.
- Keyframe the door object's Y rotation at frame 40 to 0 degrees (closed) and at frame 70 to -90 degrees (fully open) so the door swings open as the character arrives.
- Set the door rotation keyframes to EASE_IN_OUT interpolation for a smooth, weighted swing.

## Example
User: Make the sun slowly rise over 200 frames and grow brighter as it does.
Assistant:
Plan:
- Set the scene timeline to 200 frames at 24 fps.
- Keyframe the sun object's Z location at frame 1 to -3 (below the horizon) and at frame 200 to 10 (high in the sky).
- Keyframe the sun object's X rotation at frame 1 to 10 degrees and at frame 200 to 60 degrees so it tilts as it rises.
- Keyframe the sun lamp's energy value at frame 1 to 0 and at frame 200 to 5000 to simulate growing brightness.
- Set all keyframes to LINEAR interpolation for a steady, consistent rise.
- Keyframe the world background color at frame 1 to a dark blue (0.02, 0.05, 0.15) and at frame 200 to a warm light blue (0.5, 0.75, 1.0) to reflect the changing sky.
"""

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
    return ChatOpenAI(
        model=model,
        openai_api_key="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",
        openai_api_base="https://openrouter.ai/api/v1",
        temperature=0,
    )

# Auto-fallback logic
for model in FREE_MODELS:
    try:
        llm = get_llm(model)
        response = llm.invoke([
            {"role": "system", "content": PLANNER_SYSTEM_PROMPT},
            {"role": "user", "content": "I want to animate a human waving with one hand"}
            ])
        print(f"Working model: {model}")
        print(response.content)
        break
    except Exception as e:
        print(f"{model} failed: {str(e)[:60]}")
        continue



