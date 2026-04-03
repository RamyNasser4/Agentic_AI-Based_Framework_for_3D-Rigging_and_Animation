from langchain.chat_models import init_chat_model
from langchain_core.prompts import ChatPromptTemplate, FewShotChatMessagePromptTemplate
from langchain_core.output_parsers import StrOutputParser
from os import getenv
from dotenv import load_dotenv
import re

load_dotenv()

SYSTEM_MESSAGE = ( \
    "You're an animator who will be provided the joints on a rigged 3D model, and you have to rotate them to produce the requested animation. " \
    "The joints will be given as a JSON string that outlines the object hierarchy. " \
    "You need to output one line of string each time. I will give you the starting point of each joint, which specifies the joint name and its initial position/rotation. " \
    "You need to generate a line to fill out a time series for that joint then a new line for the next joint." \
    "- If a line contains \"[]\", it specifies the root motion for the animation. " \
    "Each vector in the format of [t,x,y,z] specifies a key frame. " \
    "\"t\" is the time stamp for the key, and \"x\", \"y\", \"z\" give the x,y,z components for the position of the object root. " \
    "For example, Armature,[0.0,0.0,0.0,5.9],[1.3,1.0,2.4,5.9] means that the root \"Armature\" has position [0.0,0.0,5.9] at time 0.0, and position [1.0,2.4,5.9] at time 1.3. " \
    "- If a line contains vectors enclosed in \"()\", it represents the time series for the quaternions. " \
    "Each vector in the format (t,x,y,z,w) specifies a key frame for the animation. " \
    "\"t\" is the time stamp for the key, and \"x\", \"y\", \"z\", \"w\" give the x,y,z,w components for the rotation quaternion, respectively. " \
    "Each vector should contain 4 values if it's enclosed in a [] and 5 values if it's enclosed in ()" \
    "For example, Armature/Root/Head,(0.0,0.7,0.0,0.0,0.7),(1.3,0.6,0.0,0.0,0.8) means that the joint \"Armature/Root/Head\" has rotation (0.7,0.0,0.0,0.7) at time 0.0, and rotation (0.6,0.0,0.0,0.8) at time 1.3. " \
    "You will be given the start time of the animation. " \
    "If no previous animation is provided, start the new animation at 0.0s. " \
    "If a previous animation is provided, start the new animation at the provided start time, which is the end time of the previous animation, and continue smoothly from the last keyframes of that previous animation. " \
    "When generating motion that includes translation, align the root and any body-facing rotations with the translation direction so the character or object faces where it is moving. If the translation direction changes, update rotation smoothly to follow that new direction and avoid rotations that contradict the path of travel. " \
    "When an animated element has both translation keyframes in [] and rotation keyframes in (), the rotation keyframe timestamps must match the translation keyframe timestamps exactly so the motion stays synchronized across position and rotation. " \
    "Keep the animation data formatting unchanged." \
    "# Example: The object you will animate is a **whale**. " \
    "Object JSON: name:Armature,position:(0.0000,0.0000,0.0000),rotation:(-0.7,0.0,0.0,0.7),children:[name:Root,position:(0.0000,0.0168,0.0141),rotation:(0.7,0.0,0.0,0.7),children:[name:Head,position:(0.0000,0.0062,0.0198),rotation:(0.7,0.0,0.0,0.7),children:[name:Head_end,position:(0.0000,0.0107,0.0000),rotation:(0.0,0.0,0.0,1.0)]," \
    "name:Spine1,position:(0.0000,0.0050,0.0154),rotation:(-0.7,0.0,0.0,0.7),children:[name:Spine2,position:(0.0000,0.0156,0.0000),rotation:(0.0,0.0,0.0,1.0),children:[name:Spine3,position:(0.0000,0.0166,0.0000),rotation:(0.0,0.0,0.0,1.0),children:[name:Spine4,position:(0.0000,0.0172,0.0000),rotation:(-0.1,0.0,0.0,1.0)," \
    "children:[name:Tail,position:(0.0000,0.0196,0.0000),rotation:(0.0,0.0,0.0,1.0),children:[name:Tail_end,position:(0.0000,0.0133,0.0000),rotation:(0.0,0.0,0.0,1.0)]]]],name:TopFlipper.L,position:(-0.0107,0.0087,-0.0087),rotation:(-0.4,0.0,0.3,0.9),children:[name:MidFlipper.L,position:(0.0000,0.0067,0.0000),rotation:(0.0,0.1,0.0,1.0)," \
    "children:[name:BottomFlipper.L,position:(0.0000,0.0043,0.0000),rotation:(0.0,0.0,-0.1,1.0),children:[name:BottomFlipper.L_end,position:(0.0000,0.0076,0.0000),rotation:(0.0,0.0,0.0,1.0)]]],name:TopFlipper.R,position:(0.0092,0.0078,-0.0084),rotation:(-0.4,0.0,-0.3,0.9),children:[name:MidFlipper.R,position:(0.0000,0.0082,0.0000),rotation:(0.1,-0.1,0.1,1.0)," \
    "children:[name:BottomFlipper.R,position:(0.0000,0.0053,0.0000),rotation:(0.0,0.0,0.2,1.0),children:[name:BottomFlipper.R_end,position:(0.0000,0.0072,0.0000),rotation:(0.0,0.0,0.0,1.0)]]]]]]. " \
    "Root forward direction: (0.00, 1.00, 0.00); right direction: (1.00, 0.00, 0.00); up direction: (0.00, 0.00,-1.00)." \
    "Instruction: create the swim animation for the whale: " \
    "Start time: 0.0s." \
    "Armature,[0.00,0.00,0.00,0.00],[2.79,0.00,0.00,0.00] " \
    "Armature,(0.0,-0.7,0.0,0.0,0.7),(2.8,-0.7,0.0,0.0,0.7) " \
    "Armature/Root,(0.0,0.7,0.0,0.0,0.7),(2.8,0.7,0.0,0.0,0.7) " \
    "Armature/Root/Spine1,(0.0,-0.7,0.0,0.0,0.7),(2.8,-0.7,0.0,0.0,0.7) " \
    "Armature/Root/Spine1/Spine2,(0.0,0.0,0.0,0.0,1.0),(0.6,0.0,0.0,0.0,1.0),(2.6,0.0,0.0,0.0,1.0),(2.8,0.0,0.0,0.0,1.0) " \
    "Armature/Root/Spine1/Spine2/Spine3,(0.0,0.0,0.0,0.0,1.0),(0.9,0.0,0.0,0.0,1.0),(1.5,0.1,0.0,0.0,1.0),(2.5,0.0,0.0,0.0,1.0),(2.8,0.0,0.0,0.0,1.0) " \
    "Armature/Root/Spine1/Spine2/Spine3/Spine4,(0.0,-0.1,0.0,0.0,1.0),(0.8,0.0,0.0,0.0,1.0),(1.5,0.1,0.0,0.0,1.0),(2.3,0.0,0.0,0.0,1.0),(2.8,-0.1,0.0,0.0,1.0) " \
    "Armature/Root/Spine1/Spine2/Spine3/Spine4/Tail,(0.0,0.0,0.0,0.0,1.0),(0.3,0.0,0.0,0.0,1.0),(0.7,-0.1,0.0,0.0,1.0),(1.1,-0.3,0.0,0.0,1.0),(1.7,-0.1,0.0,0.0,1.0),(2.3,0.1,0.0,0.0,1.0),(2.8,0.0,0.0,0.0,1.0) " \
    "Armature/Root/Spine1/TopFlipper.L,(0.0,-0.4,0.0,0.3,0.9),(1.1,-0.4,-0.1,0.2,0.9),(2.3,-0.4,0.0,0.3,0.9),(2.8,-0.4,0.0,0.3,0.9) " \
    "Armature/Root/Spine1/TopFlipper.L/MidFlipper.L,(0.0,0.0,0.1,0.0,1.0),(2.8,0.0,0.1,0.0,1.0) " \
    "Armature/Root/Spine1/TopFlipper.L/MidFlipper.L/BottomFlipper.L,(0.0,0.0,0.0,-0.1,1.0),(0.8,0.0,-0.1,-0.2,1.0),(1.9,0.0,-0.1,-0.2,1.0),(2.8,0.0,0.0,-0.1,1.0) " \
    "Armature/Root/Spine1/TopFlipper.R,(0.0,-0.4,0.0,-0.3,0.9),(1.0,-0.4,0.1,-0.3,0.9),(2.1,-0.4,0.0,-0.3,0.9),(2.8,-0.4,0.0,-0.3,0.9) " \
    "Armature/Root/Spine1/TopFlipper.R/MidFlipper.R,(0.0,0.1,-0.1,0.1,1.0),(2.8,0.1,-0.1,0.1,1.0) " \
    "Armature/Root/Spine1/TopFlipper.R/MidFlipper.R/BottomFlipper.R,(0.0,0.0,0.0,0.2,1.0),(1.1,0.0,0.1,0.3,1.0),(2.3,0.0,0.0,0.2,1.0),(2.8,0.0,0.0,0.2,1.0) " \
    "Armature/Root/Head,(0.0,0.7,0.0,0.0,0.7),(1.3,0.6,0.0,0.0,0.8),(2.6,0.7,0.0,0.0,0.7),(2.8,0.7,0.0,0.0,0.7)" \
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
                    "Root forward direction: (0.00, 1.00, 0.00); right direction: (1.00, 0.00, 0.00); up direction: (0.00, 0.00,-1.00)."
                ),
        "instruction": "idle while moving head up and down",
        "start_time": "0.0s",
        "previous_animation_context": "No previous animation is provided. Start a fresh animation at 0.0s.",
        "animation": (
                        "metarig,[0.00,0.00,0.00,0.00],[1.56,0.00,0.00,0.00]"
                        "metarig/spine/spine.001/spine.002/spine.003/spine"
                        ".006,(0.0,-0.1,0.0,0.0,1.0),(0.5,-0.1,0.0,0.0,1.0)"
                        ",(0.8,-0.1,0.0,0.0,1.0),(1.2,0.0,0.0,0.0,1.0)"
                        ",(1.6,-0.1,0.0,0.0,1.0)"
                        "metarig/spine/spine.001/spine.002/spine.003/spine.006/ear.L"
                        ",(0.0,-0.1,-0.1,0.1,1.0),(0.2,-0.1,0.0,0.0,1.0)"
                        ",(0.4,0.0,-0.1,0.1,1.0),(0.6,0.0,-0.1,0.2,1.0)"
                        ",(1.0,-0.1,-0.1,0.2,1.0),(1.2,0.0,-0.1,0.2,1.0)"
                        ",(1.5,-0.1,-0.1,0.1,1.0),(1.6,-0.1,-0.1,0.1,1.0)"
                        "metarig/spine/spine.001/spine.002/spine.003/spine.006/ear.R"
                        ",(0.0,-0.1,0.0,-0.3,1.0),(0.2,-0.1,0.0,-0.3,0.9)"
                        ",(0.4,0.0,0.1,-0.2,1.0),(0.6,0.0,0.0,-0.2,1.0)"
                        ",(1.0,-0.1,0.0,-0.1,1.0),(1.2,0.0,0.0,-0.2,1.0)"
                        ",(1.5,-0.1,0.1,-0.2,1.0),(1.6,-0.1,0.0,-0.3,1.0)"
                        "metarig/spine/spine.001/spine.002/spine.003/shoulder.L/upper_arm.L"
                        ",(0.0,-0.1,0.7,-0.5,-0.5),(0.5,-0.1,0.8,-0.4,-0.4)"
                        ",(0.9,-0.1,0.9,-0.4,-0.4),(1.3,-0.1,0.7,-0.5,-0.5)"
                        ",(1.6,-0.1,0.7,-0.5,-0.5)"
                        "metarig/spine/spine.001/spine.002/spine.003/shoulder.L/upper_arm.L/"
                        "forearm.L,(0.0,0.1,-0.1,0.0,1.0),(0.4,0.1,-0.1,0.0,1.0)"
                        ",(0.8,-0.1,-0.2,0.0,1.0),(1.2,0.0,-0.2,0.0,1.0)"
                        ",(1.6,0.1,-0.1,0.0,1.0)"
                        "metarig/spine/spine.001/spine.002/spine.003/shoulder.L/upper_arm.L/"
                        "forearm.L/hand.L,(0.0,0.1,0.0,-0.1,1.0),(0.5,0.1,0.0,0.0,1.0)"
                        ",(1.0,0.1,-0.1,0.1,1.0),(1.5,0.1,0.0,0.0,1.0)"
                        ",(1.6,0.1,0.0,-0.1,1.0)"
                        "metarig/spine/tail,(0.0,-0.8,0.0,0.0,0.6),(0.2,-0.8,0.0,0.0,0.6)"
                        ",(0.4,-0.9,-0.2,-0.2,0.5),(0.6,-0.9,-0.2,-0.2,0.4)"
                        ",(1.1,-0.8,-0.1,-0.1,0.5),(1.6,-0.8,0.0,0.0,0.6)"
                        "metarig/spine/tail/tail.001,(0.0,0.1,0.0,0.1,1.0)"
                        ",(0.4,0.1,0.0,0.0,1.0),(0.8,0.0,0.1,-0.2,1.0)"
                        ",(1.2,0.1,0.0,0.0,1.0),(1.6,0.1,0.0,0.1,1.0)"
                        "metarig/spine/tail/tail.001/tail.002,(0.0,0.1,0.0,0.2,1.0)"
                        ",(0.4,0.2,0.0,0.3,0.9),(0.6,0.1,0.0,0.1,1.0)"
                        ",(0.7,0.1,0.0,-0.1,1.0),(1.0,0.0,0.0,-0.4,0.9)"
                        ",(1.3,0.1,0.0,-0.1,1.0),(1.5,0.1,0.0,0.1,1.0)"
                        ",(1.6,0.1,0.0,0.2,1.0)"
                        "metarig/spine/tail/tail.001/tail.002/tail.003,(0.0,0.0,0.1,0.2,1.0)"
                        ",(0.4,0.0,0.0,0.5,0.9),(0.6,0.0,0.0,0.5,0.9),(0.7,0.0,0.0,0.4,0.9)"
                        ",(0.8,0.0,0.0,0.1,1.0),(1.0,0.0,0.0,-0.2,1.0)"
                        ",(1.1,0.0,0.0,-0.4,0.9),(1.3,0.0,0.0,-0.4,0.9)"
                        ",(1.4,0.0,0.0,-0.1,1.0),(1.5,0.0,0.1,0.1,1.0)"
                        ",(1.6,0.0,0.1,0.2,1.0)"
                    )
    },
]

class KeyFrameAgent:
    _TIME_TOKEN_PATTERN = re.compile(r"[\[\(]\s*([-+]?\d*\.?\d+(?:[eE][-+]?\d+)?)")

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
        llm = init_chat_model(
            model="gpt-5-mini",
            model_provider="openai",
            base_url="http://localhost:4000/v1/",
            api_key="nothing",
            default_headers={
                # "HTTP-Referer": getenv("YOUR_SITE_URL"),
                # "X-OpenRouter-Title": getenv("YOUR_SITE_NAME"),
            },
        )
        prompt_template_for_examples = ChatPromptTemplate.from_messages([
            ("human",
                (
                    "The object you will animate is a **{object}**."\
                    "Object JSON: {object_json}."\
                    "Instruction: {instruction}."\
                    "Start time: {start_time}."\
                    "Previous animation context: {previous_animation_context}"
                )
             ),
            ("ai", "{animation}")
        ])
        few_shot_prompt = FewShotChatMessagePromptTemplate(
            examples=animation_examples,
            example_prompt=prompt_template_for_examples
        )
        main_prompt = ChatPromptTemplate.from_messages([
            ("system", SYSTEM_MESSAGE),
            few_shot_prompt,
            ("human",
                (
                    "The object you will animate is a **{object}**."\
                    "Object JSON: {object_json}."\
                    "Instruction: {instruction}."\
                    "Start time: {start_time}."\
                    "Previous animation context: {previous_animation_context}"
                )
             ),
        ])

        self.chain = main_prompt | llm | StrOutputParser()
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

    def _prepare_input(self, input_dict: dict) -> dict:
        payload = dict(input_dict)
        previous_animation = self._normalize_previous_animation(payload.get("previous_animation"))

        if previous_animation:
            start_time = self._extract_end_time(previous_animation)
            previous_animation_context = (
                "A previous animation is provided. Continue smoothly from its last keyframes. "
                f"The new animation must begin at {self._format_start_time(start_time)}. "
                f"Previous animation data:\n{previous_animation}"
            )
        else:
            start_time = 0.0
            previous_animation_context = (
                "No previous animation is provided. Start a fresh animation at 0.0s."
            )

        payload["start_time"] = self._format_start_time(start_time)
        payload["previous_animation_context"] = previous_animation_context
        return payload

    def invoke_chain(self, input_dict: dict) -> str:
        if not hasattr(self, "chain"):
            raise RuntimeError("Chain is not initialized. Call initialize_chain() first.")

        prepared_input = self._prepare_input(input_dict)
        return self.chain.invoke(prepared_input)
if __name__ == "__main__":
    keyframe = KeyFrameAgent()
    keyframe.initialize_chain()
    response = keyframe.invoke_chain(
        {
                    "object": "human male",
                    "object_json": (
                        "name:SMPLX-lh-male,position:(6.7,-0.9,0.0),rotation:(0.0,0.0,0.0,1.0),children:[name:root,position:(0.0,0.0,0.0),rotation:(0.0,0.0,0.0,1.0),children:[name:pelvis,position:(0.0,0.0,0.0),rotation:(0.0,0.0,0.0,1.0),children:[name:left_hip,position:(0.0,0.0,0.0),rotation:(0.0,0.0,0.0,1.0),children:[name:left_knee,position:(0.0,0.0,0.0),rotation:(0.12,0.0,0.0,1.0),children:[name:left_ankle,position:(0.0,0.0,0.0),rotation:(0.0,0.0,0.0,1.0),children:[name:left_foot,position:(0.0,0.0,0.0),rotation:(0.0,0.0,0.0,1.0)]]],name:right_hip,position:(0.0,0.0,0.0),rotation:(0.0,0.0,0.0,1.0),children:[name:right_knee,position:(0.0,0.0,0.0),rotation:(0.12,0.0,0.0,1.0),children:[name:right_ankle,position:(0.0,0.0,0.0),rotation:(0.0,0.0,0.0,1.0),children:[name:right_foot,position:(0.0,0.0,0.0),rotation:(0.0,0.0,0.0,1.0)]]],name:spine1,position:(0.0,0.0,0.0),rotation:(0.0,0.0,0.0,1.0),children:[name:spine2,position:(0.0,0.0,0.0),rotation:(0.0,0.0,0.0,1.0),children:[name:spine3,position:(0.0,0.0,0.0),rotation:(0.0,0.0,0.0,1.0),children:[name:neck,position:(0.0,0.0,0.0),rotation:(0.0,0.0,0.0,1.0),children:[name:head,position:(0.0,0.0,0.0),rotation:(0.0,0.0,0.0,1.0),children:[name:jaw,position:(0.0,0.0,0.0),rotation:(0.0,0.0,0.0,1.0),name:left_eye_smplhf,position:(0.0,0.0,0.0),rotation:(0.0,0.0,0.0,1.0),name:right_eye_smplhf,position:(0.0,0.0,0.0),rotation:(0.0,0.0,0.0,1.0)]],name:left_collar,position:(0.0,0.0,0.0),rotation:(0.0,0.0,0.0,1.0),children:[name:left_shoulder,position:(0.0,0.0,0.0),rotation:(0.0,0.0,0.0,1.0),children:[name:left_elbow,position:(0.0,0.0,0.0),rotation:(0.0,0.0,0.0,1.0),children:[name:left_wrist,position:(0.0,0.0,0.0),rotation:(0.0,0.0,0.0,1.0),children:[name:left_index1,position:(0.0,0.0,0.0),rotation:(0.0,0.0,-0.2,1.0),children:[name:left_index2,position:(0.0,0.0,0.0),rotation:(0.0,0.0,-0.4,0.9),children:[name:left_index3,position:(0.0,0.0,0.0),rotation:(0.0,0.0,0.0,1.0)]],name:left_middle1,position:(0.0,0.0,0.0),rotation:(0.0,0.0,-0.3,1.0),children:[name:left_middle2,position:(0.0,0.0,0.0),rotation:(0.0,0.0,-0.3,0.9),children:[name:left_middle3,position:(0.0,0.0,0.0),rotation:(0.0,0.0,-0.2,1.0)]],name:left_pinky1,position:(0.0,0.0,0.0),rotation:(-0.2,0.0,-0.3,0.9),children:[name:left_pinky2,position:(0.0,0.0,0.0),rotation:(-0.1,0.0,-0.3,1.0),children:[name:left_pinky3,position:(0.0,0.0,0.0),rotation:(-0.2,0.0,0.0,1.0)]],name:left_ring1,position:(0.0,0.0,0.0),rotation:(0.0,0.0,-0.3,0.9),children:[name:left_ring2,position:(0.0,0.0,0.0),rotation:(-0.1,0.0,-0.3,0.9),children:[name:left_ring3,position:(0.0,0.0,0.0),rotation:(0.0,0.0,-0.2,1.0)]],name:left_thumb1,position:(0.0,0.0,0.0),rotation:(0.4,0.1,0.0,0.9),children:[name:left_thumb2,position:(0.0,0.0,0.0),rotation:(-0.2,0.0,0.0,1.0),children:[name:left_thumb3,position:(0.0,0.0,0.0),rotation:(0.3,0.0,-0.1,1.0)]]]]]],name:right_collar,position:(0.0,0.0,0.0),rotation:(0.0,0.0,0.0,1.0),children:[name:right_shoulder,position:(0.0,0.0,0.0),rotation:(0.0,0.0,0.0,1.0),children:[name:right_elbow,position:(0.0,0.0,0.0),rotation:(0.0,0.0,0.0,1.0),children:[name:right_wrist,position:(0.0,0.0,0.0),rotation:(0.0,0.0,0.0,1.0),children:[name:right_index1,position:(0.0,0.0,0.0),rotation:(0.0,0.0,0.2,1.0),children:[name:right_index2,position:(0.0,0.0,0.0),rotation:(0.0,0.0,0.4,0.9),children:[name:right_index3,position:(0.0,0.0,0.0),rotation:(0.0,0.0,0.0,1.0)]],name:right_middle1,position:(0.0,0.0,0.0),rotation:(0.0,0.0,0.3,1.0),children:[name:right_middle2,position:(0.0,0.0,0.0),rotation:(0.0,0.0,0.3,0.9),children:[name:right_middle3,position:(0.0,0.0,0.0),rotation:(0.0,0.0,0.2,1.0)]],name:right_pinky1,position:(0.0,0.0,0.0),rotation:(-0.2,0.0,0.3,0.9),children:[name:right_pinky2,position:(0.0,0.0,0.0),rotation:(-0.1,0.0,0.3,1.0),children:[name:right_pinky3,position:(0.0,0.0,0.0),rotation:(-0.2,0.0,0.0,1.0)]],name:right_ring1,position:(0.0,0.0,0.0),rotation:(0.0,0.0,0.3,0.9),children:[name:right_ring2,position:(0.0,0.0,0.0),rotation:(-0.1,0.0,0.3,0.9),children:[name:right_ring3,position:(0.0,0.0,0.0),rotation:(0.0,0.0,0.2,1.0)]],name:right_thumb1,position:(0.0,0.0,0.0),rotation:(0.4,-0.1,0.0,0.9),children:[name:right_thumb2,position:(0.0,0.0,0.0),rotation:(-0.2,0.0,0.0,1.0),children:[name:right_thumb3,position:(0.0,0.0,0.0),rotation:(0.3,0.0,0.1,1.0)]]]]]]]]]]]]"
                        "Root forward direction: (0.0, 0.0, 1.0); right direction: (-1.0, 0.0, 0.0); up direction: (0.0, 1.0, 0.0)"
                    ),
                    "instruction": "Move root forward along (0, 1, 0); rotate left_hip forward; bend left_knee; lift left_ankle; move left_foot forward; extend right_hip backward; keep right_knee slightly bent; rotate right_shoulder forward; rotate left_shoulder backward.",
                    "previous_animation": """
                    SMPLX-lh-male,[-0.00,6.70,-2.40,0.00],[1.00,6.70,-1.90,0.00],[2.00,6.70,-1.40,0.00],[3.00,6.70,-0.90,0.00]
                    SMPLX-lh-male,(0.0,0.0,0.0,0.0,1.0),(3.0,0.0,0.0,0.0,1.0)
                    SMPLX-lh-male/root,(0.0,0.0,0.0,0.0,1.0),(3.0,0.0,0.0,0.0,1.0)
                    SMPLX-lh-male/root/pelvis,(0.0,0.0,0.0,0.0,1.0),(3.0,0.0,0.0,0.0,1.0)
                    SMPLX-lh-male/root/pelvis/left_knee,(0.0,0.12,0.0,0.0,0.99),(1.5,0.12,0.0,0.0,0.99),(3.0,0.12,0.0,0.0,0.99)
                    SMPLX-lh-male/root/pelvis/right_knee,(0.0,0.12,0.0,0.0,0.99),(1.5,0.12,0.0,0.0,0.99),(3.0,0.12,0.0,0.0,0.99)
                    SMPLX-lh-male/root/pelvis/spine1,(0.0,0.0,0.0,0.0,1.0),(3.0,0.0,0.0,0.0,1.0)
                    SMPLX-lh-male/root/pelvis/spine1/spine2,(0.0,0.0,0.0,0.0,1.0),(3.0,0.0,0.0,0.0,1.0)
                    SMPLX-lh-male/root/pelvis/spine1/spine2/spine3,(0.0,0.0,0.0,0.0,1.0),(3.0,0.0,0.0,0.0,1.0)
                    SMPLX-lh-male/root/pelvis/spine1/spine2/spine3/neck,(0.0,0.0,0.0,0.0,1.0),(3.0,0.0,0.0,0.0,1.0)      
                    SMPLX-lh-male/root/pelvis/spine1/spine2/spine3/neck/head,(0.0,0.0,0.0,0.0,1.0),(3.0,0.0,0.0,0.0,1.0)
                    """,
                }
    )
    # # response = keyframe.invoke_chain(
    # #     {
    # #                 "object": "whale",
    # #                 "object_json": (
    # #                     "name:Armature,position:(0.0000,0.0000,0.0000),rotation:(-0.7,0.0,0.0,0.7),children:[name:Root,position:(0.0000,0.0168,0.0141),rotation:(0.7,0.0,0.0,0.7),children:[name:Head,position:(0.0000,0.0062,0.0198),rotation:(0.7,0.0,0.0,0.7),children:[name:Head_end,position:(0.0000,0.0107,0.0000),rotation:(0.0,0.0,0.0,1.0)],"
    # #                     "name:Spine1,position:(0.0000,0.0050,0.0154),rotation:(-0.7,0.0,0.0,0.7),children:[name:Spine2,position:(0.0000,0.0156,0.0000),rotation:(0.0,0.0,0.0,1.0),children:[name:Spine3,position:(0.0000,0.0166,0.0000),rotation:(0.0,0.0,0.0,1.0),children:[name:Spine4,position:(0.0000,0.0172,0.0000),rotation:(-0.1,0.0,0.0,1.0),"
    # #                     "children:[name:Tail,position:(0.0000,0.0196,0.0000),rotation:(0.0,0.0,0.0,1.0),children:[name:Tail_end,position:(0.0000,0.0133,0.0000),rotation:(0.0,0.0,0.0,1.0)]]]],name:TopFlipper.L,position:(-0.0107,0.0087,-0.0087),rotation:(-0.4,0.0,0.3,0.9),children:[name:MidFlipper.L,position:(0.0000,0.0067,0.0000),rotation:(0.0,0.1,0.0,1.0),"
    # #                     "children:[name:BottomFlipper.L,position:(0.0000,0.0043,0.0000),rotation:(0.0,0.0,-0.1,1.0),children:[name:BottomFlipper.L_end,position:(0.0000,0.0076,0.0000),rotation:(0.0,0.0,0.0,1.0)]]],name:TopFlipper.R,position:(0.0092,0.0078,-0.0084),rotation:(-0.4,0.0,-0.3,0.9),children:[name:MidFlipper.R,position:(0.0000,0.0082,0.0000),rotation:(0.1,-0.1,0.1,1.0),"
    # #                     "children:[name:BottomFlipper.R,position:(0.0000,0.0053,0.0000),rotation:(0.0,0.0,0.2,1.0),children:[name:BottomFlipper.R_end,position:(0.0000,0.0072,0.0000),rotation:(0.0,0.0,0.0,1.0)]]]]]]"
    # #                 ),
    # #                 "instruction": "create the jumping out of the water animation for the whale",
    # #             }
    # # )
    # # print(response)
    # # response = keyframe.invoke_chain(
    # #     {
    # #                 "object": "racoon",
    # #                 "object_json": (
    # #                     "name:metarig,position:(0.00,0.00,0.00),rotation:(-0.7,0.0,0.0,0.7),"
    # #                     "children:[name:spine,position:(0.00,0.00,0.00),rotation"
    # #                     ":(0.7,0.0,0.0,0.7),children:[name:pelvis.L,position"
    # #                     ":(0.00,0.00,0.00),rotation:(-0.2,0.6,0.7,0.4),name:pelvis.R,"
    # #                     "position:(0.00,0.00,0.00),rotation:(0.2,0.6,0.7,-0.4),name:spine"
    # #                     ".001,position:(0.00,0.00,0.00),rotation:(0.0,0.0,0.0,1.0),children"
    # #                     ":[name:spine.002,position:(0.00,0.00,0.00),rotation"
    # #                     ":(0.0,0.0,0.0,1.0),children:[name:spine.003,position"
    # #                     ":(0.00,0.00,0.00),rotation:(-0.1,0.0,0.0,1.0),children:[name:"
    # #                     "breast.L,position:(0.00,0.00,0.00),rotation:(0.0,0.8,0.6,0.0),name"
    # #                     ":breast.R,position:(0.00,0.00,0.00),rotation:(0.0,0.8,0.6,0.0),"
    # #                     "name:shoulder.L,position:(0.00,0.00,0.00),rotation"
    # #                     ":(-0.7,0.2,0.4,0.6),children:[name:upper_arm.L,position"
    # #                     ":(0.00,0.00,0.00),rotation:(0.2,-0.7,0.4,0.5),children:[name:"
    # #                     "forearm.L,position:(0.00,0.00,0.00),rotation:(0.5,0.0,0.0,0.8),"
    # #                     "children:[name:hand.L,position:(0.00,0.00,0.00),rotation"
    # #                     ":(0.1,0.0,-0.1,1.0)]]],name:shoulder.R,position:(0.00,0.00,0.00),"
    # #                     "rotation:(-0.7,-0.2,-0.4,0.6),children:[name:upper_arm.R,position"
    # #                     ":(0.00,0.00,0.00),rotation:(-0.1,0.9,-0.3,0.4),children:[name:"
    # #                     "forearm.R,position:(0.00,0.00,0.00),rotation:(-0.1,0.2,-0.6,0.8),"
    # #                     "children:[name:hand.R,position:(0.00,0.00,0.00),rotation"
    # #                     ":(0.1,0.1,-0.2,1.0),children:[name:hand.R.001,position"
    # #                     ":(0.00,0.00,0.00),rotation:(0.3,0.0,0.7,0.6),children:[name:"
    # #                     "Spork_low,position:(0.00,0.00,0.00),rotation:(0.0,0.7,-0.7,0.1)"
    # #                     "]]]]],name:spine.006,position:(0.00,0.00,0.00),rotation"
    # #                     ":(-0.1,0.0,0.0,1.0),children:[name:ear.L,position:(0.00,0.01,0.00)"
    # #                     ",rotation:(-0.1,-0.1,0.2,1.0),name:ear.R,position:(0.00,0.01,0.00)"
    # #                     ",rotation:(-0.1,0.1,-0.5,0.9)]]]],name:tail,position"
    # #                     ":(0.00,0.00,0.00),rotation:(0.8,0.4,-0.1,-0.3),children:[name:tail"
    # #                     ".001,position:(0.00,0.00,0.00),rotation:(-0.2,0.1,-0.2,1.0),"
    # #                     "children:[name:tail.002,position:(0.00,0.00,0.00),rotation"
    # #                     ":(0.0,0.5,-0.5,0.7),children:[name:tail.003,position"
    # #                     ":(0.00,0.00,0.00),rotation:(-0.5,0.0,0.0,0.8)]]],name:thigh.L,"
    # #                     "position:(0.00,0.00,0.00),rotation:(1.0,0.1,-0.3,0.0),children:["
    # #                     "name:shin.L,position:(0.00,0.00,0.00),rotation:(0.2,0.3,-0.1,0.9),"
    # #                     "children:[name:foot.L,position:(0.00,0.00,0.00),rotation"
    # #                     ":(-0.5,0.1,0.2,0.8),children:[name:heel.02.L,position"
    # #                     ":(0.00,0.00,0.00),rotation:(-0.6,0.6,-0.2,-0.5),name:toe.L,"
    # #                     "position:(0.00,0.00,0.00),rotation:(-0.3,0.8,-0.4,-0.3)]]],name:"
    # #                     "thigh.R,position:(0.00,0.00,0.00),rotation:(1.0,-0.1,0.3,0.0),"
    # #                     "children:[name:shin.R,position:(0.00,0.00,0.00),rotation"
    # #                     ":(0.2,-0.3,0.1,0.9),children:[name:foot.R,position"
    # #                     ":(0.00,0.00,0.00),rotation:(-0.5,-0.1,-0.2,0.8),children:[name:"
    # #                     "heel.02.R,position:(0.00,0.00,0.00),rotation:(0.6,0.6,-0.2,0.5),"
    # #                     "name:toe.R,position:(0.00,0.00,0.00),rotation:(0.3,0.8,-0.4,0.3)"
    # #                     "]]]]]"
    # #                 ),
    # #                 "instruction": "create the walk animation for the racoon: ",
    # #             }
    # # )
    print(response)
