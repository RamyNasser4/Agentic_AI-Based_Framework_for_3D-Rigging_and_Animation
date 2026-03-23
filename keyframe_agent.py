from langchain.chat_models import init_chat_model
from langchain_core.prompts import ChatPromptTemplate, FewShotChatMessagePromptTemplate
from langchain_core.output_parsers import StrOutputParser
from os import getenv
from dotenv import load_dotenv

load_dotenv()

SYSTEM_MESSAGE = ( \
    "You're an animator who will be provided the joints on a rigged 3D model, and you have to rotate them to produce the requested animation. " \
    "The joints will be given as a JSON string that outlines the object hierarchy. " \
    "You need to output one line of string each time. I will give you the starting point of that string, which specifies the joint name and its initial position/rotation. " \
    "You need to complete the line to fill out a time series for that joint. " \
    "- If a line contains \"[]\", it specifies the root motion for the animation. " \
    "Each vector in the format of [t,x,y,z] specifies a key frame. " \
    "\"t\" is the time stamp for the key, and \"x\", \"y\", \"z\" give the x,y,z components for the position of the object root. " \
    "For example, Armature,[0.0,0.0,0.0,5.9],[1.3,1.0,2.4,5.9] means that the root \"Armature\" has position [0.0,0.0,5.9] at time 0.0, and position [1.0,2.4,5.9] at time 1.3. " \
    "- If a line contains vectors enclosed in \"()\", it represents the time series for the quaternions. " \
    "Each vector in the format (t,x,y,z,w) specifies a key frame for the animation. " \
    "\"t\" is the time stamp for the key, and \"x\", \"y\", \"z\", \"w\" give the x,y,z,w components for the rotation quaternion, respectively. " \
    "For example, Armature/Root/Head,(0.0,0.7,0.0,0.0,0.7),(1.3,0.6,0.0,0.0,0.8) means that the joint \"Armature/Root/Head\" has rotation (0.7,0.0,0.0,0.7) at time 0.0, and rotation (0.6,0.0,0.0,0.8) at time 1.3. " \
    "# Example: The object you will animate is a **whale**. " \
    "Object JSON: name:Armature,position:(0.0000,0.0000,0.0000),rotation:(-0.7,0.0,0.0,0.7),children:[name:Root,position:(0.0000,0.0168,0.0141),rotation:(0.7,0.0,0.0,0.7),children:[name:Head,position:(0.0000,0.0062,0.0198),rotation:(0.7,0.0,0.0,0.7),children:[name:Head_end,position:(0.0000,0.0107,0.0000),rotation:(0.0,0.0,0.0,1.0)]," \
    "name:Spine1,position:(0.0000,0.0050,0.0154),rotation:(-0.7,0.0,0.0,0.7),children:[name:Spine2,position:(0.0000,0.0156,0.0000),rotation:(0.0,0.0,0.0,1.0),children:[name:Spine3,position:(0.0000,0.0166,0.0000),rotation:(0.0,0.0,0.0,1.0),children:[name:Spine4,position:(0.0000,0.0172,0.0000),rotation:(-0.1,0.0,0.0,1.0)," \
    "children:[name:Tail,position:(0.0000,0.0196,0.0000),rotation:(0.0,0.0,0.0,1.0),children:[name:Tail_end,position:(0.0000,0.0133,0.0000),rotation:(0.0,0.0,0.0,1.0)]]]],name:TopFlipper.L,position:(-0.0107,0.0087,-0.0087),rotation:(-0.4,0.0,0.3,0.9),children:[name:MidFlipper.L,position:(0.0000,0.0067,0.0000),rotation:(0.0,0.1,0.0,1.0)," \
    "children:[name:BottomFlipper.L,position:(0.0000,0.0043,0.0000),rotation:(0.0,0.0,-0.1,1.0),children:[name:BottomFlipper.L_end,position:(0.0000,0.0076,0.0000),rotation:(0.0,0.0,0.0,1.0)]]],name:TopFlipper.R,position:(0.0092,0.0078,-0.0084),rotation:(-0.4,0.0,-0.3,0.9),children:[name:MidFlipper.R,position:(0.0000,0.0082,0.0000),rotation:(0.1,-0.1,0.1,1.0)," \
    "children:[name:BottomFlipper.R,position:(0.0000,0.0053,0.0000),rotation:(0.0,0.0,0.2,1.0),children:[name:BottomFlipper.R_end,position:(0.0000,0.0072,0.0000),rotation:(0.0,0.0,0.0,1.0)]]]]]]. " \
    "Root forward direction: (0.00, 1.00, 0.00); right direction: (1.00, 0.00, 0.00); up direction: (0.00, 0.00,-1.00)." \
    "Instruction: create the swim animation for the whale: " \
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
                ),
        "instruction": "idle while moving head up and down",
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
                    "Instruction: {instruction}"
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
                    "Instruction: {instruction}"
                )
             ),
        ])

        self.chain = main_prompt | llm | StrOutputParser()
        return self.chain
    def invoke_chain(self, input_dict: dict) -> str:
        if not hasattr(self, "chain"):
            raise RuntimeError("Chain is not initialized. Call initialize_chain() first.")

        return self.chain.invoke(input_dict)

keyframe = KeyFrameAgent()
keyframe.initialize_chain()
response = keyframe.invoke_chain(
    {
                "object": "human male",
                "object_json": (
                    "name:SMPLX-lh-male,position:(-3.8392,0.0000,0.0000),rotation:(1.0000,0.0000,0.0000,0.0000),children:[name:root,position:(0.0000,0.0000,0.0000),rotation:(0.7071,0.7071,0.0000,0.0000),children:[name:pelvis,position:(0.0012,-0.3668,0.0127),rotation:(1.0000,0.0000,0.0000,0.0000),children:[name:left_hip,position:(0.0561,-0.0945,-0.0235),rotation:(1.0000,0.0000,0.0000,0.0000),children:[name:left_knee,position:(0.0672,-0.3969,-0.0067),rotation:(1.0000,0.0000,0.0000,0.0000),children:[name:left_ankle,position:(-0.0456,-0.4213,-0.0411),rotation:(1.0000,0.0000,0.0000,0.0000),children:[name:left_foot,position:(0.0443,-0.0607,0.1352),rotation:(1.0000,0.0000,0.0000,0.0000)]]],name:right_hip,position:(-0.0579,-0.1052,-0.0166),rotation:(1.0000,0.0000,0.0000,0.0000),children:[name:right_knee,position:(-0.0507,-0.3798,-0.0145),rotation:(1.0000,0.0000,0.0000,0.0000),children:[name:right_ankle,position:(0.0172,-0.4349,-0.0399),rotation:(1.0000,0.0000,0.0000,0.0000),children:[name:right_foot,position:(-0.0354,-0.0597,0.1418),rotation:(1.0000,0.0000,0.0000,0.0000)]]],name:spine1,position:(-0.0013,0.1104,-0.0379),rotation:(1.0000,0.0000,0.0000,0.0000),children:[name:spine2,position:(0.0102,0.1510,0.0044),rotation:(1.0000,0.0000,0.0000,0.0000),children:[name:spine3,position:(-0.0090,0.0578,0.0227),rotation:(1.0000,0.0000,0.0000,0.0000),children:[name:neck,position:(-0.0096,0.1660,-0.0272),rotation:(1.0000,0.0000,0.0000,0.0000),children:[name:head,position:(0.0230,0.1609,0.0230),rotation:(1.0000,0.0000,0.0000,0.0000),children:[name:jaw,position:(-0.0209,-0.0009,-0.0062),rotation:(1.0000,0.0000,0.0000,0.0000),name:left_eye_smplhf,position:(0.0180,0.0452,0.0658),rotation:(1.0000,0.0000,0.0000,0.0000),name:right_eye_smplhf,position:(-0.0470,0.0452,0.0658),rotation:(1.0000,0.0000,0.0000,0.0000)]],name:left_collar,position:(0.0424,0.0763,-0.0053),rotation:(1.0000,0.0000,0.0000,0.0000),children:[name:left_shoulder,position:(0.1409,0.0604,-0.0149),rotation:(1.0000,0.0000,0.0000,0.0000),children:[name:left_elbow,position:(0.2547,-0.0762,-0.0455),rotation:(1.0000,0.0000,0.0000,0.0000),children:[name:left_wrist,position:(0.2716,0.0256,-0.0006),rotation:(1.0000,0.0000,0.0000,0.0000),children:[name:left_index1,position:(0.1061,-0.0086,0.0211),rotation:(1.0000,0.0000,0.0000,0.0000),children:[name:left_index2,position:(0.0333,0.0016,0.0037),rotation:(1.0000,0.0000,0.0000,0.0000),children:[name:left_index3,position:(0.0235,-0.0027,-0.0002),rotation:(1.0000,0.0000,0.0000,0.0000)]],name:left_middle1,position:(0.1142,-0.0059,-0.0040),rotation:(1.0000,0.0000,0.0000,0.0000),children:[name:left_middle2,position:(0.0319,0.0004,-0.0045),rotation:(1.0000,0.0000,0.0000,0.0000),children:[name:left_middle3,position:(0.0244,-0.0025,-0.0044),rotation:(1.0000,0.0000,0.0000,0.0000)]],name:left_pinky1,position:(0.0881,-0.0147,-0.0465),rotation:(1.0000,0.0000,0.0000,0.0000),children:[name:left_pinky2,position:(0.0170,-0.0017,-0.0120),rotation:(1.0000,0.0000,0.0000,0.0000),children:[name:left_pinky3,position:(0.0167,-0.0020,-0.0112),rotation:(1.0000,0.0000,0.0000,0.0000)]],name:left_ring1,position:(0.1019,-0.0089,-0.0289),rotation:(1.0000,0.0000,0.0000,0.0000),children:[name:left_ring2,position:(0.0296,0.0006,-0.0051),rotation:(1.0000,0.0000,0.0000,0.0000),children:[name:left_ring3,position:(0.0239,-0.0023,-0.0068),rotation:(1.0000,0.0000,0.0000,0.0000)]],name:left_thumb1,position:(0.0417,-0.0201,0.0277),rotation:(1.0000,0.0000,0.0000,0.0000),children:[name:left_thumb2,position:(0.0171,0.0018,0.0273),rotation:(1.0000,0.0000,0.0000,0.0000),children:[name:left_thumb3,position:(0.0209,-0.0047,0.0170),rotation:(1.0000,0.0000,0.0000,0.0000)]]]]]],name:right_collar,position:(-0.0461,0.0768,-0.0085),rotation:(1.0000,0.0000,0.0000,0.0000),children:[name:right_shoulder,position:(-0.1304,0.0588,-0.0126),rotation:(1.0000,0.0000,0.0000,0.0000),children:[name:right_elbow,position:(-0.2725,-0.0459,-0.0324),rotation:(1.0000,0.0000,0.0000,0.0000),children:[name:right_wrist,position:(-0.2634,0.0000,-0.0118),rotation:(1.0000,0.0000,0.0000,0.0000),children:[name:right_index1,position:(-0.1056,-0.0122,0.0201),rotation:(1.0000,0.0000,0.0000,0.0000),children:[name:right_index2,position:(-0.0333,0.0016,0.0037),rotation:(1.0000,0.0000,0.0000,0.0000),children:[name:right_index3,position:(-0.0235,-0.0027,-0.0002),rotation:(1.0000,0.0000,0.0000,0.0000)]],name:right_middle1,position:(-0.1137,-0.0095,-0.0049),rotation:(1.0000,0.0000,0.0000,0.0000),children:[name:right_middle2,position:(-0.0319,0.0004,-0.0045),rotation:(1.0000,0.0000,0.0000,0.0000),children:[name:right_middle3,position:(-0.0244,-0.0025,-0.0044),rotation:(1.0000,0.0000,0.0000,0.0000)]],name:right_pinky1,position:(-0.0877,-0.0184,-0.0474),rotation:(1.0000,0.0000,0.0000,0.0000),children:[name:right_pinky2,position:(-0.0170,-0.0017,-0.0120),rotation:(1.0000,0.0000,0.0000,0.0000),children:[name:right_pinky3,position:(-0.0167,-0.0020,-0.0112),rotation:(1.0000,0.0000,0.0000,0.0000)]],name:right_ring1,position:(-0.1014,-0.0126,-0.0299),rotation:(1.0000,0.0000,0.0000,0.0000),children:[name:right_ring2,position:(-0.0296,0.0006,-0.0051),rotation:(1.0000,0.0000,0.0000,0.0000),children:[name:right_ring3,position:(-0.0239,-0.0023,-0.0068),rotation:(1.0000,0.0000,0.0000,0.0000)]],name:right_thumb1,position:(-0.0412,-0.0237,0.0268),rotation:(1.0000,0.0000,0.0000,0.0000),children:[name:right_thumb2,position:(-0.0171,0.0018,0.0273),rotation:(1.0000,0.0000,0.0000,0.0000),children:[name:right_thumb3,position:(-0.0209,-0.0047,0.0170),rotation:(1.0000,0.0000,0.0000,0.0000)]]]]]]]]]]]]"
                ),
                "instruction": "create the walking animation for the human male",
            }
)
# response = keyframe.invoke_chain(
#     {
#                 "object": "whale",
#                 "object_json": (
#                     "name:Armature,position:(0.0000,0.0000,0.0000),rotation:(-0.7,0.0,0.0,0.7),children:[name:Root,position:(0.0000,0.0168,0.0141),rotation:(0.7,0.0,0.0,0.7),children:[name:Head,position:(0.0000,0.0062,0.0198),rotation:(0.7,0.0,0.0,0.7),children:[name:Head_end,position:(0.0000,0.0107,0.0000),rotation:(0.0,0.0,0.0,1.0)],"
#                     "name:Spine1,position:(0.0000,0.0050,0.0154),rotation:(-0.7,0.0,0.0,0.7),children:[name:Spine2,position:(0.0000,0.0156,0.0000),rotation:(0.0,0.0,0.0,1.0),children:[name:Spine3,position:(0.0000,0.0166,0.0000),rotation:(0.0,0.0,0.0,1.0),children:[name:Spine4,position:(0.0000,0.0172,0.0000),rotation:(-0.1,0.0,0.0,1.0),"
#                     "children:[name:Tail,position:(0.0000,0.0196,0.0000),rotation:(0.0,0.0,0.0,1.0),children:[name:Tail_end,position:(0.0000,0.0133,0.0000),rotation:(0.0,0.0,0.0,1.0)]]]],name:TopFlipper.L,position:(-0.0107,0.0087,-0.0087),rotation:(-0.4,0.0,0.3,0.9),children:[name:MidFlipper.L,position:(0.0000,0.0067,0.0000),rotation:(0.0,0.1,0.0,1.0),"
#                     "children:[name:BottomFlipper.L,position:(0.0000,0.0043,0.0000),rotation:(0.0,0.0,-0.1,1.0),children:[name:BottomFlipper.L_end,position:(0.0000,0.0076,0.0000),rotation:(0.0,0.0,0.0,1.0)]]],name:TopFlipper.R,position:(0.0092,0.0078,-0.0084),rotation:(-0.4,0.0,-0.3,0.9),children:[name:MidFlipper.R,position:(0.0000,0.0082,0.0000),rotation:(0.1,-0.1,0.1,1.0),"
#                     "children:[name:BottomFlipper.R,position:(0.0000,0.0053,0.0000),rotation:(0.0,0.0,0.2,1.0),children:[name:BottomFlipper.R_end,position:(0.0000,0.0072,0.0000),rotation:(0.0,0.0,0.0,1.0)]]]]]]"
#                 ),
#                 "instruction": "create the jumping out of the water animation for the whale",
#             }
# )
# print(response)
# response = keyframe.invoke_chain(
#     {
#                 "object": "racoon",
#                 "object_json": (
#                     "name:metarig,position:(0.00,0.00,0.00),rotation:(-0.7,0.0,0.0,0.7),"
#                     "children:[name:spine,position:(0.00,0.00,0.00),rotation"
#                     ":(0.7,0.0,0.0,0.7),children:[name:pelvis.L,position"
#                     ":(0.00,0.00,0.00),rotation:(-0.2,0.6,0.7,0.4),name:pelvis.R,"
#                     "position:(0.00,0.00,0.00),rotation:(0.2,0.6,0.7,-0.4),name:spine"
#                     ".001,position:(0.00,0.00,0.00),rotation:(0.0,0.0,0.0,1.0),children"
#                     ":[name:spine.002,position:(0.00,0.00,0.00),rotation"
#                     ":(0.0,0.0,0.0,1.0),children:[name:spine.003,position"
#                     ":(0.00,0.00,0.00),rotation:(-0.1,0.0,0.0,1.0),children:[name:"
#                     "breast.L,position:(0.00,0.00,0.00),rotation:(0.0,0.8,0.6,0.0),name"
#                     ":breast.R,position:(0.00,0.00,0.00),rotation:(0.0,0.8,0.6,0.0),"
#                     "name:shoulder.L,position:(0.00,0.00,0.00),rotation"
#                     ":(-0.7,0.2,0.4,0.6),children:[name:upper_arm.L,position"
#                     ":(0.00,0.00,0.00),rotation:(0.2,-0.7,0.4,0.5),children:[name:"
#                     "forearm.L,position:(0.00,0.00,0.00),rotation:(0.5,0.0,0.0,0.8),"
#                     "children:[name:hand.L,position:(0.00,0.00,0.00),rotation"
#                     ":(0.1,0.0,-0.1,1.0)]]],name:shoulder.R,position:(0.00,0.00,0.00),"
#                     "rotation:(-0.7,-0.2,-0.4,0.6),children:[name:upper_arm.R,position"
#                     ":(0.00,0.00,0.00),rotation:(-0.1,0.9,-0.3,0.4),children:[name:"
#                     "forearm.R,position:(0.00,0.00,0.00),rotation:(-0.1,0.2,-0.6,0.8),"
#                     "children:[name:hand.R,position:(0.00,0.00,0.00),rotation"
#                     ":(0.1,0.1,-0.2,1.0),children:[name:hand.R.001,position"
#                     ":(0.00,0.00,0.00),rotation:(0.3,0.0,0.7,0.6),children:[name:"
#                     "Spork_low,position:(0.00,0.00,0.00),rotation:(0.0,0.7,-0.7,0.1)"
#                     "]]]]],name:spine.006,position:(0.00,0.00,0.00),rotation"
#                     ":(-0.1,0.0,0.0,1.0),children:[name:ear.L,position:(0.00,0.01,0.00)"
#                     ",rotation:(-0.1,-0.1,0.2,1.0),name:ear.R,position:(0.00,0.01,0.00)"
#                     ",rotation:(-0.1,0.1,-0.5,0.9)]]]],name:tail,position"
#                     ":(0.00,0.00,0.00),rotation:(0.8,0.4,-0.1,-0.3),children:[name:tail"
#                     ".001,position:(0.00,0.00,0.00),rotation:(-0.2,0.1,-0.2,1.0),"
#                     "children:[name:tail.002,position:(0.00,0.00,0.00),rotation"
#                     ":(0.0,0.5,-0.5,0.7),children:[name:tail.003,position"
#                     ":(0.00,0.00,0.00),rotation:(-0.5,0.0,0.0,0.8)]]],name:thigh.L,"
#                     "position:(0.00,0.00,0.00),rotation:(1.0,0.1,-0.3,0.0),children:["
#                     "name:shin.L,position:(0.00,0.00,0.00),rotation:(0.2,0.3,-0.1,0.9),"
#                     "children:[name:foot.L,position:(0.00,0.00,0.00),rotation"
#                     ":(-0.5,0.1,0.2,0.8),children:[name:heel.02.L,position"
#                     ":(0.00,0.00,0.00),rotation:(-0.6,0.6,-0.2,-0.5),name:toe.L,"
#                     "position:(0.00,0.00,0.00),rotation:(-0.3,0.8,-0.4,-0.3)]]],name:"
#                     "thigh.R,position:(0.00,0.00,0.00),rotation:(1.0,-0.1,0.3,0.0),"
#                     "children:[name:shin.R,position:(0.00,0.00,0.00),rotation"
#                     ":(0.2,-0.3,0.1,0.9),children:[name:foot.R,position"
#                     ":(0.00,0.00,0.00),rotation:(-0.5,-0.1,-0.2,0.8),children:[name:"
#                     "heel.02.R,position:(0.00,0.00,0.00),rotation:(0.6,0.6,-0.2,0.5),"
#                     "name:toe.R,position:(0.00,0.00,0.00),rotation:(0.3,0.8,-0.4,0.3)"
#                     "]]]]]"
#                 ),
#                 "instruction": "create the walk animation for the racoon: ",
#             }
# )
print(response)
