from src.utils.registry import registry
from src.models.base import BaseChat, Response
from src.models.llava_chat import LLaVAChat
# from src.models.llava_rlhf_chat import LLaVARLHFChat
# from src.models.mplug_owl2_chat import mPLUGOwl2Chat
# from src.models.internvl_chat import InternVLChat
# from src.models.lrv_instruction_chat import LRVInstructionChat
from src.models.openai_chat import OpenAIChat
# from src.models.google_chat import GoogleChat
# from src.models.claude3_chat import ClaudeChat
# from src.models.qwen_plus_chat import QwenPlusChat
# from src.models.minigpt4_chat import MiniGPT4Chat
# from src.models.instructblip_chat import InstructBLIPChat
# from src.models.qwen_chat import QwenChat
# from src.models.otter_chat import OtterChat
# from src.models.mplug_owl_chat import mPLUGOwlChat
# from src.models.internlm_xcomposer_chat import InternLMXComposerChat
# from src.models.sharegpt4v_chat import ShareGPT4VChat
# from src.models.cogvlm_chat import CogVLMChat
# from src.models.phi3_chat import Phi3Chat
# from src.models.deepseek_chat import DeepSeekVL
# from src.models.hunyuan_chat import HunyuanChat
from typing import List


def load_chatmodel(model_id: str, device: str = "cuda:0") -> 'BaseChat':
    return registry.get_chatmodel_class(model_id)(model_id=model_id, device=device)


def model_zoo() -> List['BaseChat']:
    return registry.list_chatmodels()
