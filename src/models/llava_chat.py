import os
import json
import pandas
import random
import pickle
import numpy as np
from tqdm import tqdm
from PIL import Image
from scipy.special import softmax
import requests
from typing import List, Dict, Any, Tuple, Callable
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from transformers import CLIPImageProcessor, CLIPVisionModel
from src.models.llava_med import LlavaMPTForCausalLM, LlavaLlamaForCausalLM, conv_templates, SeparatorStyle
# from torch.nn import CrossEntropyLoss
from torchvision import transforms
# from transformers import AutoProcessor
from src.utils.registry import registry
from src.models.base import BaseChat, Response
from omegaconf import OmegaConf
from src.utils.utils import get_abs_path

from PIL import Image
from io import BytesIO


DEFAULT_IMAGE_TOKEN = "<image>"
DEFAULT_IMAGE_PATCH_TOKEN = "<im_patch>"
DEFAULT_IM_START_TOKEN = "<im_start>"
DEFAULT_IM_END_TOKEN = "<im_end>"


image_transform = transforms.Compose(
    [
        # transforms.Resize(
        #     224, interpolation=transforms.InterpolationMode.BICUBIC
        # ),
        # transforms.CenterCrop(224),
        transforms.ToTensor(),
        # transforms.Normalize(
        #     mean=(0.48145466, 0.4578275, 0.40821073),
        #     std=(0.26862954, 0.26130258, 0.27577711),
        # ),
    ]
)


@registry.register_chatmodel()
class LLaVAChat(BaseChat):

    MODEL_CONFIG = {
        "llava-med-1.5-7b": 'configs/models/llava/llava-med-1.5-7b.yaml',
        "llava-med-7b-delta": 'configs/models/llava/llava-med-7b-delta.yaml',
        "llava-med-7b-mpt": 'configs/models/llava/llava-med-7b-mpt.yaml',
        "llava-med": 'configs/models/llava/llava-med.yaml',
    }
    def __init__(self, model_id: str, device="cuda:0"):
        super().__init__(model_id)
        self.device = device
        config = self.MODEL_CONFIG[self.model_id]
        self.config = OmegaConf.load(get_abs_path(config))
        # model_path = "liuhaotian/LLaVA-Lightning-MPT-7B-preview"
        # model_path = "microsoft/llava-med-v1.5-mistral-7b"
        # model_path = "microsoft/llava-med-7b-delta"
        # model_path = "kms-healthcare/llava-med"

        model_name = self.config.model.model_name
        self.model_name = model_name
        model_path = self.config.model.model_path

        self.tokenizer, self.model, self.image_processor, self.context_len = self.load_model(
            model_path, model_name)
        self.conv = self.get_conv(model_name)
        self.image_process_mode = "Resize"  # Crop, Resize, Pad
        self.dtype = torch.float16

        vision_tower = self.model.get_model().vision_tower[0]
        vision_tower.to(device=self.device, dtype=self.dtype)
        self.model.to(device=self.device, dtype=self.dtype)

    def get_image(self, image):
        if type(image) is str:
            try:
                return Image.open(image).convert("RGB")
            except Exception as e:
                print(f"Fail to read image: {image}")
                exit(-1)
        elif type(image) is Image.Image:
            return image
        else:
            raise NotImplementedError(f"Invalid type of Image: {type(image)}")

    @torch.no_grad()
    def chat(self, messages: List, **generation_kwargs):
        # TODO: if system message provided.
        assert len(messages) == 1, 'Only support one-turn conversation currently'
        for message in messages:
            if message["role"] in ["system", "user", "assistant"]:
                if message["role"] == "user":
                    if isinstance(message["content"], dict):
                        # multimodal
                        image_path = message["content"]["image_path"]
                        user_message = message["content"]["text"]
                    else:
                        image_path = None
                        user_message = message["content"]
                elif message["role"] == "assistant":
                    # TODO: add assistant answer into the conversation
                    pass
            else:
                raise ValueError("Unsupported role. Only system, user and assistant are supported.")

        if generation_kwargs.get("do_sample") == False:
            temperature = 0.0
            top_p = 1.0
        else:
            # TODO: set the default value
            temperature = generation_kwargs.get("temperature", 0.0)
            top_p = generation_kwargs.get("top_p", 1.0)

        args = type('Args', (), {
            "model_name": self.model_name,
            "query": user_message,
            "conv_mode": None,
            "image_file": image_path,
            "sep": ",",
            "temperature": temperature,
            "top_p": top_p,
            "num_beams": 1,
            "max_new_tokens": generation_kwargs.get("max_new_tokens", 256),
            'dtype': torch.float16
        })() # TODO: include args in the chat function

        image = self.get_image(image_path)
        prompt, stop_str = self.generate_prompt(user_message, image)
        output_text = self.do_generate(prompt, image, stop_str=stop_str, dtype=self.dtype)
        return Response(self.model_id, output_text, None, None)

    def generate_prompt(self, question, image):
        conv = self.conv.copy()
        text = question + '\n<image>'
        text = (text, image, self.image_process_mode)
        conv.append_message(conv.roles[0], text)
        conv.append_message(conv.roles[1], None)
        prompt = conv.get_prompt()
        stop_str = conv.sep if conv.sep_style in [
            SeparatorStyle.SINGLE, SeparatorStyle.MPT] else conv.sep2
        return prompt, stop_str

    
    def get_conv(self, model_name):
        if "llava" in model_name.lower():
            if "v1" in model_name.lower():
                template_name = "llava_v1"
            elif "mpt" in model_name.lower():
                template_name = "mpt_multimodal"
            else:
                template_name = "multimodal"
        elif "mpt" in model_name:
            template_name = "mpt_text"
        elif "koala" in model_name:  # Hardcode the condition
            template_name = "bair_v1"
        elif "v1" in model_name:    # vicuna v1_1/v1_2
            template_name = "vicuna_v1_1"
        else:
            template_name = "v1"
        return conv_templates[template_name].copy()


    def load_model(self, model_path, model_name, dtype=torch.float16, device='cpu'):
        # get tokenizer and model
        tokenizer = AutoTokenizer.from_pretrained(model_path)
        if 'llava' in model_name.lower():
            if 'mpt' in model_name.lower():
                model = LlavaMPTForCausalLM.from_pretrained(
                    model_path, torch_dtype=dtype, low_cpu_mem_usage=True)
            else:
                model = LlavaLlamaForCausalLM.from_pretrained(
                    model_path, torch_dtype=dtype, low_cpu_mem_usage=True)
        elif 'mpt' in model_name.lower():
            model = AutoModelForCausalLM.from_pretrained(
                model_path, torch_dtype=dtype, low_cpu_mem_usage=True, trust_remote_code=True)
        else:
            model = AutoModelForCausalLM.from_pretrained(
                model_path, torch_dtype=dtype, low_cpu_mem_usage=True)

        # get image processor
        image_processor = None
        if 'llava' in model_name.lower():
            image_processor = CLIPImageProcessor.from_pretrained(
                model.config.mm_vision_tower, torch_dtype=dtype)

            mm_use_im_start_end = getattr(
                model.config, "mm_use_im_start_end", False)
            tokenizer.add_tokens([DEFAULT_IMAGE_PATCH_TOKEN], special_tokens=True)
            if mm_use_im_start_end:
                tokenizer.add_tokens(
                    [DEFAULT_IM_START_TOKEN, DEFAULT_IM_END_TOKEN], special_tokens=True)

            vision_tower = model.get_model().vision_tower[0]
            if vision_tower.device.type == 'meta':
                vision_tower = CLIPVisionModel.from_pretrained(
                    vision_tower.config._name_or_path, torch_dtype=dtype, low_cpu_mem_usage=True).to(device=device)
                model.get_model().vision_tower[0] = vision_tower
            else:
                vision_tower.to(device=device, dtype=dtype)

            vision_config = vision_tower.config
            vision_config.im_patch_token = tokenizer.convert_tokens_to_ids(
                [DEFAULT_IMAGE_PATCH_TOKEN])[0]
            vision_config.use_im_start_end = mm_use_im_start_end
            if mm_use_im_start_end:
                vision_config.im_start_token, vision_config.im_end_token = tokenizer.convert_tokens_to_ids(
                    [DEFAULT_IM_START_TOKEN, DEFAULT_IM_END_TOKEN])

        if hasattr(model.config, "max_sequence_length"):
            context_len = model.config.max_sequence_length
        else:
            context_len = 2048

        model = model.to(device=device)
        # model = model.to_empty(device="cuda")

        return tokenizer, model, image_processor, context_len

    def do_generate(self, prompt, image, dtype=torch.float16, temperature=0.2, max_new_tokens=64, stop_str=None, keep_aspect_ratio=False):
        images = [image]
        assert len(images) == prompt.count(
            DEFAULT_IMAGE_TOKEN), "Number of images does not match number of <image> tokens in prompt"

        if keep_aspect_ratio:
            new_images = []
            for image_idx, image in enumerate(images):
                max_hw, min_hw = max(image.size), min(image.size)
                aspect_ratio = max_hw / min_hw
                max_len, min_len = 448, 224
                shortest_edge = int(min(max_len / aspect_ratio, min_len))
                image = self.image_processor.preprocess(image, return_tensors='pt', do_center_crop=False, size={
                                                        "shortest_edge": shortest_edge})['pixel_values'][0]
                new_images.append(image.to(self.model.device, dtype=dtype))
                # replace the image token with the image patch token in the prompt (each occurrence)
                cur_token_len = (image.shape[1]//14) * (image.shape[2]//14)
                replace_token = DEFAULT_IMAGE_PATCH_TOKEN * cur_token_len
                if getattr(self.model.config, 'mm_use_im_start_end', False):
                    replace_token = DEFAULT_IM_START_TOKEN + replace_token + DEFAULT_IM_END_TOKEN
                prompt = prompt.replace(DEFAULT_IMAGE_TOKEN, replace_token, 1)
            images = new_images
        else:
            images = self.image_processor(images, return_tensors='pt')[
                'pixel_values']
            images = images.to(self.model.device, dtype=dtype)
            # HACK: 256 is the max image token length hacked
            replace_token = DEFAULT_IMAGE_PATCH_TOKEN * 256
            if getattr(self.model.config, 'mm_use_im_start_end', False):
                replace_token = DEFAULT_IM_START_TOKEN + replace_token + DEFAULT_IM_END_TOKEN
            prompt = prompt.replace(DEFAULT_IMAGE_TOKEN, replace_token)

        stop_idx = None
        if stop_str is not None:
            stop_idx = self.tokenizer(stop_str).input_ids
            if len(stop_idx) == 1:
                stop_idx = stop_idx[0]
            else:
                stop_idx = None

        input_ids = self.tokenizer(prompt).input_ids
        output_ids = list(input_ids)
        pred_ids = []

        max_src_len = self.context_len - max_new_tokens - 8
        input_ids = input_ids[-max_src_len:]

        for i in range(max_new_tokens):
            if i == 0:
                out = self.model(
                    torch.as_tensor([input_ids]).to(self.model.device),
                    use_cache=True,
                    images=images)
                logits = out.logits
                past_key_values = out.past_key_values
            else:
                out = self.model(input_ids=torch.as_tensor([[token]], device=self.model.device),
                                 use_cache=True,
                                 attention_mask=torch.ones(
                                     1, past_key_values[0][0].shape[-2] + 1, device=self.model.device),
                                 past_key_values=past_key_values)
                logits = out.logits
                past_key_values = out.past_key_values

            last_token_logits = logits[0][-1]
            if temperature < 1e-4:
                token = int(torch.argmax(last_token_logits))
            else:
                probs = torch.softmax(last_token_logits / temperature, dim=-1)
                token = int(torch.multinomial(probs, num_samples=1))

            output_ids.append(token)
            pred_ids.append(token)

            if stop_idx is not None and token == stop_idx:
                break
            elif token == self.tokenizer.eos_token_id:
                break
            elif i == max_new_tokens - 1:
                break

        output = self.tokenizer.decode(pred_ids, skip_special_tokens=True)
        if stop_str is not None:
            pos = output.rfind(stop_str)
            if pos != -1:
                output = output[:pos]

        return output


def load_candidates_medical(data):
    a, b, c, d = data.get('option_A'), data.get(
        'option_B'), data.get('option_C'), data.get('option_D')
    answer_list = [a, b]
    if c is not None:
        answer_list.append(c)
    if d is not None:
        answer_list.append(d)
    return answer_list


def load_prompt(question, idx=4):
    prompts = ["{}".format(question),
               "{} Answer:".format(question),
               "{} The answer is".format(question),
               "Question: {} Answer:".format(question),
               "Question: {} The answer is".format(question)
               ]
    return prompts[idx]


def bytes2PIL(bytes_img):
    '''Transform bytes image to PIL.
    Args:
        bytes_img: Bytes image.
    '''
    pil_img = Image.open(BytesIO(bytes_img)).convert("RGB")
    return pil_img

