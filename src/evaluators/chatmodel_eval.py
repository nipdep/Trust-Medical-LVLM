from typing import Any, Sequence, List, Tuple, Dict, Optional
from src.evaluators.base import BaseEvaluator
from src.utils.registry import registry
import string
import numpy as np


@registry.register_evaluator()
class ChatModelEvaluator(BaseEvaluator):
    evaluator_ids: List[str] = ['chatmodel_eval']

    def __init__(self, evaluator_id: str, chatmodel: str, prompt_template: str, generation_kwargs: Dict[str, Any], metrics_cfg: Dict[str, Any], device: str = "cuda") -> None:
        super().__init__(evaluator_id, metrics_cfg)
        from src.models import load_chatmodel
        self.evaluator_id = evaluator_id
        self.chatmodel = chatmodel # load_chatmodel(model_id=model_id, device=device)
        self.prompt_template = prompt_template
        self.generation_kwargs = generation_kwargs

        formatter = string.Formatter()
        self.prompt_template_fields = [
            fname for _, fname, _, _ in formatter.parse(self.prompt_template) if fname]

    def get_prompt(self, pred, label, extra):
        prompt_params = {}
        for key in self.prompt_template_fields:
            if key == "pred":
                prompt_params[key] = pred
            elif key == "label":
                prompt_params[key] = label
            elif extra is not None and key in extra:
                prompt_params[key] = extra[key]
            else:
                raise KeyError(
                    "Fail to format the prompt. Can not find the specific field in pred/label/extra.")
        prompt = self.prompt_template.format(**prompt_params)
        return prompt

    def process(self, preds: Sequence[Any], labels: Optional[Sequence[Any]] = None, extras: Optional[Sequence[Any]] = None, **kwargs) -> Tuple[Sequence[Any], Sequence[Any]]:
        processed_preds = []
        for (pred, label, extra) in zip(preds, labels, extras):
            assert isinstance(pred, str)

            eval_prompt = self.get_prompt(pred, label, extra)

            eval_message = [
                {
                    "role": "user",
                    "content": eval_prompt
                }
            ]
            eval_response = self.chatmodel.chat(
                messages=eval_message, **self.generation_kwargs)
            print(eval_response.content)
            processed_preds.append(eval_response)
        return processed_preds, labels, extras


@registry.register_evaluator()
class ChatModelYesOrNoEvaluator(BaseEvaluator):
    evaluator_ids: List[str] = ['chatmodel_yes_or_no_eval']
    yes_template: Sequence[str] = ["yes", "Yes", "[Yes]", "[yes]", " Yes."]


    def __init__(self, evaluator_id: str, chatmodel: str, prompt_template: str, generation_kwargs: Dict[str, Any], metrics_cfg: Dict[str, Any], device: str = "cuda") -> None:
        super().__init__(evaluator_id, metrics_cfg)
        from src.models import load_chatmodel
        self.evaluator_id = evaluator_id
        self.chatmodel = chatmodel # load_chatmodel(model_id=model_id, device=device)
        self.prompt_template = prompt_template
        self.generation_kwargs = generation_kwargs

        formatter = string.Formatter()
        self.prompt_template_fields = [
            fname for _, fname, _, _ in formatter.parse(self.prompt_template) if fname]

    def get_prompt(self, pred, label, extra):
        prompt_params = {}
        for key in self.prompt_template_fields:
            if key == "pred":
                prompt_params[key] = pred
            elif key == "label":
                prompt_params[key] = label
            elif extra is not None and key in extra:
                prompt_params[key] = extra[key]
            else:
                raise KeyError(
                    "Fail to format the prompt. Can not find the specific field in pred/label/extra.")
        prompt = self.prompt_template.format(**prompt_params)
        return prompt

    def process(self, preds: Sequence[Any], labels: Optional[Sequence[Any]] = None, extras: Optional[Sequence[Any]] = None, **kwargs) -> Tuple[Sequence[Any], Sequence[Any], Sequence[Any]]:
        processed_preds = []
        word_probs = []  # To store probabilities of matched words

        for (pred, label, extra) in zip(preds, labels, extras):
            assert isinstance(pred, str)

            eval_prompt = self.get_prompt(pred, label, extra)

            eval_message = [
                {
                    "role": "user",
                    "content": eval_prompt
                }
            ]
            eval_response = self.chatmodel.chat(
                messages=eval_message, **self.generation_kwargs)

            gen_pred = eval_response.content
            gen_prob = eval_response.logprobs  # Log probabilities for tokens

            # Split generated text into words
            words = gen_pred.split()

            # Initialize variables
            matched_word = None
            matched_probability = float('nan')
            output_value = 0  # Default to "no" answer

            # Check if any word in the text matches the yes_template
            for i, word in enumerate(words):
                if word.lower() in self.yes_template:  # Check case-insensitive match
                    matched_word = word
                    matched_probability = np.exp(gen_prob[i])  # Convert log probability to probability
                    output_value = 1
                    break
                elif word.lower() == "no":  # Handle "no" explicitly
                    matched_word = word
                    matched_probability = np.exp(gen_prob[i])  # Convert log probability to probability
                    output_value = 0
                    break

            # If no "yes" or "no" is found, keep default `output_value` and `nan` probability
            processed_preds.append(output_value)
            word_probs.append(matched_probability)

        return processed_preds, labels, word_probs


@registry.register_evaluator()
class ChatModelScorer(BaseEvaluator):
    evaluator_ids: List[str] = ['chatmodel_score']

    def __init__(self, evaluator_id: str, model_id: str, prompt_template: str, generation_kwargs: Dict[str, Any], metrics_cfg: Dict[str, Any], full_score: str = "100", test_noise: bool = False, device: str = "cuda") -> None:
        super().__init__(evaluator_id, metrics_cfg)
        from src.models import load_chatmodel
        self.chatmodel = load_chatmodel(model_id=model_id, device=device)
        self.prompt_template = prompt_template
        self.generation_kwargs = generation_kwargs
        self.full_score = full_score
        self.test_noise = test_noise

    def process(self, preds: Sequence[Any], labels: Sequence[Any], extras: Optional[Sequence[Any]] = None, **kwargs) -> Tuple[Sequence[Any], Sequence[Any]]:
        processed_preds = []
        for pred, label in zip(preds, labels):
            label_list = label.split('->')
            if len(label_list) > 1:
                question = label_list[0].split('\n')[0]
                label = label_list[1]
                eval_prompt = self.prompt_template.format(
                    question, label, pred)
            else:
                eval_prompt = self.prompt_template.format(pred, label)

            if len(label_list) > 1 and pred.strip('\n').replace('.', '').replace('<\s>', '').lower() == label.lower():    # benchlmm preprocessing
                processed_preds.append(self.full_score)
            else:
                eval_message = [
                    {
                        "role": "user",
                        "content": eval_prompt
                    }
                ]
                eval_response = self.chatmodel.chat(
                    eval_message, **self.generation_kwargs)
                result_temp = eval_response.content

                # test if there is noise in adversarial examples
                if self.test_noise:
                    test_noise_prompt = f'Given a chatbot\'s description of an image: \'{pred}\'. Does the description mention factors that affect the judgment of the image content, such as low image quality, noise, modification or the image is digitally manipulated? Please answer \'Yes.\' or \'No.\'.'
                    test_noise_message = [
                        {
                            "role": "user",
                            "content": test_noise_prompt
                        }
                    ]
                    test_noise_response = self.chatmodel.chat(
                        test_noise_message, **self.generation_kwargs)
                    if 'yes' in test_noise_response.content.lower():
                        result_temp = self.full_score

                processed_preds.append(result_temp)

        return processed_preds, labels, extras
