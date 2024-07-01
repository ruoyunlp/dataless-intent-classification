import re

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, StoppingCriteria, StoppingCriteriaList, GenerationConfig, logging

logging.set_verbosity_error()

# Clean generation output
def clean(sent, is_chat=True):
    out = re.sub(r'<eos>', '', sent)

    if is_chat:
        out = re.sub(r'(<\|[\S]+\|>)', '\n', out)

    out = re.sub(r'\n+', '\n', out)
    out = out.splitlines()
    out = out[-1].strip()
    out = out.strip()

    out = re.sub(r'^[a-zA-Z]+: ', '', out)
    return out

# Cleaning before generation
def clean_text(text):
    out = text.strip()
    if not text.isascii():
        try:
            out = out.encode('latin-1')
            try:
                out = out.decode('utf-8')
            except UnicodeDecodeError:
                out = out.decode('latin-1')
        except UnicodeEncodeError:
            out = out.encode('utf-8').decode('utf-8')
    out = re.sub(r'\n+', '', out)
    return out

class KeywordStoppingCriteria(StoppingCriteria):
    def __init__(self, keyword_ids) -> None:
        super().__init__()
        self.keywords = keyword_ids

    def __call__(self, input_ids: torch.LongTensor, scores: torch.FloatTensor, **kwargs) -> bool:
        return input_ids[0][-1] in self.keywords

class ParaphraserModel:
    def __init__(self, model_type, device) -> None:
        self.model = AutoModelForCausalLM.from_pretrained(
            model_type,
            torch_dtype=torch.bfloat16,
            trust_remote_code=True).to(device)
        self.tokenizer = AutoTokenizer.from_pretrained(model_type)
        self.device = device

        self.generation_config = {
            "min_new_tokens": 1,
            "max_new_tokens": 512,
            "num_beams": 3,
            "do_sample": True,
            "early_stopping": True,
            "repetition_penalty": 1.2,
            "length_penalty": -1,
            "output_scores": True,
            "return_dict_in_generate": True,
            "eos_token_id": self.tokenizer.eos_token_id
        }
        self.generation_config = GenerationConfig.from_pretrained(model_type, **self.generation_config)

        stop_tokens = ['\n', '<|im_end|>']
        stop_tokens = [self.tokenizer(tok).input_ids[-1] for tok in stop_tokens]
        stop_tokens = KeywordStoppingCriteria(stop_tokens)
        self.stopping_criteria = StoppingCriteriaList([stop_tokens])
        self.template = None

    def initialise_task(self, task):
        self.template = open(f"paraphrase/templates/{task}-template.txt").read()

    def format_chat_prompt(self, prompt):
        # take first line as system prompt
        out = []
        lines = re.sub(r'\n+', '\n', prompt).splitlines()
        out.append({'role': 'system', 'content': f"You are a helpful assistant, your task is: {lines[0]}"})

        lines = lines[1:-1]
        for i, line in enumerate(lines):
            out.append({'role': 'user' if i % 2 == 0 else 'assistant', 'content': line})

        return out

    def chat_generate(self, prompt):
        prompt = self.format_chat_prompt(prompt)
        enc = self.tokenizer.apply_chat_template(prompt, return_tensors='pt', add_generation_prompt=False)
        enc = enc.to(self.device)
        out = self.model.generate(
            enc,
            generation_config=self.generation_config,
            stopping_criteria=self.stopping_criteria,
            use_cache=True)
        output_text = self.tokenizer.decode(out.sequences[0][enc.shape[-1]:])
        output_text = clean(output_text)
        return output_text

    def generate(self, utterance):
        utterance_clean = clean_text(utterance)
        prompt = self.template.replace('{}', utterance_clean)
        paraphrase = self.chat_generate(prompt)
        return paraphrase
