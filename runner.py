import os
import json
import pathlib

import torch
from torch.utils.data import DataLoader

from sklearn.metrics import accuracy_score, f1_score
from typing import List, Iterable
from tqdm import tqdm

from paraphrase.paraphraser import ParaphraserModel
from masker import MASKER
from wrappers import BaseInferenceModel, GTEModel, BGEModel, E5Model, InstructorModel

model_to_wrapper = {
    'bge-small-en-v1.5': GTEModel,
    'bge-base-en-v1.5': GTEModel,
    'bge-large-en-v1.5': GTEModel,

    'gte-small': GTEModel,
    'gte-base': GTEModel,
    'gte-large': GTEModel,

    'stella-base-en-v2': E5Model,

    'instructor-large': InstructorModel,

    'e5-base-v2': E5Model,
    'e5-large': E5Model,
    'e5-large-v2': E5Model,
    'multilingual-e5-large': E5Model,
    'multilingual-e5-large-instruct': E5Model
}

def batched_compute(encoder: BaseInferenceModel, data: List[str], batch_size: int=8, norm: bool=True, desc=None):
    """Compute sentence embeddings in a batched manner

    Args:
        encoder (BaseInferenceModel): Some form of encoder which produces sentence embeddings as torch tensors
        data (List[str]): Dataset to embed, must be supported by torch DataLoaders
        batch_size (int, optional): Defaults to 8.
        norm (bool, optional): Normalise embedding tensor. Defaults to False.

    Returns:
        torch.Tensor: Embedding tensors of num_entries x hidden_size
    """
    data_loader = DataLoader(data, batch_size=batch_size, shuffle=False)
    output = []
    data_iterator = tqdm(data_loader)
    if desc is not None:
        data_iterator.set_description(desc)
    with torch.no_grad():
        for batch in data_iterator:
            out = encoder.get_sentence_embeds(batch, norm=norm)
            output.append(out)
        output = torch.cat(output, dim=0)
    return output

def get_sims(embeds: torch.Tensor, desc_embeds: torch.Tensor, task: str):
    sims = normalise(embeds) @ normalise(desc_embeds).T
    if task == 'atis':  # Filter classes as per previous works
        sims[:, [6, 8]] = -1
    return sims

def get_preds(embeds: torch.Tensor, desc_embeds: torch.Tensor, task: str):
    sims = get_sims(embeds, desc_embeds, task)
    preds = torch.argmax(sims, dim=1)
    return preds

def score(labels: torch.Tensor, preds: torch.Tensor, do_print: bool=True, tag: str=None):
    acc = accuracy_score(labels, preds)
    f1 = f1_score(labels, preds, average='macro')
    if do_print:
        tag = f"[{tag}]" if tag is not None else ''
        print(f"{tag} acc: {(acc):.4f} macro-f1: {(f1):.4f}")
        print(f"{(acc * 100):.4g} {(f1 * 100):.4g}")
    return acc, f1

def normalise(v: torch.Tensor):
    return v / v.norm(dim=-1).unsqueeze(-1)


class ParaphraseStep(BaseInferenceModel):
    def __init__(self, paraphraser: ParaphraserModel, encoder: BaseInferenceModel, task: str) -> None:
        super().__init__()
        self.paraphraser = paraphraser
        self.encoder = encoder

        self.paraphraser.initialise_task(task)

    def get_sentence_embeds(self, sents: List[str], norm=True, **kwargs) -> torch.Tensor:
        out = [self.paraphraser.generate(sent) for sent in sents]
        out = self.encoder.get_sentence_embeds(out, norm=norm)
        return out

class MaskStep(BaseInferenceModel):
    def __init__(self, encoder: BaseInferenceModel) -> None:
        super().__init__()
        self.encoder = encoder
        self.masker = MASKER

    def get_sentence_embeds(self, sents: List[str], norm=True, **kwargs) -> torch.Tensor:
        out = self.masker.mask_sents(sents)
        out = self.encoder.get_sentence_embeds(out, norm=norm)
        return out

def run_inference(
        task=None,
        path_enc=None,
        path_par=None,
        path_out=None,
        path_desc=None,
        do_paraphrase=False,
        do_masking=False,
        do_save=False,
        do_eval=False,
        top_k=False,
        device=None,
        seed=None
    ):
    torch.random.manual_seed(seed)
    if do_save and not os.path.exists(path_out):
        os.mkdir(path_out)

    model_name = os.path.split(path_enc)[-1]
    wrapper = model_to_wrapper[model_name]
    encoder = wrapper(model_type=path_enc, device=device)
    pipeline_steps = [encoder]

    path_desc = pathlib.Path(path_desc)
    if path_desc.suffix == '.json':  # Need to precompute description embeddings
        path_precomp = 'precompute'
        if not os.path.exists(path_precomp):
            os.mkdir(path_precomp)
        descriptions = list(json.load(open(f"data/preprocessed/{task}/{path_desc}")).values())
        desc_embeds = batched_compute(encoder, descriptions, batch_size=8, norm=False)
        torch.save(desc_embeds, f"{path_precomp}/{task}-{model_name}-{path_desc.stem}.pt")

    elif path_desc.suffix == '.pt':  # Load precomputed description embeddings
        desc_embeds = torch.load(path_desc)
    else:
        raise TypeError(f"Description file type not supported {repr(path_desc.suffix)}")

    # Run inference
    data = json.load(open(f"data/preprocessed/{task}/data-full-shuffled.json"))['data']
    # Filter according to previous work
    filter_idx = torch.tensor([entry['intent'] not in ['day_name', 'cheapest'] for entry in data])
    data = list(filter(lambda x: x['intent'] not in ['day_name', 'cheapest'], data))
    intents = open(f"data/preprocessed/{task}/intents.txt").read().splitlines()
    labels = torch.tensor([intents.index(entry['intent']) for entry in data])
    texts = [entry['text'] for entry in data]

    embeds = batched_compute(encoder, texts, batch_size=8, norm=False, desc='Computing Embeddings')
    sent_rep = embeds

    # Add masking component if necessary
    if do_masking:
        # Instantiate Masker if necessary
        masker = MaskStep(encoder)
        pipeline_steps.append(masker)

        # Compute embeddings
        masked_embeds = batched_compute(masker, texts, batch_size=8, norm=False, desc='Computing Masking')
        overlap = torch.load(f"data/overlaps/{task}-overlaps.pt")  # Load overlap matrix

        # Calculate similarity scores for overlap detection
        sims = get_sims(sent_rep, desc_embeds, task)
        top_preds = torch.argsort(sims, descending=True, dim=1)
        topkpreds = top_preds[:, :top_k]
        overlaps = []
        for i in range(len(texts)):
            candidates = torch.combinations(topkpreds[i, :])
            overlaps.append((overlap[candidates[:, 0], candidates[:, 1]]).any())
        overlaps = torch.tensor(overlaps)

        overlaps = ((masked_embeds - embeds).norm(dim=1) > 0.001) & overlaps
        if task == 'atis': overlaps = torch.ones(len(embeds))  # always use masking as dataset only has 1 domain
        masked_embeds = overlaps.unsqueeze(-1) * masked_embeds

        sent_rep += masked_embeds

    # Load paraphraser if necessary
    if do_paraphrase:
        if os.path.exists(f"{path_par}/{task}-clean-userdescs.json"):
            f_paraph = json.load(open(f"{path_par}/{task}-clean-userdescs.json"))
            paraphrases = [entry['userdesc'] for entry in f_paraph]
            paraph_embeds = batched_compute(encoder, paraphrases, batch_size=8, norm=False, desc='Computing Paraphrasal')
            paraph_embeds = paraph_embeds[filter_idx]
        else:
            paraphraser = ParaphraserModel(path_par, device)
            paraphraser = ParaphraseStep(paraphraser, encoder, task)
            paraph_embeds = batched_compute(paraphraser, texts, batch_size=8, norm=False, desc='Computing Paraphrasal')

        sent_rep += paraph_embeds

    if do_save:
        torch.save(sent_rep, f"{path_out}/{task}-{model_name}-{int(do_paraphrase)}{int(do_masking)}.pt")

    if do_eval:
        preds = get_preds(sent_rep, desc_embeds, task)
        score(labels, preds)
