import torch
import re

import spacy
from spacy.tokens.doc import Doc

from typing import List, Tuple


class BaseParser:
    def __init__(self, processor) -> None:
        self.proc = processor

    def mask(self, sentence: str, tokens: Doc=None, **kwargs) -> Tuple[Doc, List[bool]]:
        raise NotImplementedError()

class NominalSpanParser(BaseParser):
    def mask(self, sentence: str, tokens: Doc=None, **kwargs) -> Tuple[Doc, List[bool]]:
        if tokens is None:
            tokens = self.proc(sentence)
        mask = torch.zeros(len(tokens)).bool()
        nominals = list(tokens.noun_chunks)  # Assume nominal spans are non-overlapping

        def f_func(sp):
            return len(sp) > 1 or sp[0].pos_ in ['NOUN', 'PROPN']

        nominals = list(filter(f_func, nominals))
        for span in nominals:
            mask[span.start:span.end] = 1

        return tokens, mask.tolist()

class ObjectRelParser(BaseParser):
    def mask(self, sentence: str, tokens: Doc=None, **kwargs) -> Tuple[Doc, List[bool]]:
        if tokens is None:
            tokens = self.proc(sentence)
        mask = torch.zeros(len(tokens)).bool()

        def mark_obj(token, markers, mark_true=False):
            if (mark_true or token.dep_ in ['ccomp', 'pobj', 'dobj']):
                markers[token.i] = True
            for child in token.children:
                mark_obj(child, markers, mark_true=markers[token.i])

        # Find root node
        roots = list(filter(lambda x: len(list(x.ancestors)) == 0, tokens))
        # Traverse tree and mark
        for root in roots:
            mark_obj(root, mask)

        return tokens, mask.tolist()


class Masker:
    def __init__(self, parse_func: BaseParser) -> None:
        self.parse_func = parse_func
        self.mask_token = '[MASK]'
        self.mask_pattern = r'\[MASK\]((\s|(\s[.,!?;]\s))?\[MASK\])+'

    def mask_sent(self, sentence: str) -> str:
        tokens, mask = self.parse_func.mask(sentence)
        if all(mask):  # Do not mask if entire sentence would be masked
            return sentence

        # substitute masked tokens
        output = []
        for (masked, tok) in zip(mask, list(tokens)):
            if masked:
                output.append(self.mask_token + tok.whitespace_)
            else:
                output.append(tok.text_with_ws)
        output = ''.join(output)

        # replace multiple consecutive mask tokens with a single
        output = re.sub(self.mask_pattern, self.mask_token, output)
        # replace multiple whitespaces
        output = re.sub(r'[^\S\n]+', ' ', output)
        output = re.sub(r'\.$', '', output)
        output = output.strip()

        return output

    def mask_sents(self, sentences: List[str]) -> List[str]:
        output = [self.mask_sent(sent) for sent in sentences]
        return output

dpm = spacy.load('en_core_web_trf')  # Change this for different dependency parsers
parser = ObjectRelParser(processor=dpm)

MASKER = Masker(parse_func=parser)
