#!/bin/bash

python3 inference.py --model bge-large-en-v1.5 \
                     --paraphraser data/paraphrase/stablelm-2-1_6b-chat \
                     --descriptor descriptions.json \
                     --device cuda \
                     --task atis \
                     --do_save \
                     --do_paraphrase \
                     --do_masking \
                     --do_eval