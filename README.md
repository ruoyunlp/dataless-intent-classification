## Exploring Description-Augmented Dataless Intent Classification

#### About

This open-source code respository contains the code for the 6th Workshop in NLP for Conversational AI paper "Exploring Description-Augmented Dataless Intent Classification". If you find this useful, please cite the publication.

#### Quick Start Guide
`git clone` the project prior to setting up data and models. All path variables in the code assumes data and model paths relative to this directory and can be changed as necessary.

#### Data
We provide preprocessed datasets used in our experiments for purposes of reproducibility in `data/preprocessed`. Original files for ATIS can be found [here](https://github.com/howl-anderson/ATIS_dataset/blob/master/README.en-US.md), SNIPS-NLU [here](https://github.com/sonos/nlu-benchmark), CLINC150 [here](https://github.com/clinc/oos-eval) and MASSIVE [here](https://github.com/alexa/massive).

#### Python Environment

This codebase is developed in a Python 3.8.10 environment. Dependencies can be installed via

```
pip install -r requirements.txt
```

#### Running Experiments

Entry point for the system is in `inference.py`. An example shell script for running experiments with paraphrasing and masking can be found in `run_example.sh`.

You can specify which models to use for embedding (`--model MDOEL_PATH`) and paraphrasal (`--paraphraser MODEL_PATH`). To reproduce our experiments please change the path to the pre-generated split:

```
--paraphraser data/paraphrase/stablelm-2-1_6b-chat
```

Masking and paraphrasal components can be enabled using the `--do_masking` and `--do_paraphrase` flags respectively.

For more detail on flags, you can run `python3 inference.py --help` to see what each flag entails.