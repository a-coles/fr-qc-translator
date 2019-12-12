# Seq2Seq Style Transfer from Quebec to Metropolitan French

This repo contains the submission for the COMP 550 final project at McGill University, and was written by:
* Arlie Coles
* Badreddine Sayah
* Vilayphone Vilaysouk

All code to rerun the experiments is included *except the corpus*, which contains copyrighted material. We can provide it upon request.

For the sake of repository size, we also do not include our experiment logs, but can provide these upon request.

## Prerequisites

Several dependencies are required to run this project. The easiest way to manage them is to create a virtual environment using Conda. You can do so from the root of the repo this way:

`conda create --name <name-of-env> --file requirements.txt`

If you get an `ImportError` regarding AllenNLP, you can install it manually this way:

`pip install allennlp`

## Training and testing

To run model training, run `python train_seq2seq.py` with the following possible options:

```
usage: train_seq2seq.py [-h] [--log_dir LOG_DIR] [--model_dir MODEL_DIR]
                        [--name NAME] [--continue_model CONTINUE_MODEL] [--bi]
                        [--bn] [--att]
                        config_file

positional arguments:
  config_file           Training config.

optional arguments:
  -h, --help            show this help message and exit
  --log_dir LOG_DIR     Log output dir.
  --model_dir MODEL_DIR
                        Saved model dir.
  --name NAME           Name for model.
  --continue_model CONTINUE_MODEL
                        Path to model for continuing training.
  --bi                  Use a bidirectional encoder.
  --bn                  Use batch normalization at encoder.
  --att                 Use a decoder with attention.

```

To run model testing, run `python test_seq2seq.py` with the following possible options:
```
usage: test_seq2seq.py [-h] [--log_dir LOG_DIR] [--name NAME] [--bi] [--att]
                       [--bn] [--write_idx WRITE_IDX]
                       config_file model_path

positional arguments:
  config_file           Testing config. Should be same as used to train.
  model_path            Path to model to test with.

optional arguments:
  -h, --help            show this help message and exit
  --log_dir LOG_DIR     Log output dir.
  --name NAME           Name for model.
  --bi                  Use a bidirectional encoder.
  --att                 Use a decoder with attention.
  --bn                  Use batch normalization at encoder.
  --write_idx WRITE_IDX
                        Index of output examples to write. Change for new
                        examples.

```
