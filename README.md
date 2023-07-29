# FutureTOD

This repository contains code and data for the ACL 2023 paper "FutureTOD: Teaching Future Knowledge to Pre-trained Language Model for Task-Oriented Dialogue"

Full version with Appendix: [FutureTOD](https://aclanthology.org/2023.acl-long.360.pdf)

## Abstract

Pre-trained language models based on general text enable huge success in the NLP scenario.
But the intrinsical difference of linguistic patterns between general text and task-oriented dialogues makes existing pre-trained language models less useful in practice. Current dialogue pre-training methods rely on a contrastive framework and face the challenges of both selecting true positives and hard negatives. In this paper, we propose a novel dialogue pretraining model, **FutureTOD**, which distills future knowledge to the representation of the previous dialogue context using a self-training framework. Our intuition is that a good dialogue representation both learns local context information and predicts future information. Extensive experiments on diverse downstream dialogue tasks demonstrate the effectiveness of our model, especially the generalization, robustness, and learning discriminative dialogue representations capabilities.

## Citation
If you use any source codes, pretrained models or datasets included in this repo in your work, please cite the following paper. The bibtex is listed below:
```
@inproceedings{zeng-etal-2023-futuretod,
    title = "{F}uture{TOD}: Teaching Future Knowledge to Pre-trained Language Model for Task-Oriented Dialogue",
    author = "Zeng, Weihao  and
      He, Keqing  and
      Wang, Yejie  and
      Zeng, Chen  and
      Wang, Jingang  and
      Xian, Yunsen  and
      Xu, Weiran",
    booktitle = "Proceedings of the 61st Annual Meeting of the Association for Computational Linguistics (Volume 1: Long Papers)",
    month = jul,
    year = "2023",
    address = "Toronto, Canada",
    publisher = "Association for Computational Linguistics",
    url = "https://aclanthology.org/2023.acl-long.360",
    pages = "6532--6546",
    abstract = "Pre-trained language models based on general text enable huge success in the NLP scenario. But the intrinsical difference of linguistic patterns between general text and task-oriented dialogues makes existing pre-trained language models less useful in practice. Current dialogue pre-training methods rely on a contrastive framework and face the challenges of both selecting true positives and hard negatives. In this paper, we propose a novel dialogue pre-training model, FutureTOD, which distills future knowledge to the representation of the previous dialogue context using a self-training framework. Our intuition is that a good dialogue representation both learns local context information and predicts future information. Extensive experiments on diverse downstream dialogue tasks demonstrate the effectiveness of our model, especially the generalization, robustness, and learning discriminative dialogue representations capabilities.",
}
```


## Pretrained Models

You can easily load the pre-trained model using huggingface Transformers library using the AutoModel function. Following pre-trained versions are supported:

- AndrewZeng/futuretod-base-v1.0: FutureTOD pre-trained using both the MLM and Distill objectives

```
import torch
from transformers import *
tokenizer = AutoTokenizer.from_pretrained("AndrewZeng/futuretod-base-v1.0")
tod_bert = AutoModel.from_pretrained("AndrewZeng/futuretod-base-v1.0")
```

You can also downloaded the pre-trained models from the following links:

- [AndrewZeng/futuretod-base-v1.0](https://huggingface.co/AndrewZeng/futuretod-base-v1.0)

```
model_name_or_path = <path_to_the_downloaded_tod-bert>
model_class, tokenizer_class, config_class = BertModel, BertTokenizer, BertConfig
tokenizer = tokenizer_class.from_pretrained(model_name_or_path)
tod_bert = model_class.from_pretrained(model_name_or_path)
```

## Direct Usage

We used the same data format as [TOD-BERT](https://github.com/jasonwu0731/ToD-BERT). Please refer to the following guide how to use our pre-trained FutureTOD models. Our model is built on top of the PyTorch library and huggingface Transformers library. Let's do a very quick overview of the model architecture and code. Detailed examples for model architecturecan be found in the paper.

```
# Encode text 
input_text = "[CLS] [SYS] Hello, what can I help with you today? [USR] Find me a cheap restaurant nearby the north town."
input_tokens = tokenizer.tokenize(input_text)
story = torch.Tensor(tokenizer.convert_tokens_to_ids(input_tokens)).long()

if len(story.size()) == 1: 
    story = story.unsqueeze(0) # batch size dimension

if torch.cuda.is_available(): 
    tod_bert = tod_bert.cuda()
    story = story.cuda()

with torch.no_grad():
    input_context = {"input_ids": story, "attention_mask": (story > 0).long()}
    hiddens = tod_bert(**input_context)[0]
```

## Pre-training

We will open source the pre-training code as soon as possible.


## Fine-tuning

We will open source the Finetuning and Inference code as soon as possible.

## Report

Feel free to create an issue or send email to the ZengWH@bupt.edu.cn
