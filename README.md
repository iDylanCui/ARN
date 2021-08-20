# Anticipation Reasoning Network

Source codes and datasets for the paper "Incorporating Anticipation Embedding into Reinforcement Learning Framework for Multi-hop Knowledge Graph Question Answering".

![](/Volumes/Hai/code_full/RL_KEQA_release/model_overview.png)

## Train

```
cd /Code/RL_A3C
python main.py --train --dataset=<dataset> --KGE_model=<KGE> --strategy=<strategy>
```

`dataset` is the name of datasets. In our experiments, `dataset` could be `PQ-2H`, `PQ-3H`, `PQ-mix`, `PQL-2H`, `PQL-3H`, `PQL-mix`, `MetaQA-1H`, `MetaQA-2H` or `MetaQA-3H`.

`KGE` is the model of knowledge graph embedding. In our experiments, `KGE` could be `DistMult`, `ComplEx`, `ConvE` or `TuckER`.

`strategy` is the strategy to obtain anticipation embeddings. In our experiments, `strategy` could be `sample`, `avg` or `top1`.

## Test

```
cd /Code/RL_A3C
python main.py --eval --dataset=<dataset>
```

 ## Acknowledgements

We thank a lot for the following outstanding works:

- [Episodic Memory Reader](https://github.com/h19920918/emr)

- [EmbedKGQA](https://github.com/malllabiisc/EmbedKGQA)

- [DacKGR](https://github.com/THU-KEG/DacKGR)

- [PyTorch-A3C](https://github.com/ikostrikov/pytorch-a3c)

  

  

