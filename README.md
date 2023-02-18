# Event Detection on Dynamic Graphs

This is a PyTorch implementation of DyGED (Dynamic Graph Event Detection) that is proposed event detection architecture in our paper:
<br/>
> Event Detection on Dynamic Graphs.<br>
> Mert Kosan, Arlei Silva, Sourav Medya, Brian Uzzi, Ambuj Singh.<br>
> Deep Learning on Graphs: Method and Applications, Association for the Advancement of Artificial Intelligence 2023 (DLG-AAAI’23). <br>
> Workshop version: https://drive.google.com/file/d/1ijb9ngmi-yaNAmtRi_9Bc7B9INn8Q_Ir/view <br>
> Longer version: https://arxiv.org/abs/2110.12148

### DLG-AAAI'23 Poster

![](DLG-AAAI23%20Poster.png)

[PDF version](DLG-AAAI23%20Poster.pdf)

### Data

We share Twitter Weather and NYC Cab datasets in [DyGED_data.zip](DyGED_data.zip). Hedge Fund data unfortunately cannot be publicized.
We also provide [data_utils.py](source/data_utils.py) which process the graphs into PyTorch Geometric format.
Please contact us if any issue encountered.

### Implementation

We share our framework in [models.py](source/models.py), but it does not support PyTorch geometric, but raw data. We hope to implement PyTorch Geometric version soon.

### Citing

If you find our framework or data useful, please consider citing the following paper:

```
@article{kosan2021event,
  title={Event detection on dynamic graphs},
  author={Kosan, Mert and Silva, Arlei and Medya, Sourav and Uzzi, Brian and Singh, Ambuj},
  journal={arXiv preprint arXiv:2110.12148},
  year={2021}
}
```



