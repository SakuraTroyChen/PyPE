# PyPE: Pyramid-descent Visual Position Encoding

[![ArXiv](https://img.shields.io/badge/ArXiv-2501.10967-b31b1b.svg?logo=arXiv)](https://arxiv.org/abs/2501.10967) [![License](https://img.shields.io/badge/License-Apache-yellow)](https://github.com/SakuraTroyChen/PyPE/blob/main/LICENSE)

## Updates

- [2025-01-27]: Codes of PyPE released.
- [2025-01-19]: Paper of PyPE online.

## TODO

- [X] Release the code.
- [ ] Release the model checkpoint.
- [ ] Release the training details.

## Getting Started

1. Clone this repository and navigate to PyPE folder.

```
git clone https://github.com/SakuraTroyChen/PyPE.git
cd PyPE
```

2. Install the packages based on the specific type of model you intend to test.

- For LLaVA:

```
cd LLaVA
bash install.sh
```

- For TinyLLaVA:

```
cd TinyLLaVA
bash install.sh
```

## Evaluation

We use [lmms-eval](https://github.com/EvolvingLMMs-Lab/lmms-eval) to evaluate the performance of the models on various tasks. To conduct the evaluations, first install the package:

```
cd PyPE
pip install -e lmms-eval
```

Then you can refer to `PyPE/eval_scripts` and run the scripts you need.

## Citation

If you are interested or inspired by this work, you can cite us by:

```sh
@article{chen2025advancing,
  title={Advancing General Multimodal Capability of Vision-language Models with Pyramid-descent Visual Position Encoding},
  author={Chen, Zhanpeng and Li, Mingxiao and Chen, Ziyang and Du, Nan and Li, Xiaolong and Zou, Yuexian},
  journal={arXiv preprint arXiv:2501.10967},
  year={2025}
}
```

## Related Projects

This research builds significantly upon the foundations established by the following works:

- [LLaVA](https://github.com/haotian-liu/LLaVA): Large Language and Vision Assistant
- [CCA-LLaVA](https://github.com/xing0047/cca-llava): Mitigating Object Hallucination via Concentric Causal Attention
- [TinyLLaVA](https://github.com/TinyLLaVA/TinyLLaVA_Factory): A Framework of Small-scale Large Multimodal Models
- [lmms-eval](https://github.com/EvolvingLMMs-Lab/lmms-eval): The Evaluation Suite of Large Multimodal Models

We extend our gratitude to all the authors for their outstanding contributions to this field.
