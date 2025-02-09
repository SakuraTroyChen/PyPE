# PyPE: Pyramid-descent Visual Position Encoding

## Getting Started

Install the packages based on the specific type of model you intend to test.

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

## Related Projects

This research builds significantly upon the foundations established by the following works:

- [LLaVA](https://github.com/haotian-liu/LLaVA): Large Language and Vision Assistant
- [CCA-LLaVA](https://github.com/xing0047/cca-llava): Mitigating Object Hallucination via Concentric Causal Attention
- [TinyLLaVA](https://github.com/TinyLLaVA/TinyLLaVA_Factory): A Framework of Small-scale Large Multimodal Models
- [lmms-eval](https://github.com/EvolvingLMMs-Lab/lmms-eval): The Evaluation Suite of Large Multimodal Models

We extend our gratitude to all the authors for their outstanding contributions to this field.
