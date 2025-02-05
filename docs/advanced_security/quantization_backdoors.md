# EZKL Security Note: Quantization-Induced Model Backdoors

> Note: this only affects a situation where a party separate to an application's developer has access to the model's weights and can modify them. This is a common scenario in adversarial machine learning research, but can be less common in real-world applications. If you're building your models in house and deploying them yourself, this is less of a concern. If you're building a permisionless system where anyone can submit models, this is more of a concern.

Models processed through EZKL's quantization step can harbor backdoors that are dormant in the original full-precision model but activate during quantization. These backdoors force specific outputs when triggered, with impact varying by application.

Key Factors:

- Larger models increase attack feasibility through more parameter capacity
- Smaller quantization scales facilitate attacks by allowing greater weight modifications
- Rebase ratio of 1 enables exploitation of convolutional layer consistency

Limitations: 

- Attack effectiveness depends on calibration settings and internal rescaling operations. 
- Further research needed on backdoor persistence through witness/proof stages.
- Can be mitigated by evaluating the quantized model (using `ezkl gen-witness`), rather than relying on the evaluation of the original model. 

References:

1. [Quantization Backdoors to Deep Learning Commercial Frameworks (Ma et al., 2021)](https://arxiv.org/abs/2108.09187)
2. [Planting Undetectable Backdoors in Machine Learning Models (Goldwasser et al., 2022)](https://arxiv.org/abs/2204.06974)
