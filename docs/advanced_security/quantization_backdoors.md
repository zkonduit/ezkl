# EZKL Security Note: Quantization-Activated Model Backdoors

## Model backdoors and provenance

Machine learning models inherently suffer from robustness issues, which can lead to various
kinds of attacks, from backdoors to evasion attacks. These vulnerabilities are a direct byproductof how machine learning models learn and cannot be remediated.

We say a model has a backdoor whenever a specific attacker-chosen trigger in the input leads
to the model misbehaving. For instance, if we have an image classifier discriminating cats from dogs, the ability to turn any image of a cat into an image classified as a dog by changing a specific pixel pattern constitutes a backdoor.

Backdoors can be introduced using many different vectors. An attacker can introduce a
backdoor using traditional security vulnerabilities. For instance, they could directly alter the file containing model weights or dynamically hack the Python code of the model. In addition, backdoors can be introduced by the training data through a process known as poisoning. In this case, an attacker adds malicious data points to the dataset before the model is trained so that the model learns to associate the backdoor trigger with the intended misbehavior.

All these vectors constitute a whole range of provenance challenges, as any component of an
AI system can virtually be an entrypoint for a backdoor. Although provenance is already a
concern with traditional code, the issue is exacerbated with AI, as retraining a model is
cost-prohibitive. It is thus impractical to translate the “recompile it yourself” thinking to AI.

## Quantization activated backdoors

Backdoors are a generic concern in AI that is outside the scope of EZKL. However, EZKL may
activate a specific subset of backdoors. Several academic papers have demonstrated the
possibility, both in theory and in practice, of implanting undetectable and inactive backdoors in a full precision model that can be reactivated by quantization.

An external attacker may trick the user of an application running EZKL into loading a model
containing a quantization backdoor. This backdoor is active in the resulting model and circuit but not in the full-precision model supplied to EZKL, compromising the integrity of the target application and the resulting proof.

### When is this a concern for me as a user?

Any untrusted component in your AI stack may be a backdoor vector. In practice, the most
sensitive parts include:

- Datasets downloaded from the web or containing crowdsourced data
- Models downloaded from the web even after finetuning
- Untrusted software dependencies (well-known frameworks such as PyTorch can typically
be considered trusted)
- Any component loaded through an unsafe serialization format, such as Pickle.
Because backdoors are inherent to ML and cannot be eliminated, reviewing the provenance of
these sensitive components is especially important.

### Responsibilities of the user and EZKL

As EZKL cannot prevent backdoored models from being used, it is the responsibility of the user to review the provenance of all the components in their AI stack to ensure that no backdoor could have been implanted. EZKL shall not be held responsible for misleading prediction proofs resulting from using a backdoored model or for any harm caused to a system or its users due to a misbehaving model.

### Limitations:

- Attack effectiveness depends on calibration settings and internal rescaling operations.
- Further research needed on backdoor persistence through witness/proof stages.
- Can be mitigated by evaluating the quantized model (using `ezkl gen-witness`), rather than relying on the evaluation of the original model in pytorch or onnx-runtime as difference in evaluation could reveal a backdoor.

References:

1. [Quantization Backdoors to Deep Learning Commercial Frameworks (Ma et al., 2021)](https://arxiv.org/abs/2108.09187)
2. [Planting Undetectable Backdoors in Machine Learning Models (Goldwasser et al., 2022)](https://arxiv.org/abs/2204.06974)
