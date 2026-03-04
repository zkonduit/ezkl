# PLA Sigmoid: Replace Lookup with Piecewise Linear (Copy/Arithmetic Constraints)

This example shows how to **avoid Sigmoid lookup tables** in ezkl by using a **piecewise linear approximation (PLA)** that uses only multiplications, additions, and comparisons. The circuit then relies on **copy and arithmetic constraints** instead of lookups, which can reduce proof cost and simplify settings.

## Why this helps

- **Sigmoid** is often implemented in ezkl via a **lookup table**, which can be expensive in circuit size and proving time.
- Replacing it with a **PLA** (segment-wise linear: `y = slope_i * x + intercept_i` per segment) gives:
  - **No lookup**: only mul/add/compare ops → no `lookup_range` or large lookup tables.
  - **Lighter settings**: e.g. `num_inner_cols=1`, `lookup_range=(0,1)`, `bounded_log_lookup=False`.
  - **Faster proving** in practice when the rest of the circuit is comparable.

## Accuracy

- The PLA is fitted so that **max relative error vs true sigmoid ≤ 0.5%** (i.e. **≥ 99.5% accuracy**) over the chosen input range (default `[-2, 2]`).
- You can recalibrate with `python pwl_sigmoid.py` (writes `pwl_params.npz`).

## Files

| File | Description |
|------|-------------|
| `pwl_sigmoid.py` | PLA fitting and `PWLSigmoid` PyTorch module (mul/add/compare only). |
| `gen.py` | Builds the model (conv + PLA Sigmoid), exports ONNX and `input.json`. |
| `network.onnx` | Exported model (no Sigmoid op; PLA is expanded into linear ops). |
| `input.json` | Sample inputs and shapes for ezkl. |
| `pwl_params.npz` | Pre-fitted PLA parameters (optional; `gen.py` can regenerate). |

## How to run

1. **Export ONNX and input (if not already present):**
   ```bash
   cd examples/onnx/pla_sigmoid
   pip install torch numpy
   python gen.py
   ```
   This produces `network.onnx` and `input.json` (and optionally `pwl_params.npz`).

2. **Use with ezkl** (no lookup needed; lightweight settings):
   ```python
   import ezkl
   py_run_args = ezkl.PyRunArgs()
   py_run_args.input_visibility = "public"
   py_run_args.output_visibility = "public"
   py_run_args.param_visibility = "fixed"
   py_run_args.num_inner_cols = 1
   py_run_args.lookup_range = (0, 1)
   py_run_args.bounded_log_lookup = False
   py_run_args.logrows = 16  # or as required by your circuit size
   ezkl.gen_settings("network.onnx", "settings.json", py_run_args=py_run_args)
   ezkl.compile_circuit("network.onnx", "network.compiled", "settings.json")
   # Then: SRS, setup, gen_witness, prove as in the main ezkl docs.
   ```

## Author

[changshenhan](https://github.com/changshenhan) — PLA-on-Sigmoid example for ezkl.
