# Custom Lookup Table

When your ONNX model uses **Sigmoid** (or other activations that use a built-in lookup table), you can optionally replace the default table with a **user-defined piecewise-linear (PWL)** lookup table. This is done by setting `custom_lookup_path` (Python: `py_run_args.custom_lookup_path`, CLI: `--custom-lookup-path`) to the path of a JSON file that describes the PWL mapping.

## How to produce the lookup table

The JSON file must contain three arrays:

| Field         | Length | Description |
|---------------|--------|-------------|
| `breakpoints` | n+1    | Strictly increasing x-values. Segment *i* is the interval `[breakpoints[i], breakpoints[i+1])`. |
| `slopes`      | n      | For segment *i*, the map is `y = slopes[i] * x + intercepts[i]`. |
| `intercepts`  | n      | Intercept for each segment. |

**Steps to generate the file:**

1. Choose breakpoints that cover the range of inputs your activation will see (e.g. for sigmoid, a symmetric range like `[-6, 6]`).
2. Evaluate the target function (e.g. `sigmoid(x) = 1/(1+exp(-x))`) at each breakpoint.
3. For each segment `[a, b]`:
   - `slope = (f(b) - f(a)) / (b - a)`
   - `intercept = f(a) - slope * a`
4. Write `breakpoints`, `slopes`, and `intercepts` to a JSON file.

An example for a 4-segment sigmoid approximation is in `examples/pwl_sigmoid_example.json`. The notebook `examples/notebooks/custom_lookup_demo.ipynb` shows how to generate this in Python and run the full EZKL pipeline with a custom lookup.

## Caveats and soundness

- **Input must be within the defined breakpoints.** The circuit **enforces** this: if the lookup range (from `run_args.lookup_range`, converted to float using the op scale) extends outside `[breakpoints[0], breakpoints[n]]`, EZKL returns a clear error. For example, if your PWL is defined only on `[-5, 5]` but the model or calibration produces inputs equivalent to 10, you must either extend the breakpoints in your JSON to cover that range or reduce `lookup_range` so that the float range stays within the breakpoints.
- **Correctness:** The PWL table is used both for the circuit (constraint system) and for witness generation. If the table does not match the function your model was trained with (e.g. a poor approximation of sigmoid), the committed output may differ from the true model output, which can lead to failed verification or unsound claims.
- **Error handling:** Invalid JSON, non-increasing breakpoints, mismatched array lengths, or input range outside breakpoints produce clear errors (see `src/circuit/ops/lookup.rs`). The loader validates structure and reports soundness-related warnings where applicable.

## Usage

- **Python:** Set `py_run_args.custom_lookup_path = "/path/to/pwl.json"` (absolute path recommended) before calling `ezkl.gen_settings(...)`. The path is stored in the generated settings and used by `compile_circuit`, `gen_witness`, and proving.
- **CLI:** Pass `--custom-lookup-path /path/to/pwl.json` when running `gen-settings`. Use the same settings file for `calibrate-settings`, `compile-circuit`, and `prove`.
- **Production / safety margin:** If you set `lookup_range` explicitly (e.g. to leave headroom for runtime values), the circuit’s lookup range must still lie **within** the PWL breakpoints. Build your PWL over an extended range (e.g. `[x_min - margin, x_max + margin]`) so that the table covers the full lookup range; then set `lookup_range` to match. See the notebook’s “Production tip” for a pipeline that extends the PWL by a margin (e.g. build breakpoints over `[x_min - margin, x_max + margin]` and set `lookup_range` to match).
