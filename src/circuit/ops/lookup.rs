//! Custom lookup table support: user-provided PWL (piecewise-linear) mapping via JSON.
//!
//! When `run_args.custom_lookup_path` is set, ONNX Sigmoid is implemented as `LookupOp::Custom { scale, path }`
//! instead of `LookupOp::Sigmoid`. The file at `path` must contain breakpoints, slopes, and intercepts (see README).
//! Table layout reuses the same Halo2 lookup machinery as other LookupOp variants; only the table contents differ.
//!
//! We use a global cache (lazy_static + Mutex) for PWL params so that during prove, multiple layout passes
//! (possibly from different threads) reuse the same loaded data without re-reading the file, avoiding both
//! redundant I/O and thread-local cache misses that could cause hangs.

use super::*;
use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use std::fs;
use std::path::Path;
use std::sync::Mutex;

use crate::{
    circuit::{layouts, table::Range, utils},
    fieldutils::{felt_to_integer_rep, integer_rep_to_felt, IntegerRep},
    tensor::{self, Tensor, TensorError, TensorType},
};

use super::Op;
use halo2curves::ff::PrimeField;
use lazy_static::lazy_static;

/// Piecewise-linear lookup params loaded from JSON at `path`.
/// Format: `{ "breakpoints": [f64], "slopes": [f64], "intercepts": [f64] }`.
/// - `breakpoints`: length n+1, strictly increasing; segments are `[breakpoints[i], breakpoints[i+1])`.
/// - `slopes`, `intercepts`: length n; segment i gives `y = slopes[i]*x + intercepts[i]`.
/// Out-of-range x uses the first/last segment for extrapolation.
#[derive(Clone, Deserialize)]
struct PwlParams {
    breakpoints: Vec<f64>,
    slopes: Vec<f64>,
    intercepts: Vec<f64>,
}

fn load_pwl_from_path(path: &str) -> Result<PwlParams, TensorError> {
    let s = fs::read_to_string(Path::new(path))
        .map_err(|e| TensorError::FileLoadError(format!("custom lookup file '{}': {}", path, e)))?;
    let p: PwlParams = serde_json::from_str(&s)
        .map_err(|e| TensorError::InvalidArgument(format!("custom lookup JSON ({}): {}", path, e)))?;
    if p.breakpoints.len() < 2
        || p.slopes.len() != p.breakpoints.len() - 1
        || p.intercepts.len() != p.breakpoints.len() - 1
    {
        return Err(TensorError::InvalidArgument(
            "custom lookup: require breakpoints (n+1), slopes (n), intercepts (n) with n>=1".to_string(),
        ));
    }
    for i in 1..p.breakpoints.len() {
        if p.breakpoints[i] <= p.breakpoints[i - 1] {
            return Err(TensorError::InvalidArgument(format!(
                "custom lookup: breakpoints must be strictly increasing; got breakpoints[{}]={} <= breakpoints[{}]={}",
                i,
                p.breakpoints[i],
                i - 1,
                p.breakpoints[i - 1]
            )));
        }
    }
    Ok(p)
}

// Global cache for PWL params so that all layout passes (including during prove, possibly on different threads)
// reuse the same data without re-reading the file. A thread-local cache was tried but caused prove to hang
// when layout ran on a worker thread that had an empty cache while the main thread was blocking on rayon.
lazy_static! {
    static ref PWL_CACHE: Mutex<HashMap<String, PwlParams>> = Mutex::new(HashMap::new());
}

fn get_pwl_cached(path: &str) -> Result<PwlParams, TensorError> {
    let mut m = PWL_CACHE
        .lock()
        .map_err(|e| TensorError::InvalidArgument(format!("PWL cache lock: {}", e)))?;
    if let Some(p) = m.get(path) {
        return Ok(p.clone());
    }
    let p = load_pwl_from_path(path)?;
    log::warn!(
        "custom lookup loaded from '{}': ensure this PWL approximates the intended activation (e.g. sigmoid) to avoid soundness issues or proof failure",
        path
    );
    m.insert(path.to_string(), p.clone());
    Ok(p)
}

/// Apply piecewise-linear map: x_float -> y_float using loaded params, then quantize with scale.
/// Fails with a clear error if the input range (in float) extends outside the PWL breakpoints,
/// to avoid unsound extrapolation (e.g. PWL defined on [-5, 5] but model input 10).
fn apply_pwl(
    x: &Tensor<IntegerRep>,
    scale_mult: f64,
    pwl: &PwlParams,
) -> Result<Tensor<IntegerRep>, TensorError> {
    let bp = &pwl.breakpoints;
    let sl = &pwl.slopes;
    let ic = &pwl.intercepts;
    let n = sl.len();
    let bp_min = bp[0];
    let bp_max = bp[n];

    let min_int = x
        .iter()
        .copied()
        .min()
        .ok_or_else(|| TensorError::InvalidArgument("custom lookup: empty input range".to_string()))?;
    let max_int = x
        .iter()
        .copied()
        .max()
        .ok_or_else(|| TensorError::InvalidArgument("custom lookup: empty input range".to_string()))?;
    let min_float = min_int as f64 / scale_mult;
    let max_float = max_int as f64 / scale_mult;
    if min_float < bp_min || max_float > bp_max {
        return Err(TensorError::InvalidArgument(format!(
            "custom lookup: input range [{}, {}] (float) must be within PWL breakpoints [{}, {}]. \
             Either extend breakpoints in your JSON to cover the circuit lookup range, or reduce run_args.lookup_range.",
            min_float, max_float, bp_min, bp_max
        )));
    }

    let res = x.map(|int_val| {
        let x_float = int_val as f64 / scale_mult;
        let y_float = if x_float <= bp[0] {
            sl[0] * x_float + ic[0]
        } else if x_float >= bp[n] {
            sl[n - 1] * x_float + ic[n - 1]
        } else {
            let i = (0..n).find(|&i| x_float >= bp[i] && x_float < bp[i + 1]).unwrap_or(n - 1);
            sl[i] * x_float + ic[i]
        };
        (y_float * scale_mult).round() as IntegerRep
    });
    Ok(res)
}

#[allow(missing_docs)]
/// An enum representing the operations that can be used to express more complex operations via accumulation
#[derive(Clone, Debug, PartialEq, Eq, Hash, PartialOrd, Ord, Deserialize, Serialize)]
pub enum LookupOp {
    Div { denom: utils::F32 },
    IsOdd,
    PowersOfTwo { scale: utils::F32 },
    Ln { scale: utils::F32 },
    Sigmoid { scale: utils::F32 },
    Exp { scale: utils::F32, base: utils::F32 },
    Cos { scale: utils::F32 },
    ACos { scale: utils::F32 },
    Cosh { scale: utils::F32 },
    ACosh { scale: utils::F32 },
    Sin { scale: utils::F32 },
    ASin { scale: utils::F32 },
    Sinh { scale: utils::F32 },
    ASinh { scale: utils::F32 },
    Tan { scale: utils::F32 },
    ATan { scale: utils::F32 },
    Tanh { scale: utils::F32 },
    ATanh { scale: utils::F32 },
    Erf { scale: utils::F32 },
    Pow { scale: utils::F32, a: utils::F32 },
    HardSwish { scale: utils::F32 },
    /// Custom lookup from a JSON file (piecewise-linear mapping). `scale` is the fixed-point scale for
    /// input/output; `path` points to JSON with `breakpoints` (n+1), `slopes` (n), `intercepts` (n).
    /// Same Halo2 lookup constraint as other LookupOp variants; only the table contents are user-defined.
    Custom {
        scale: utils::F32,
        path: String,
    },
}

impl LookupOp {
    /// Returns the range of values that can be represented by the table
    pub fn bit_range(max_len: usize) -> Range {
        let range = (max_len - 1) as f64 / 2_f64;
        let range = range as IntegerRep;
        (-range, range)
    }

    /// as path
    pub fn as_path(&self) -> String {
        match self {
            LookupOp::Pow { scale, a } => format!("pow_{}_{}", scale, a),
            LookupOp::Ln { scale } => format!("ln_{}", scale),
            LookupOp::PowersOfTwo { scale } => format!("pow2_{}", scale),
            LookupOp::IsOdd => "is_odd".to_string(),
            LookupOp::Div { denom } => format!("div_{}", denom),
            LookupOp::Sigmoid { scale } => format!("sigmoid_{}", scale),
            LookupOp::Erf { scale } => format!("erf_{}", scale),
            LookupOp::Exp { scale, base } => format!("exp_{}_{}", scale, base),
            LookupOp::Cos { scale } => format!("cos_{}", scale),
            LookupOp::ACos { scale } => format!("acos_{}", scale),
            LookupOp::Cosh { scale } => format!("cosh_{}", scale),
            LookupOp::ACosh { scale } => format!("acosh_{}", scale),
            LookupOp::Sin { scale } => format!("sin_{}", scale),
            LookupOp::ASin { scale } => format!("asin_{}", scale),
            LookupOp::Sinh { scale } => format!("sinh_{}", scale),
            LookupOp::ASinh { scale } => format!("asinh_{}", scale),
            LookupOp::Tan { scale } => format!("tan_{}", scale),
            LookupOp::ATan { scale } => format!("atan_{}", scale),
            LookupOp::ATanh { scale } => format!("atanh_{}", scale),
            LookupOp::Tanh { scale } => format!("tanh_{}", scale),
            LookupOp::HardSwish { scale } => format!("hardswish_{}", scale),
            LookupOp::Custom { scale, path } => {
                format!("custom_{}_{}", scale, path.replace('/', "_"))
            }
        }
    }

    /// Matches a [Op] to an operation in the `tensor::ops` module.
    pub(crate) fn f<F: PrimeField + TensorType + PartialOrd + std::hash::Hash>(
        &self,
        x: &[Tensor<F>],
    ) -> Result<ForwardResult<F>, TensorError> {
        let x = x[0].clone().map(|x| felt_to_integer_rep(x));
        let res =
            match &self {
                LookupOp::Ln { scale } => {
                    Ok::<_, TensorError>(tensor::ops::nonlinearities::ln(&x, scale.into()))
                }
                LookupOp::PowersOfTwo { scale } => {
                    Ok::<_, TensorError>(tensor::ops::nonlinearities::ipow2(&x, scale.0.into()))
                }
                LookupOp::IsOdd => Ok::<_, TensorError>(tensor::ops::nonlinearities::is_odd(&x)),
                LookupOp::Pow { scale, a } => Ok::<_, TensorError>(
                    tensor::ops::nonlinearities::pow(&x, scale.0.into(), a.0.into()),
                ),
                LookupOp::Div { denom } => Ok::<_, TensorError>(
                    tensor::ops::nonlinearities::const_div(&x, f32::from(*denom).into()),
                ),
                LookupOp::Sigmoid { scale } => {
                    Ok::<_, TensorError>(tensor::ops::nonlinearities::sigmoid(&x, scale.into()))
                }
                LookupOp::Erf { scale } => {
                    Ok::<_, TensorError>(tensor::ops::nonlinearities::erffunc(&x, scale.into()))
                }
                LookupOp::Exp { scale, base } => Ok::<_, TensorError>(
                    tensor::ops::nonlinearities::exp(&x, scale.into(), base.into()),
                ),
                LookupOp::Cos { scale } => {
                    Ok::<_, TensorError>(tensor::ops::nonlinearities::cos(&x, scale.into()))
                }
                LookupOp::ACos { scale } => {
                    Ok::<_, TensorError>(tensor::ops::nonlinearities::acos(&x, scale.into()))
                }
                LookupOp::Cosh { scale } => {
                    Ok::<_, TensorError>(tensor::ops::nonlinearities::cosh(&x, scale.into()))
                }
                LookupOp::ACosh { scale } => {
                    Ok::<_, TensorError>(tensor::ops::nonlinearities::acosh(&x, scale.into()))
                }
                LookupOp::Sin { scale } => {
                    Ok::<_, TensorError>(tensor::ops::nonlinearities::sin(&x, scale.into()))
                }
                LookupOp::ASin { scale } => {
                    Ok::<_, TensorError>(tensor::ops::nonlinearities::asin(&x, scale.into()))
                }
                LookupOp::Sinh { scale } => {
                    Ok::<_, TensorError>(tensor::ops::nonlinearities::sinh(&x, scale.into()))
                }
                LookupOp::ASinh { scale } => {
                    Ok::<_, TensorError>(tensor::ops::nonlinearities::asinh(&x, scale.into()))
                }
                LookupOp::Tan { scale } => {
                    Ok::<_, TensorError>(tensor::ops::nonlinearities::tan(&x, scale.into()))
                }
                LookupOp::ATan { scale } => {
                    Ok::<_, TensorError>(tensor::ops::nonlinearities::atan(&x, scale.into()))
                }
                LookupOp::ATanh { scale } => {
                    Ok::<_, TensorError>(tensor::ops::nonlinearities::atanh(&x, scale.into()))
                }
                LookupOp::Tanh { scale } => {
                    Ok::<_, TensorError>(tensor::ops::nonlinearities::tanh(&x, scale.into()))
                }
                LookupOp::HardSwish { scale } => {
                    Ok::<_, TensorError>(tensor::ops::nonlinearities::hardswish(&x, scale.into()))
                }
                LookupOp::Custom { scale, path } => {
                    let scale_mult: f64 = scale.into();
                    let pwl = get_pwl_cached(path)?;
                    apply_pwl(&x, scale_mult, &pwl)
                }
            }?;

        let output = res.map(|x| integer_rep_to_felt(x));

        Ok(ForwardResult { output })
    }
}

impl<F: PrimeField + TensorType + PartialOrd + std::hash::Hash> Op<F> for LookupOp {
    /// Returns a reference to the Any trait.
    fn as_any(&self) -> &dyn Any {
        self
    }

    /// Returns the name of the operation
    fn as_string(&self) -> String {
        match self {
            LookupOp::Ln { scale } => format!("LN(scale={})", scale),
            LookupOp::PowersOfTwo { scale } => format!("POWERS_OF_TWO(scale={})", scale),
            LookupOp::IsOdd => "IS_ODD".to_string(),
            LookupOp::Pow { a, scale } => format!("POW(scale={}, exponent={})", scale, a),
            LookupOp::Div { denom, .. } => format!("DIV(denom={})", denom),
            LookupOp::Sigmoid { scale } => format!("SIGMOID(scale={})", scale),
            LookupOp::Erf { scale } => format!("ERF(scale={})", scale),
            LookupOp::Exp { scale, base } => format!("EXP(scale={}, base={})", scale, base),
            LookupOp::Tan { scale } => format!("TAN(scale={})", scale),
            LookupOp::ATan { scale } => format!("ATAN(scale={})", scale),
            LookupOp::Tanh { scale } => format!("TANH(scale={})", scale),
            LookupOp::ATanh { scale } => format!("ATANH(scale={})", scale),
            LookupOp::Cos { scale } => format!("COS(scale={})", scale),
            LookupOp::ACos { scale } => format!("ACOS(scale={})", scale),
            LookupOp::Cosh { scale } => format!("COSH(scale={})", scale),
            LookupOp::ACosh { scale } => format!("ACOSH(scale={})", scale),
            LookupOp::Sin { scale } => format!("SIN(scale={})", scale),
            LookupOp::ASin { scale } => format!("ASIN(scale={})", scale),
            LookupOp::Sinh { scale } => format!("SINH(scale={})", scale),
            LookupOp::ASinh { scale } => format!("ASINH(scale={})", scale),
            LookupOp::HardSwish { scale } => format!("HARDSWISH(scale={})", scale),
            LookupOp::Custom { scale, path } => format!("CUSTOM(scale={}, path={})", scale, path),
        }
    }

    fn layout(
        &self,
        config: &mut crate::circuit::BaseConfig<F>,
        region: &mut RegionCtx<F>,
        values: &[&ValTensor<F>],
    ) -> Result<Option<ValTensor<F>>, CircuitError> {
        Ok(Some(layouts::nonlinearity(
            config,
            region,
            values[..].try_into()?,
            self,
        )?))
    }

    /// Returns the scale of the output of the operation.
    fn out_scale(&self, inputs_scale: Vec<crate::Scale>) -> Result<crate::Scale, CircuitError> {
        let scale = inputs_scale[0];
        Ok(scale)
    }

    fn clone_dyn(&self) -> Box<dyn Op<F>> {
        Box::new(self.clone()) // Forward to the derive(Clone) impl
    }
}
