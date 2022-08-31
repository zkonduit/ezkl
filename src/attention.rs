use halo2_proofs::{arithmetic::FieldExt, circuit::Value, plonk::Assigned};
use std::marker::PhantomData;

use crate::fieldutils::i32tofelt;
use crate::tensorutils::{map2, map3, map4};

pub type AttentionProjection<F> = Vec<Vec<Vec<Vec<Value<Assigned<F>>>>>>;

#[derive(Clone)]
pub struct Attention<
    F: FieldExt,
    Inner,
    const SEQLEN: usize,
    const OUT: usize,
    const NUM_HEADS: usize,
    const QKV_DIM: usize,
> {
    pub input: Vec<Vec<Inner>>, //  SEQLEN
    pub output: Vec<Vec<Inner>>,
    // Q,K,V weight matrices
    pub weights: Vec<Vec<Inner>>, // OUT x SEQLEN
    pub _marker: PhantomData<F>,
}

impl<
        F: FieldExt,
        Inner,
        const SEQLEN: usize,
        const EMBED: usize,
        const NUM_HEADS: usize,
        const QKV_DIM: usize,
    > Attention<F, Inner, SEQLEN, EMBED, NUM_HEADS, QKV_DIM>
{
    pub fn fill<Func1, Func2>(mut f: Func1, w: Func2) -> Self
    where
        Func1: FnMut(usize, usize) -> Inner,
        Func2: FnMut(usize, usize) -> Inner,
        [(); 3 * NUM_HEADS * QKV_DIM]:,
    {
        Attention {
            input: map2::<_, _, SEQLEN, EMBED>(&mut f),
            output: map2::<_, _, EMBED, SEQLEN>(f),
            //  we only need one weight matrix !
            weights: map2::<_, _, { 3 * NUM_HEADS * QKV_DIM }, EMBED>(w),

            _marker: PhantomData,
        }
    }
    pub fn without_witnesses() -> Attention<F, Value<Assigned<F>>, SEQLEN, EMBED, NUM_HEADS, QKV_DIM>
    where
        [(); 3 * NUM_HEADS * QKV_DIM]:,
    {
        Attention::<F, Value<Assigned<F>>, SEQLEN, EMBED, NUM_HEADS, QKV_DIM>::fill(
            |_, _| Value::default(),
            |_, _| Value::default(),
        )
    }
}

impl<
        F: FieldExt,
        const SEQLEN: usize,
        const EMBED: usize,
        const NUM_HEADS: usize,
        const QKV_DIM: usize,
    > Attention<F, Value<Assigned<F>>, SEQLEN, EMBED, NUM_HEADS, QKV_DIM>
{
    /// Assign parameters, leaving input and output as unknown Values.
    pub fn from_parameters(weights: Vec<Vec<i32>>) -> Self
    where
        [(); 3 * NUM_HEADS * QKV_DIM]:,
    {
        let weights: Vec<Vec<Value<Assigned<F>>>> =
            map2::<_, _, { 3 * NUM_HEADS * QKV_DIM }, EMBED>(|i, j| {
                Value::known(i32tofelt::<F>(weights[i][j]).into())
            });

        let input: Vec<Vec<Value<Assigned<F>>>> =
            map2::<_, _, SEQLEN, EMBED>(|_, _| Value::default());
        let output: Vec<Vec<Value<Assigned<F>>>> =
            map2::<_, _, SEQLEN, EMBED>(|_, _| Value::default());

        Attention {
            input,
            output,
            weights,
            _marker: PhantomData,
        }
    }

    pub fn _self_attention_projection(
        &mut self,
        // input is sequence length SEQLEN by embedding size EMBED
        input: Vec<Vec<Value<Assigned<F>>>>,
    ) -> AttentionProjection<F> {
        self.input = input.clone();

        let mut output: AttentionProjection<F> =
            map4::<_, _, 3, NUM_HEADS, SEQLEN, QKV_DIM>(|_, _, _, _| {
                Value::known(F::zero().into())
            });

        for (p, qkv) in output.iter_mut().enumerate().take(3) {
            for (i, head) in qkv.iter_mut().enumerate().take(NUM_HEADS) {
                for (j, x) in input.iter().enumerate().take(SEQLEN) {
                    for k in 0..QKV_DIM {
                        for x_l in x.iter().take(EMBED) {
                            head[j][k] = head[j][k] + self.weights[p * i * k][k] * x_l;
                        }
                    }
                }
            }
        }
        output
    }

    pub fn scaled_dot_product(
        &mut self,
        query: Vec<Vec<Vec<Value<Assigned<F>>>>>,
        key: Vec<Vec<Vec<Value<Assigned<F>>>>>,
        value: Vec<Vec<Vec<Value<Assigned<F>>>>>,
    ) -> Vec<Vec<Vec<Value<Assigned<F>>>>> {
        let mut alogits: Vec<Vec<Vec<Value<Assigned<F>>>>> =
            map3::<_, _, NUM_HEADS, SEQLEN, SEQLEN>(|_, _, _| Value::known(F::zero().into()));

        let normalizer = F::sqrt_ratio(&F::one(), &F::from_u128(QKV_DIM as u128)).1;
        // equivalent to Q @ K.T
        for i in 0..NUM_HEADS {
            for j in 0..SEQLEN {
                for k in 0..SEQLEN {
                    for l in 0..QKV_DIM {
                        alogits[i][j][k] =
                            alogits[i][j][k] + query[i][j][l] * key[i][k][l] * normalizer
                    }
                }
            }
        }
        let mut output: Vec<Vec<Vec<Value<Assigned<F>>>>> =
            map3::<_, _, NUM_HEADS, SEQLEN, QKV_DIM>(|_, _, _| Value::known(F::zero().into()));
        // equivalent to Q @ K.T
        for i in 0..NUM_HEADS {
            for j in 0..SEQLEN {
                for k in 0..QKV_DIM {
                    for l in 0..SEQLEN {
                        output[i][j][k] = output[i][j][k] + alogits[i][j][l] * value[i][l][k]
                    }
                }
            }
        }
        output
    }

    /// Take a layer with set parameters, accept an input, perform forward pass, and return output.
    /// Mutates self to assign the input and computed output.
    pub fn forward(
        &mut self,
        input: Vec<Vec<Value<Assigned<F>>>>,
    ) -> Vec<Vec<Vec<Value<Assigned<F>>>>> {
        self.input = input.clone();

        let qkv = self._self_attention_projection(input);
        self.scaled_dot_product(qkv[0].clone(), qkv[1].clone(), qkv[2].clone())
    }
}
