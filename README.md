# Halo2 DL [![Test](https://github.com/jasonmorton/halommrl/workflows/Test/badge.svg)](https://github.com/jasonmorton/halommrl/actions?query=workflow%3ATest)

Model parameters (weights and biases) are either fixed or treated as a private input to the circuit.  The general approach is to use PyTorch or similar for witness generation (recording each activation), producing an _activation trace_.  Quantization happens at inference time, and the activation trace should be i8s. The activation trace is then post-processed (adding supplementary witness such as approximate quotients) to produce a witness (sometimes we may need to do the quantization in the post-processing step).  This witness is then ready to be fed to the prover, which will show that given the weights and quantization scheme, each layer transition is correct. 

We avoid number-to-bits conversions entirely and use native field arithmetic to approximate floating-point arithmetic.  Here, care must be taken with overflow issues.  We can also use limb representations.


Basic idea: first, we apply a linear tensor transformation such as convolution or matrix multiply to transform the tensor (this uses a quadratic constraint for each output, and is usually a large sum), then apply a precomputed fused dropout-batchnorm-quantize-relu lookup table.  The linear transformation may require a larger accumulation number representation such as i16 or i32 (if necessary an i32 can be quantized in-proof to i16 as described in Section \ref{section_WQ} below). The lookup table is one-input one-output, i8 to i8 or i16 to i8 or possibly i32 to i8, and is computed by exhaustively passing all inputs into the PyTorch layer.  Typically only a few tables will be needed as the same transformation is repeated many times.


## Witness Quantization
Suppose we have an i32 $x_{32}$, clip it to i16 range or scale it by dividing by $s_{32} = 2^{31}/2^{15} = 2^{16}$ or some other factor, and round to the nearest i16 to obtain $x_{16}$. If we perform that in witness generation, to avoid a $2^{32}$ lookup table, we would like to use constraints in $\FF_p[x_{32}, x_{16}, ...]$ to express that $x_{16} = {\rm round_{i16}}(x_{32}/s_{32})$.

Rationals of the form $x_{32} / 2^{16}$ have the same range $[-2^{15}, 2^{15}]$ as i16s but have 16 bits of precision. 
For example in the interval [-1,1] the i16s are -1,0,1 but there are $2^{16}$ rationals $x_{32} / 2^{16}$ from $-\frac{2^{16}}{2^{16}}$ to $\frac{2^{16}}{2^{16}}$ evenly spaced in the interval.
Anything in the numerator between 0 and $2^{15}$ inclusive rounds to zero, and between $2^{15}+1$ and $2^{16}$ inclusive rounds to 1.
Thus the numerator error is $s_{32} x_{16} - x_{32}$ and rounding is correct if the absolute value of this difference is less than $2^{15}$.

So in terms of constraints and lookups (with fixed scaling), we have three witnesses: the input $x_{32}$, the claimed output $x_{16}$, and the claimed error $e$. We require $e = s_{32} x_{16} - x_{32}$ and require $e,x_{16} \in [-2^{15},2^{15}]$ with lookups. 

We apply this only after a linear accumulation operation, so that we do not need to check that $x_{32} \in [-2^{31},2^{31}]$ because this can be proved at compile time from the fact that the input activations and weights are $i8$ and there are not too many of them (e.g. a dot product of two length-$2^8$ vectors of $i8$s can be at most $2^{22}$, and in most cases will likely be smaller than $2^{15}$).



## Witness Clip
The clip(m) operation sets $x_{out}$ to $x_{in}$ if $x_{in} \in [-m,m]$ and to -m or m if smaller (larger).  The prover should provide, as well as $x_{in}$, the proposed $x_{out}$, and a value $c$ which is claimed to be  $c = 1$ if we clip to $m$, $0$ if no clipping occurs, or  $-1$ if we clip to $-m$.  So we have $c(c-1)(c+1) = 0$, $x_{out} = x_{in} - c*(x_{in} - m)$, $x_{out} \in [-m,m]$.


This could be done with a lookup table if we also have a larger max, i.e. numbers 0..m are left alone, m..M are set to m, and values above $M$ cause the proof to fail.
