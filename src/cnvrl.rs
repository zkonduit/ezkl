use halo2_proofs::{
    arithmetic::FieldExt,
    circuit::{AssignedCell, Layouter, SimpleFloorPlanner, Value},
    plonk::{
        create_proof, keygen_pk, keygen_vk, verify_proof, Advice, Assigned, Circuit, Column,
        ConstraintSystem, Constraints, Error, Expression, Instance, Selector, SingleVerifier,
        TableColumn,
    },
    poly::{commitment::Params, Rotation},
    transcript::{Blake2bRead, Blake2bWrite, Challenge255},
};
use pasta_curves::{pallas, vesta};
use rand::rngs::OsRng;
use std::marker::PhantomData;

use crate::tensorutils::{dot3, flatten3, flatten4, map3, map3r, map4, map4r};

// The linear part of a 2D convolution layer.
// Compile-time constants are:
// IH: Height in pixels of the input image
// IW: Width in pixels of the input image
// CHIN: Number of input channels
// CHOUT: Number of output channels = number of filters
// KH: Kernel height in pixels
// KW: Kernel width in pixels
// BITS: Bits in the activation & weight representations (e.g. 8 for i8)
// For now we only support stride of 1 and no padding.  Thus the ouput shape is
// (IH - KH+1) pixels high and (IW-KW+1) pixels wide
// PyTorch: https://pytorch.org/docs/stable/generated/torch.nn.Conv2d.html
// PyTorch Conv2d activations are (N, CHIN, H, W), and we are doing single-input
// inference so batchsize N=1.
// We may add support for different types of padding and stride, and possibly dilation, groups, and bias.

#[derive(Clone)]
struct Conv2dAdvice<
    F: FieldExt,
    const IH: usize,
    const IW: usize,
    const CHIN: usize,
    const CHOUT: usize,
    const KH: usize,
    const KW: usize,
    const OH: usize,
    const OW: usize,
    const BITS: usize,
> {
    kernel: Vec<Vec<Vec<Vec<Column<Advice>>>>>, // CHOUT x CHIN x KH x KW
    input: Vec<Vec<Vec<Column<Advice>>>>,       // CHIN x IH x IW
    lin_output: Vec<Vec<Vec<Column<Advice>>>>,  // CHOUT x OH x OW
    //    nl_output: Vec<Vec<Vec<Column<Advice>>>>,   // CHOUT x OH x OW
    _marker: PhantomData<F>,
}

#[derive(Clone)]
struct Conv2dExpression<
    F: FieldExt,
    const IH: usize,
    const IW: usize,
    const CHIN: usize,
    const CHOUT: usize,
    const KH: usize,
    const KW: usize,
    const OH: usize,
    const OW: usize,
    const BITS: usize,
> {
    kernel: Vec<Vec<Vec<Vec<Expression<F>>>>>, // CHOUT x CHIN x KH x KW
    input: Vec<Vec<Vec<Expression<F>>>>,       // CHIN x IH x IW
    lin_output: Vec<Vec<Vec<Expression<F>>>>,  // CHOUT x OH x OW
                                               //    nl_output: Vec<Vec<Vec<Column<Advice>>>>,   // CHOUT x OH x OW
}

#[derive(Clone)]
struct Conv2dAssigned<
    F: FieldExt,
    const IH: usize,
    const IW: usize,
    const CHIN: usize,
    const CHOUT: usize,
    const KH: usize,
    const KW: usize,
    const OH: usize,
    const OW: usize,
    const BITS: usize,
> {
    kernel: Vec<Vec<Vec<Vec<Value<Assigned<F>>>>>>, // CHOUT x CHIN x KH x KW
    input: Vec<Vec<Vec<Value<Assigned<F>>>>>,       // CHIN x IH x IW
    lin_output: Vec<Vec<Vec<Value<Assigned<F>>>>>,  // CHOUT x OH x OW
                                                    //    nl_output: Vec<Vec<Vec<Column<Advice>>>>,   // CHOUT x OH x OW
}

impl<
        F: FieldExt,
        const IH: usize,
        const IW: usize,
        const CHIN: usize,
        const CHOUT: usize,
        const KH: usize,
        const KW: usize,
        const OH: usize,
        const OW: usize,
        const BITS: usize,
    > Conv2dAssigned<F, IH, IW, CHIN, CHOUT, KH, KW, OH, OW, BITS>
{
    fn without_witnesses() -> Self {
        let kernel: Vec<Vec<Vec<Vec<Value<Assigned<F>>>>>> =
            map4::<_, _, CHOUT, CHIN, KH, KW>(|i, j, k, l| Value::default());
        let input: Vec<Vec<Vec<Value<Assigned<F>>>>> =
            map3::<_, _, CHIN, IH, IW>(|i, j, k| Value::default());
        let lin_output: Vec<Vec<Vec<Value<Assigned<F>>>>> =
            map3::<_, _, CHOUT, OH, OW>(|i, j, k| Value::default());
        Self {
            kernel,
            input,
            lin_output,
        }
    }

    fn from_values<T>(
        kernel: Vec<Vec<Vec<Vec<T>>>>,
        input: Vec<Vec<Vec<T>>>,
        lin_output: Vec<Vec<Vec<T>>>,
    ) -> Self
    where
        T: Into<F> + Copy,
    {
        let kernel = map4::<_, _, CHOUT, CHIN, KH, KW>(|i, j, k, l| {
            Value::known(kernel[i][j][k][l].into().into())
        });
        let input =
            map3::<_, _, CHIN, IH, IW>(|i, j, k| Value::known(input[i][j][k].into().into()));
        let lin_output =
            map3::<_, _, CHOUT, OH, OW>(|i, j, k| Value::known(lin_output[i][j][k].into().into()));

        Self {
            kernel,
            input,
            lin_output,
        }
    }
}

#[derive(Clone)]
struct Conv2dConfig<
    F: FieldExt,
    const IH: usize,
    const IW: usize,
    const CHIN: usize,
    const CHOUT: usize,
    const KH: usize,
    const KW: usize,
    const OH: usize, //= (IH - KH + 1); //not supported yet in rust
    const OW: usize, //= (IW - KW + 1); //not supported yet in rust
    const BITS: usize,
> {
    advice: Conv2dAdvice<F, IH, IW, CHIN, CHOUT, KH, KW, OH, OW, BITS>,
    // relu_i_col: TableColumn,
    // relu_o_col: TableColumn,
    q: Selector,
    _marker: PhantomData<F>,
}

impl<
        F: FieldExt,
        const IH: usize,
        const IW: usize,
        const CHIN: usize,
        const CHOUT: usize,
        const KH: usize,
        const KW: usize,
        const OH: usize, //= (IH - KH + 1); //not supported yet in rust
        const OW: usize, //= (IW - KW + 1); //not supported yet in rust
        const BITS: usize,
    > Conv2dConfig<F, IH, IW, CHIN, CHOUT, KH, KW, OH, OW, BITS>
{
    fn define_advice(
        cs: &mut ConstraintSystem<F>,
    ) -> Conv2dAdvice<F, IH, IW, CHIN, CHOUT, KH, KW, OH, OW, BITS> {
        let kernel_advice: Vec<Vec<Vec<Vec<Column<Advice>>>>> =
            map4::<_, _, CHOUT, CHIN, KH, KW>(|i, j, k, l| cs.advice_column());
        let input_advice: Vec<Vec<Vec<Column<Advice>>>> =
            map3::<_, _, CHIN, IH, IW>(|i, j, k| cs.advice_column());
        let lin_output_advice: Vec<Vec<Vec<Column<Advice>>>> =
            map3::<_, _, CHOUT, OH, OW>(|i, j, k| cs.advice_column());
        // let nl_output_advice: Vec<Vec<Vec<Column<Advice>>>> =
        //     map3::<_, _, CHOUT, OH, OW>(|i, j, k| cs.advice_column());
        Conv2dAdvice {
            kernel: kernel_advice,
            input: input_advice,
            lin_output: lin_output_advice,
            //            nl_output: nl_output_advice,
            _marker: PhantomData,
        }
    }

    fn configure(
        cs: &mut ConstraintSystem<F>,
    ) -> Conv2dConfig<F, IH, IW, CHIN, CHOUT, KH, KW, OH, OW, BITS> {
        let advice = Self::define_advice(cs);
        let qs = cs.selector();

        cs.create_gate("cnvmul", |virtual_cells| {
            // 'allocate' all the advice and selector cols by querying the cs with the labels
            let q = virtual_cells.query_selector(qs);

            let kernel_ex: Vec<Vec<Vec<Vec<Expression<F>>>>> =
                map4::<_, _, CHOUT, CHIN, KH, KW>(|i, j, k, l| {
                    virtual_cells.query_advice(advice.kernel[i][j][k][l], Rotation::cur())
                });
            let input_ex: Vec<Vec<Vec<Expression<F>>>> = map3::<_, _, CHIN, IH, IW>(|i, j, k| {
                virtual_cells.query_advice(advice.lin_output[i][j][k], Rotation::cur())
            });
            let lin_output_ex: Vec<Vec<Vec<Expression<F>>>> =
                map3::<_, _, CHOUT, OH, OW>(|i, j, k| {
                    virtual_cells.query_advice(advice.lin_output[i][j][k], Rotation::cur())
                });
            // let nl_output_ex: Vec<Vec<Vec<Expression<F>>>> =
            //     map3::<_, _, CHOUT, OH, OW>(|i, j, k| {
            //         virtual_cells.query_advice(advice.nl_output[i][j][k], Rotation::cur())
            //     });

            // We put the negation -lin_output_ex in constraint tensor.
            let mut constraints: Vec<Vec<Vec<Expression<F>>>> =
                map3::<_, _, CHOUT, OH, OW>(|i, j, k| -lin_output_ex[i][j][k].clone());

            // Now we compute the convolution expression, collect it in a CHOUT x OH x OW tensor, and add it to constraints
            for filter in 0..CHOUT {
                let kernel = kernel_ex[filter].clone(); //CHIN x KH x KW
                for row in 0..OH {
                    for col in 0..OW {
                        //slice input to patch of kernel shape at this location
                        let patch = &input_ex[0..CHIN][row..(row + KH)][col..(col + KW)];
                        let conv2d_ex = dot3(&patch.to_vec(), &kernel);
                        constraints[filter][row][col] =
                            constraints[filter][row][col].clone() + conv2d_ex;
                    }
                }
            }

            let constraints = (0..(CHOUT * OH * OW))
                .map(|j| "c")
                .zip(flatten3(constraints));
            Constraints::with_selector(q, constraints)
        });

        // let mut pub_col: Vec<Column<Instance>> = Vec::new();
        // for _i in 0..NROWS {
        //     pub_col.push(cs.instance_column());
        // }

        // for i in 0..NROWS {
        //     cs.enable_equality(pub_col[i]);
        // }

        // let relu_i_col = cs.lookup_table_column();
        // let relu_o_col = cs.lookup_table_column();

        // for i in 0..NROWS {
        //     let _ = cs.lookup(|cs| {
        //         vec![
        //             (cs.query_advice(uadv[i], Rotation::cur()), relu_i_col),
        //             (cs.query_advice(radv[i], Rotation::cur()), relu_o_col),
        //         ]
        //     });
        // }

        Self {
            advice,
            q: qs,
            // relu_i_col,
            // relu_o_col,
            // pub_col,
            _marker: PhantomData,
        }
    }

    // // Allocates all legal input-output tuples for the ReLu function in the first 2^16 rows
    // // of the constraint system.
    // fn alloc_table(&self, layouter: &mut impl Layouter<F>) -> Result<(), Error> {
    //     layouter.assign_table(
    //         || "relu table",
    //         |mut table| {
    //             let mut row_offset = 0;
    //             let shift = F::from(127);
    //             for input in 0..255 {
    //                 table.assign_cell(
    //                     || format!("relu_i_col row {}", row_offset),
    //                     self.relu_i_col,
    //                     row_offset,
    //                     || Value::known(F::from(input) - shift), //-127..127
    //                 )?;
    //                 table.assign_cell(
    //                     || format!("relu_o_col row {}", row_offset),
    //                     self.relu_o_col,
    //                     row_offset,
    //                     || Value::known(F::from(if input < 127 { 127 } else { input }) - shift),
    //                 )?;
    //                 row_offset += 1;
    //             }
    //             Ok(())
    //         },
    //     )
    // }

    // Allocates `a` (private input) and `c` (public copy of output)
    // fn alloc_private_and_public_inputs(
    //     &self,
    //     layouter: &mut impl Layouter<F>,
    //     a: Value<Assigned<F>>,
    //     c: Value<Assigned<F>>,
    // ) -> Result<AssignedCell<Assigned<F>, F>, Error> {
    //     layouter.assign_region(
    //         || "private and public inputs",
    //         |mut region| {
    //             let row_offset = 0;
    //             region.assign_advice(|| "private input `a`", self.i_col, row_offset, || a)?;
    //             let c =
    //                 region.assign_advice(|| "public input `c`", self.o_col, row_offset, || c)?;
    //             Ok(c)
    //         },
    //     )
    // }
}

//#[derive(Default)]
struct Cnv2dCircuit<
    F: FieldExt,
    const IH: usize,
    const IW: usize,
    const CHIN: usize,
    const CHOUT: usize,
    const KH: usize,
    const KW: usize,
    const OH: usize, //= (IH - KH + 1); //not supported yet in rust
    const OW: usize, //= (IW - KW + 1); //not supported yet in rust
    const BITS: usize,
> {
    // circuit holds Values
    assigned: Conv2dAssigned<F, IH, IW, CHIN, CHOUT, KH, KW, OH, OW, BITS>,
    // Public input (from prover).
    //
}
//impl<F: FieldExt, const RANGE: usize> MvmulCircuit<F, RANGE> {}

impl<
        F: FieldExt,
        const IH: usize,
        const IW: usize,
        const CHIN: usize,
        const CHOUT: usize,
        const KH: usize,
        const KW: usize,
        const OH: usize, //= (IH - KH + 1); //not supported yet in rust
        const OW: usize, //= (IW - KW + 1); //not supported yet in rust
        const BITS: usize,
    > Circuit<F> for Cnv2dCircuit<F, IH, IW, CHIN, CHOUT, KH, KW, OH, OW, BITS>
{
    type Config = Conv2dConfig<F, IH, IW, CHIN, CHOUT, KH, KW, OH, OW, BITS>;
    type FloorPlanner = SimpleFloorPlanner;

    fn without_witnesses(&self) -> Self {
        let assigned = Conv2dAssigned::without_witnesses();
        Cnv2dCircuit { assigned }
    }

    // define the constraints, mutate the provided ConstraintSystem, and output the resulting FrameType
    fn configure(cs: &mut ConstraintSystem<F>) -> Self::Config {
        Self::Config::configure(cs)
    }

    fn synthesize(
        &self,
        config: Self::Config,
        mut layouter: impl Layouter<F>, // layouter is our 'write buffer' for the circuit
    ) -> Result<(), Error> {
        // mvmul
        //        let mut arr = Vec::new();

        layouter.assign_region(
            || "Assign values", // the name of the region
            |mut region| {
                let offset = 0;

                config.q.enable(&mut region, offset)?;
                // let kernel_res: Vec<Vec<Vec<Vec<()>>>> =
                map4r::<_, _, halo2_proofs::plonk::Error, CHOUT, CHIN, KH, KW>(|i, j, k, l| {
                    region.assign_advice(
                        || format!("kr_{i}_{j}_{k}_{l}"),
                        config.advice.kernel[i][j][k][l], // Column<Advice>
                        offset,
                        || self.assigned.kernel[i][j][k][l], //Assigned<F>
                    )
                })?;

                //                let input_res: Vec<Vec<Vec<_>>> =
                map3r::<_, _, halo2_proofs::plonk::Error, CHIN, IH, IW>(|i, j, k| {
                    region.assign_advice(
                        || format!("in_{i}_{j}_{k}"),
                        config.advice.input[i][j][k], // Column<Advice>
                        offset,
                        || self.assigned.input[i][j][k], //Assigned<F>
                    )
                })?;

                map3r::<_, _, halo2_proofs::plonk::Error, CHOUT, OH, OW>(|i, j, k| {
                    region.assign_advice(
                        || format!("out_{i}_{j}_{k}"),
                        config.advice.lin_output[i][j][k], // Column<Advice>
                        offset,
                        || self.assigned.lin_output[i][j][k], //Assigned<F>
                    )
                })?;

                Ok(())
            },
        )?;

        //        config.alloc_table(&mut layouter)?;

        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use halo2_proofs::{
        dev::{FailureLocation, MockProver, VerifyFailure},
        pasta::Fp as F,
        plonk::{Any, Circuit},
    };
    //     use nalgebra;
    use std::time::{Duration, Instant};

    #[test]
    fn test_cnv2d_succeed() {
        let k = 9; //2^k rows
        let input = vec![
            vec![
                vec![0u64, 1u64, 2u64],
                vec![3u64, 4u64, 5u64],
                vec![6u64, 7u64, 8u64],
            ],
            vec![
                vec![1u64, 2u64, 3u64],
                vec![4u64, 5u64, 6u64],
                vec![7u64, 8u64, 9u64],
            ],
        ];
        let kernel = vec![vec![
            vec![vec![0u64, 1u64], vec![2u64, 3u64]],
            vec![vec![1u64, 2u64], vec![3u64, 4u64]],
        ]];
        let output = vec![vec![vec![56u64, 72u64], vec![104u64, 120u64]]];
        let assigned =
            Conv2dAssigned::<F, 3, 3, 2, 1, 2, 2, 2, 2, 8>::from_values(kernel, input, output);

        let circuit = Cnv2dCircuit::<F, 3, 3, 2, 1, 2, 2, 2, 2, 8> { assigned };
    }
}

//         let a_00: u64 = 1;
//         let a_01: u64 = 2;
//         let a_10: u64 = 3;
//         let a_11: u64 = 4;
//         let v_0: u64 = 5;
//         let v_1: u64 = 6;
//         let u_0: u64 = 17;
//         let u_1: u64 = 39;
//         let r_0: u64 = 17;
//         let r_1: u64 = 39;

//         let pub_inputs = vec![F::from(r_0), F::from(r_1), F::from(r_1)];
//         // Successful cases

//         let circuit = MvrlCircuit::<F, 2, 2, 8> {
//             a: vec![
//                 vec![
//                     Value::known(F::from(a_00).into()),
//                     Value::known(F::from(a_01).into()),
//                 ],
//                 vec![
//                     Value::known(F::from(a_10).into()),
//                     Value::known(F::from(a_11).into()),
//                 ],
//             ],
//             v: vec![
//                 Value::known(F::from(v_0).into()),
//                 Value::known(F::from(v_1).into()),
//             ],
//             u: vec![
//                 Value::known(F::from(u_0).into()),
//                 Value::known(F::from(u_1).into()),
//             ],
//             r: vec![
//                 Value::known(F::from(r_0).into()),
//                 Value::known(F::from(r_1).into()),
//             ],
//             //          wasa,
//             //         wasc,
//         };

//         // The MockProver arguments are log_2(nrows), the circuit (with advice already assigned), and the instance variables.
//         // The MockProver will need to internally supply a Layouter for the constraint system to be actually written.

//         let prover = MockProver::run(k, &circuit, vec![]).unwrap();
//         prover.assert_satisfied();
//     }

// }
