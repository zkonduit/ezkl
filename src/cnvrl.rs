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

use tensorutils::{map3, map4};

/// The linear part of a 2D convolution layer.
/// Compile-time constants are:
/// IH: Height in pixels of the input image
/// IW: Width in pixels of the input image
/// CHIN: Number of input channels
/// CHOUT: Number of output channels = number of filters
/// KH: Kernel height in pixels
/// KW: Kernel width in pixels
/// BITS: Bits in the activation & weight representations (e.g. 8 for i8)
/// For now we only support stride of 1 and no padding.  Thus the ouput shape is
/// (IH - KH+1) pixels high and (IW-KW+1) pixels wide
/// PyTorch: https://pytorch.org/docs/stable/generated/torch.nn.Conv2d.html
/// PyTorch Conv2d activations are (N, CHIN, H, W), and we are doing single-input
/// inference so batchsize N=1.
/// We may add support for different types of padding and stride, and possibly dilation, groups, and bias.

#[derive(Clone)]
struct Conv2dAdvice<
    F: FieldExt,
    const IH: usize,
    const IW: usize,
    const CHIN: usize,
    const CHOUT: usize,
    const KH: usize,
    const KW: usize,
    const BITS: usize,
> {
    kernel: Vec<Vec<Vec<Vec<Column<Advice>>>>>, // CHOUT x CHIN x KH x KW
    input: Vec<Vec<Vec<Column<Advice>>>>,       // CHIN x IH x IW
    lin_output: Vec<Vec<Vec<Column<Advice>>>>,  // CHOUT x OH x OW
    nl_output: Vec<Vec<Vec<Column<Advice>>>>,   // CHOUT x OH x OW
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
    const BITS: usize,
> {
    // const OH = (IH - KH + 1);
    // const OW = (IW - KW + 1);
    advice: Conv2dAdvice<F, IH, IW, CHIN, CHOUT, KH, KW, BITS>,
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
        const BITS: usize,
    > Conv2dConfig<F, IH, IW, CHIN, CHOUT, KH, KW, BITS>
{
    fn define_advice(
        cs: &mut ConstraintSystem<F>,
    ) -> Conv2dAdvice<F, IH, IW, CHIN, CHOUT, KH, KW, BITS> {
        let kernel_advice: Vec<Vec<Vec<Vec<Column<Advice>>>>> =
            map4::<_, _, CHOUT, CHIN, KH, KW>(|i, j, k, l| cs.advice_column());
        let input_advice: Vec<Vec<Vec<Column<Advice>>>> =
            map3::<_, _, CHIN, IH, IW>(|i, j, k| cs.advice_column());
        let lin_output_advice: Vec<Vec<Vec<Column<Advice>>>> =
            map3::<_, _, CHOUT, OH, OW>(|i, j, k| cs.advice_column());
        let nl_output_advice: Vec<Vec<Vec<Column<Advice>>>> =
            map3::<_, _, CHOUT, OH, OW>(|i, j, k| cs.advice_column());
        Conv2dAdvice {
            kernel: kernel_advice,
            input: input_advice,
            lin_output: lin_output_advice,
            nl_output: nl_output_advice,
        }
    }

    fn configure(
        cs: &mut ConstraintSystem<F>,
    ) -> Conv2dConfig<F, IH, IW, CHIN, CHOUT, KH, KW, BITS> {
        let advice = Self::define_advice(&mut cs);
        let qs = cs.selector();

        cs.create_gate("cnvmul", |virtual_cells| {
            // 'allocate' all the advice and selector cols by querying the cs with the labels
            let q = virtual_cells.query_selector(qs);

            let kernel_ex: Vec<Vec<Vec<Vec<Expresssion<F>>>>> =
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
            let nl_output_ex: Vec<Vec<Vec<Expression<F>>>> =
                map3::<_, _, CHOUT, OH, OW>(|i, j, k| {
                    virtual_cells.query_advice(advice.nl_output[i][j][k], Rotation::cur())
                });

            // We put the negation -lin_output_ex in constraint tensor.
            let constraints: Vec<Vec<Vec<Expression<F>>>> =
                map3::<_, _, CHOUT, OH, OW>(|i, j, k| -lin_output_ex[i][j][k].clone());

            // Now we compute the convolution expression, collect it in a CHOUT x OH x OW tensor, and add it to constraints

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

        let relu_i_col = cs.lookup_table_column();
        let relu_o_col = cs.lookup_table_column();

        for i in 0..NROWS {
            let _ = cs.lookup(|cs| {
                vec![
                    (cs.query_advice(uadv[i], Rotation::cur()), relu_i_col),
                    (cs.query_advice(radv[i], Rotation::cur()), relu_o_col),
                ]
            });
        }

        Self {
            a: aadv,
            u: uadv,
            r: radv,
            v: vadv,
            q: qs,
            // i_col,
            // o_col,
            relu_i_col,
            relu_o_col,
            //            pub_col,
            _marker: PhantomData,
        }
    }

    // Allocates all legal input-output tuples for the ReLu function in the first 2^16 rows
    // of the constraint system.
    fn alloc_table(&self, layouter: &mut impl Layouter<F>) -> Result<(), Error> {
        layouter.assign_table(
            || "relu table",
            |mut table| {
                let mut row_offset = 0;
                let shift = F::from(127);
                for input in 0..255 {
                    table.assign_cell(
                        || format!("relu_i_col row {}", row_offset),
                        self.relu_i_col,
                        row_offset,
                        || Value::known(F::from(input) - shift), //-127..127
                    )?;
                    table.assign_cell(
                        || format!("relu_o_col row {}", row_offset),
                        self.relu_o_col,
                        row_offset,
                        || Value::known(F::from(if input < 127 { 127 } else { input }) - shift),
                    )?;
                    row_offset += 1;
                }
                Ok(())
            },
        )
    }

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
struct MvrlCircuit<F: FieldExt, const NROWS: usize, const NCOLS: usize, const BITS: usize> {
    // circuit holds Values
    a: Vec<Vec<Value<Assigned<F>>>>,
    v: Vec<Value<Assigned<F>>>,
    u: Vec<Value<Assigned<F>>>,
    r: Vec<Value<Assigned<F>>>,
    //    wasa: Value<Assigned<F>>,
    // Public input (from prover).
    //    wasc: Value<Assigned<F>>,
}
//impl<F: FieldExt, const RANGE: usize> MvmulCircuit<F, RANGE> {}

impl<F: FieldExt, const NROWS: usize, const NCOLS: usize, const BITS: usize> Circuit<F>
    for MvrlCircuit<F, NROWS, NCOLS, BITS>
{
    type Config = MvrlConfig<F, NROWS, NCOLS, BITS>;
    type FloorPlanner = SimpleFloorPlanner;
    //    type FloorPlanner = V1;

    fn without_witnesses(&self) -> Self {
        // put Unknown in all the Value<Assigned>
        let mut a: Vec<Vec<Value<Assigned<F>>>> = Vec::new();
        let mut v: Vec<Value<Assigned<F>>> = Vec::new();
        let mut u: Vec<Value<Assigned<F>>> = Vec::new();
        let mut r: Vec<Value<Assigned<F>>> = Vec::new();

        for _i in 0..NROWS {
            let mut row: Vec<Value<Assigned<F>>> = Vec::new();
            for _j in 0..NCOLS {
                row.push(Value::default());
            }
            a.push(row);
        }

        for _j in 0..NCOLS {
            v.push(Value::default());
        }

        for _i in 0..NROWS {
            u.push(Value::default());
        }

        for _i in 0..NROWS {
            r.push(Value::default());
        }
        //      let wasa: Value<Assigned<F>> = Value::default();
        //     let wasc: Value<Assigned<F>> = Value::default();

        MvrlCircuit {
            a,
            v,
            u,
            r,
            //       wasa,
            //     wasc,
        }
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
        let mut arr = Vec::new();

        layouter.assign_region(
            || "Assign values", // the name of the region
            |mut region| {
                let offset = 0;

                config.q.enable(&mut region, offset)?;

                for i in 0..NROWS {
                    for j in 0..NCOLS {
                        region.assign_advice(
                            || format!("a_{i}_{j}"),
                            config.a[i][j], // Column<Advice>
                            offset,
                            || self.a[i][j],
                        )?;
                    }
                }

                for i in 0..NROWS {
                    region.assign_advice(|| format!("u_{i}"), config.u[i], offset, || self.u[i])?;
                }

                for i in 0..NROWS {
                    arr.push(region.assign_advice(
                        || format!("r_{i}"),
                        config.r[i],
                        offset,
                        || self.r[i],
                    )?);
                }

                for j in 0..NCOLS {
                    region.assign_advice(|| format!("v_{j}"), config.v[j], offset, || self.v[j])?;
                }

                Ok(())
            },
        )?;

        config.alloc_table(&mut layouter)?;
        //        let c = config.alloc_private_and_public_inputs(&mut layouter, self.wasa, self.wasc)?;

        // for i in 0..NROWS {
        //     layouter.constrain_instance(arr[i].cell(), config.pub_col[i], 0)?; // equality for r and the pub_col? Why do we need the pub_col?
        // }

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
    use nalgebra;
    use std::time::{Duration, Instant};

    #[test]
    fn test_mvrl_succeed() {
        let k = 9; //2^k rows
        let a_00: u64 = 1;
        let a_01: u64 = 2;
        let a_10: u64 = 3;
        let a_11: u64 = 4;
        let v_0: u64 = 5;
        let v_1: u64 = 6;
        let u_0: u64 = 17;
        let u_1: u64 = 39;
        let r_0: u64 = 17;
        let r_1: u64 = 39;
        //        let wasa: Value<Assigned<F>> = Value::known((-F::from(3)).into());
        //        let wasc: Value<Assigned<F>> = Value::known(F::from(0).into());

        let pub_inputs = vec![F::from(r_0), F::from(r_1), F::from(r_1)];
        // Successful cases

        let circuit = MvrlCircuit::<F, 2, 2, 8> {
            a: vec![
                vec![
                    Value::known(F::from(a_00).into()),
                    Value::known(F::from(a_01).into()),
                ],
                vec![
                    Value::known(F::from(a_10).into()),
                    Value::known(F::from(a_11).into()),
                ],
            ],
            v: vec![
                Value::known(F::from(v_0).into()),
                Value::known(F::from(v_1).into()),
            ],
            u: vec![
                Value::known(F::from(u_0).into()),
                Value::known(F::from(u_1).into()),
            ],
            r: vec![
                Value::known(F::from(r_0).into()),
                Value::known(F::from(r_1).into()),
            ],
            //          wasa,
            //         wasc,
        };

        // The MockProver arguments are log_2(nrows), the circuit (with advice already assigned), and the instance variables.
        // The MockProver will need to internally supply a Layouter for the constraint system to be actually written.

        let prover = MockProver::run(k, &circuit, vec![]).unwrap();
        prover.assert_satisfied();
    }

    #[test]
    fn test_mvrl_withneg() {
        let k = 9; //2^k rows
        let a_00: u64 = 1;
        let a_01: u64 = 2;
        let a_10: u64 = 3;
        let a_11: u64 = 4;
        let v_0: u64 = 5;
        let v_1: u64 = 6;
        let u_0: u64 = 17;
        let u_1: u64 = 39;
        let r_0: u64 = 0;
        let r_1: u64 = 0;

        let circuit = MvrlCircuit::<F, 2, 2, 8> {
            a: vec![
                vec![
                    Value::known(F::from(a_00).into()),
                    Value::known(F::from(a_01).into()),
                ],
                vec![
                    Value::known(F::from(a_10).into()),
                    Value::known(F::from(a_11).into()),
                ],
            ],
            v: vec![
                Value::known((-F::from(v_0)).into()),
                Value::known((-F::from(v_1)).into()),
            ],
            u: vec![
                Value::known((-F::from(u_0)).into()),
                Value::known((-F::from(u_1)).into()),
            ],
            r: vec![
                Value::known(F::from(r_0).into()),
                Value::known(F::from(r_1).into()),
            ],
            //          wasa,
            //         wasc,
        };

        // The MockProver arguments are log_2(nrows), the circuit (with advice already assigned), and the instance variables.
        // The MockProver will need to internally supply a Layouter for the constraint system to be actually written.

        let prover = MockProver::run(k, &circuit, vec![]).unwrap();
        prover.assert_satisfied();
    }

    #[test]
    #[should_panic]
    fn test_mvrl_fail() {
        let k = 9; //2^k rows
        let a_00: u64 = 1;
        let a_01: u64 = 2;
        let a_10: u64 = 3;
        let a_11: u64 = 4;
        let v_0: u64 = 5;
        let v_1: u64 = 6;
        let u_0: u64 = 17;
        let u_1: u64 = 212;
        let r_0: u64 = 17;
        let r_1: u64 = 212;

        //        let wasa: Value<Assigned<F>> = Value::known((-F::from(3)).into());
        //       let wasc: Value<Assigned<F>> = Value::known(F::from(0).into());

        let pub_inputs = vec![F::from(r_0), F::from(r_1)];
        //        let pub_inputs = vec![F::from(0)];

        // Successful cases

        let circuit = MvrlCircuit::<F, 2, 2, 8> {
            a: vec![
                vec![
                    Value::known(F::from(a_00).into()),
                    Value::known(F::from(a_01).into()),
                ],
                vec![
                    Value::known(F::from(a_10).into()),
                    Value::known(F::from(a_11).into()),
                ],
            ],
            v: vec![
                Value::known(F::from(v_0).into()),
                Value::known(F::from(v_1).into()),
            ],
            u: vec![
                Value::known(F::from(u_0).into()),
                Value::known(F::from(u_1).into()),
            ],
            r: vec![
                Value::known(F::from(r_0).into()),
                Value::known(F::from(r_1).into()),
            ],
            //         wasa,
            //         wasc,
        };

        let prover = MockProver::run(k, &circuit, vec![]).unwrap();
        prover.assert_satisfied();
    }

    // #[test]
    // #[ignore]
    // fn test_mvmul_bigger() {
    //     const NROWS: usize = 128;
    //     const NCOLS: usize = 128; //32x52 overflows stack in verifying key creation
    //     let k = 4; //2^k rows
    //                //        let avals: [u64; 32] = rand.rand();
    //     let m_a: nalgebra::SMatrix<u64, NROWS, NCOLS> =
    //         nalgebra::SMatrix::from_iterator((0..NROWS * NCOLS).map(|x| x as u64));
    //     let m_v: nalgebra::SVector<u64, NCOLS> =
    //         nalgebra::SVector::from_iterator((0..NCOLS).map(|x| x as u64));
    //     let m_u = m_a * m_v;
    //     println!("a: {}", m_a);
    //     println!("v: {}", m_v);
    //     println!("u: {}", m_u);
    //     println!("{}", NROWS * NCOLS + NROWS + NCOLS);

    //     let mut a: Vec<Vec<Value<Assigned<F>>>> = Vec::new();
    //     for i in 0..NROWS {
    //         let mut row: Vec<Value<Assigned<F>>> = Vec::new();
    //         for j in 0..NCOLS {
    //             row.push(Value::known(F::from(m_a[(i, j)]).into()))
    //         }
    //         a.push(row);
    //     }

    //     let mut v: Vec<Value<Assigned<F>>> = Vec::new();
    //     let mut u: Vec<Value<Assigned<F>>> = Vec::new();
    //     for j in 0..NCOLS {
    //         v.push(Value::known(F::from(m_v[j]).into()));
    //     }

    //     for i in 0..NROWS {
    //         u.push(Value::known(F::from(m_u[i]).into()));
    //     }

    //     let circuit = MvrlCircuit::<F, NROWS, NCOLS, NBITS> { a, v, u, r };

    //     let params: Params<vesta::Affine> = Params::new(k);

    //     let empty_circuit = circuit.without_witnesses();

    //     // Initialize the proving key
    //     let now = Instant::now();
    //     let vk = keygen_vk(&params, &empty_circuit).expect("keygen_vk should not fail");
    //     println!("VK took {}", now.elapsed().as_secs());
    //     let now = Instant::now();
    //     let pk = keygen_pk(&params, vk.clone(), &empty_circuit).expect("keygen_pk should not fail");
    //     println!("PK took {}", now.elapsed().as_secs());
    //     let now = Instant::now();
    //     //println!("{:?} {:?}", vk, pk);
    //     let mut transcript = Blake2bWrite::<_, _, Challenge255<_>>::init(vec![]);
    //     let mut rng = OsRng;
    //     create_proof(&params, &pk, &[circuit], &[&[]], &mut rng, &mut transcript)
    //         .expect("proof generation should not fail");
    //     let proof = transcript.finalize();
    //     //println!("{:?}", proof);
    //     println!("Proof took {}", now.elapsed().as_secs());
    //     let now = Instant::now();
    //     let strategy = SingleVerifier::new(&params);
    //     let mut transcript = Blake2bRead::<_, _, Challenge255<_>>::init(&proof[..]);
    //     assert!(verify_proof(&params, pk.get_vk(), strategy, &[&[]], &mut transcript).is_ok());
    //     println!("Verify took {}", now.elapsed().as_secs());
    // }
}
