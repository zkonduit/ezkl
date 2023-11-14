/*
An easy-to-use implementation of the ElGamal Encryption in the form of a Halo2 Chip.
Huge thank you to https://github.com/timoftime/ for providing the inspiration and launching point for this <3 .
*/

///
mod add_chip;

use crate::circuit::modules::poseidon::spec::PoseidonSpec;
use crate::tensor::{Tensor, ValTensor, ValType};
use add_chip::{AddChip, AddConfig, AddInstruction};
use ark_std::rand::{CryptoRng, RngCore};
use halo2_proofs::arithmetic::Field;
use halo2_proofs::circuit::{AssignedCell, Chip, Layouter, Value};
use halo2_proofs::plonk;
use halo2_proofs::plonk::{Advice, Column, ConstraintSystem, Error, Instance};
use halo2_wrong_ecc::integer::rns::{Common, Integer, Rns};
use halo2_wrong_ecc::maingate::{
    MainGate, MainGateConfig, RangeChip, RangeConfig, RangeInstructions, RegionCtx,
};
use halo2_wrong_ecc::{AssignedPoint, BaseFieldEccChip, EccConfig};
use halo2curves::bn256::{Fq, Fr, G1Affine, G1};
use halo2curves::group::cofactor::CofactorCurveAffine;
use halo2curves::group::{Curve, Group};
use halo2curves::CurveAffine;
use serde::{Deserialize, Serialize};
use std::ops::{Mul, MulAssign};
use std::rc::Rc;
use std::vec;

use super::poseidon::{PoseidonChip, PoseidonConfig};
use super::Module;

// Absolute offsets for public inputs.
const C1_X: usize = 0;
const C1_Y: usize = 1;
const SK_H: usize = 2;
const C2_H: usize = 3;

///
const NUMBER_OF_LIMBS: usize = 4;
const BIT_LEN_LIMB: usize = 64;
/// The number of instance columns used by the ElGamal circuit.
pub const NUM_INSTANCE_COLUMNS: usize = 1;

/// The poseidon hash width.
pub const POSEIDON_WIDTH: usize = 2;
/// The poseidon hash rate.
pub const POSEIDON_RATE: usize = 1;
/// The poseidon len
pub const POSEIDON_LEN: usize = 2;

#[derive(Debug)]
/// A chip implementing ElGamal encryption.
pub struct ElGamalChip {
    /// The configuration for this chip.
    pub config: ElGamalConfig,
    /// The ECC chip.
    ecc: BaseFieldEccChip<G1Affine, NUMBER_OF_LIMBS, BIT_LEN_LIMB>,
    /// The Poseidon hash chip.
    poseidon: PoseidonChip<PoseidonSpec, POSEIDON_WIDTH, POSEIDON_RATE, POSEIDON_LEN>,
    /// The addition chip.
    add: AddChip,
}

#[derive(Debug, Clone)]
/// Configuration for the ElGamal chip.
pub struct ElGamalConfig {
    main_gate_config: MainGateConfig,
    range_config: RangeConfig,
    poseidon_config: PoseidonConfig<POSEIDON_WIDTH, POSEIDON_RATE>,
    add_config: AddConfig,
    plaintext_col: Column<Advice>,
    /// The column used for the instance.
    pub instance: Column<Instance>,
    /// The config has been initialized.
    pub initialized: bool,
}

impl ElGamalConfig {
    fn config_range(&self, layouter: &mut impl Layouter<Fr>) -> Result<(), Error> {
        let range_chip = RangeChip::<Fr>::new(self.range_config.clone());
        range_chip.load_table(layouter)?;
        Ok(())
    }

    fn ecc_chip_config(&self) -> EccConfig {
        EccConfig::new(self.range_config.clone(), self.main_gate_config.clone())
    }
}

impl Chip<Fq> for ElGamalChip {
    type Config = ElGamalConfig;
    type Loaded = ();

    fn config(&self) -> &Self::Config {
        &self.config
    }

    fn loaded(&self) -> &Self::Loaded {
        &()
    }
}

impl ElGamalChip {
    /// Create a new `ElGamalChip`.
    pub fn new(p: ElGamalConfig) -> ElGamalChip {
        ElGamalChip {
            ecc: BaseFieldEccChip::new(p.ecc_chip_config()),
            poseidon: PoseidonChip::new(p.poseidon_config.clone()),
            add: AddChip::construct(p.add_config.clone()),
            config: p,
        }
    }

    /// Configure the chip.
    fn configure(meta: &mut ConstraintSystem<Fr>) -> ElGamalConfig {
        let main_gate_config = MainGate::<Fr>::configure(meta);
        let advices = main_gate_config.advices();
        let main_fixed_columns = main_gate_config.fixed();
        let instance = main_gate_config.instance();

        let rc_a = main_fixed_columns[3..5].try_into().unwrap();
        let rc_b = [meta.fixed_column(), meta.fixed_column()];

        meta.enable_constant(rc_b[0]);

        let rns = Rns::<Fq, Fr, NUMBER_OF_LIMBS, BIT_LEN_LIMB>::construct();

        let overflow_bit_lens = rns.overflow_lengths();
        let composition_bit_lens = vec![BIT_LEN_LIMB / NUMBER_OF_LIMBS];

        let range_config = RangeChip::<Fr>::configure(
            meta,
            &main_gate_config,
            composition_bit_lens,
            overflow_bit_lens,
        );

        let poseidon_config =
            PoseidonChip::<PoseidonSpec, POSEIDON_WIDTH, POSEIDON_RATE, 2>::configure_with_cols(
                meta,
                advices[0],
                rc_a,
                rc_b,
                advices[1..3].try_into().unwrap(),
                None,
            );

        let add_config = AddChip::configure(meta, advices[0], advices[1], advices[2]);

        let plaintext_col = advices[1];

        ElGamalConfig {
            poseidon_config,
            main_gate_config,
            range_config,
            add_config,
            plaintext_col,
            instance,
            initialized: false,
        }
    }
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
/// The variables used in the ElGamal circuit.
pub struct ElGamalVariables {
    /// The randomness used in the encryption.
    pub r: Fr,
    /// The public key.
    pub pk: G1Affine,
    /// The secret key.
    pub sk: Fr,
    /// The window size used in the ECC chip.
    pub window_size: usize,
    /// The auxiliary generator used in the ECC chip.
    pub aux_generator: G1Affine,
}

impl Default for ElGamalVariables {
    fn default() -> Self {
        Self {
            r: Fr::zero(),
            pk: G1Affine::identity(),
            sk: Fr::zero(),
            window_size: 4,
            aux_generator: G1Affine::identity(),
        }
    }
}

impl ElGamalVariables {
    /// Create new variables.
    pub fn new(r: Fr, pk: G1Affine, sk: Fr, window_size: usize, aux_generator: G1Affine) -> Self {
        Self {
            r,
            pk,
            sk,
            window_size,
            aux_generator,
        }
    }

    /// Generate random variables.
    pub fn gen_random<R: CryptoRng + RngCore>(mut rng: &mut R) -> Self {
        // get a random element from the scalar field
        let sk = Fr::random(&mut rng);

        // compute secret_key*generator to derive the public key
        // With BN256, we create the private key from a random number. This is a private key value (sk
        // and a public key mapped to the G2 curve:: pk=sk.G2
        let mut pk = G1::generator();
        pk.mul_assign(sk);

        Self {
            r: Fr::random(&mut rng),
            pk: pk.to_affine(),
            sk,
            window_size: 4,
            aux_generator: <G1Affine as CurveAffine>::CurveExt::random(rng).to_affine(),
        }
    }
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
/// The cipher returned from the ElGamal encryption.
pub struct ElGamalCipher {
    /// c1 := r*G
    pub c1: G1,
    /// c2 := m*s
    pub c2: Vec<Fr>,
}

#[derive(Debug, Clone)]
/// A gadget implementing ElGamal encryption.
pub struct ElGamalGadget {
    /// The configuration for this gadget.
    pub config: ElGamalConfig,
    /// The variables used in this gadget.
    variables: Option<ElGamalVariables>,
}

impl ElGamalGadget {
    /// Load the variables into the gadget.
    pub fn load_variables(&mut self, variables: ElGamalVariables) {
        self.variables = Some(variables);
    }

    fn rns() -> Rc<Rns<Fq, Fr, NUMBER_OF_LIMBS, BIT_LEN_LIMB>> {
        let rns = Rns::<Fq, Fr, NUMBER_OF_LIMBS, BIT_LEN_LIMB>::construct();
        Rc::new(rns)
    }

    /// Encrypt a message using the public key.
    pub fn encrypt(pk: G1Affine, msg: Vec<Fr>, r: Fr) -> ElGamalCipher {
        let g = G1Affine::generator();
        let c1 = g.mul(&r);

        let coords = pk.mul(&r).to_affine().coordinates().unwrap();

        let x = Integer::from_fe(*coords.x(), Self::rns());
        let y = Integer::from_fe(*coords.y(), Self::rns());

        let dh = PoseidonChip::<PoseidonSpec, POSEIDON_WIDTH, POSEIDON_RATE, POSEIDON_LEN>::run(
            [x.native(), y.native()].to_vec(),
        )
        .unwrap()[0][0];

        let mut c2 = vec![];

        for m in &msg {
            c2.push(m + dh);
        }

        ElGamalCipher { c1, c2 }
    }

    /// Hash the msssage to be used as a public input.
    pub fn hash_encrypted_msg(msg: Vec<Fr>) -> Fr {
        PoseidonChip::<PoseidonSpec, POSEIDON_WIDTH, POSEIDON_RATE, POSEIDON_LEN>::run(msg).unwrap()
            [0][0]
    }

    /// Hash the secret key to be used as a public input.
    pub fn hash_sk(sk: Fr) -> Fr {
        PoseidonChip::<PoseidonSpec, POSEIDON_WIDTH, POSEIDON_RATE, POSEIDON_LEN>::run(vec![sk, sk])
            .unwrap()[0][0]
    }

    /// Decrypt a ciphertext using the secret key.
    pub fn decrypt(cipher: &ElGamalCipher, sk: Fr) -> Vec<Fr> {
        let c1 = cipher.c1;
        let c2 = cipher.c2.clone();

        let s = c1.mul(sk).to_affine().coordinates().unwrap();

        let x = Integer::from_fe(*s.x(), Self::rns());
        let y = Integer::from_fe(*s.y(), Self::rns());

        let dh = PoseidonChip::<PoseidonSpec, POSEIDON_WIDTH, POSEIDON_RATE, POSEIDON_LEN>::run(
            [x.native(), y.native()].to_vec(),
        )
        .unwrap()[0][0];

        let mut msg = vec![];
        for encrypted_m in &c2 {
            msg.push(encrypted_m - dh);
        }

        msg
    }

    /// Get the public inputs for the circuit.
    pub fn get_instances(cipher: &ElGamalCipher, sk_hash: Fr) -> Vec<Vec<Fr>> {
        let mut c1_and_sk = cipher
            .c1
            .to_affine()
            .coordinates()
            .map(|c| {
                let x = Integer::from_fe(*c.x(), Self::rns());
                let y = Integer::from_fe(*c.y(), Self::rns());

                vec![x.native(), y.native()]
            })
            .unwrap();

        c1_and_sk.push(sk_hash);

        c1_and_sk.push(Self::hash_encrypted_msg(cipher.c2.clone()));

        vec![c1_and_sk]
    }

    pub(crate) fn verify_encrypted_msg_hash(
        &self,
        mut layouter: impl Layouter<Fr>,
        config: &ElGamalConfig,
        encrypted_msg: &[AssignedCell<Fr, Fr>],
    ) -> Result<AssignedCell<Fr, Fr>, plonk::Error> {
        let chip = ElGamalChip::new(config.clone());

        // compute dh = poseidon_hash(randomness*pk)
        let encrypted_msg_hash = {
            let poseidon_message =
                Tensor::from(encrypted_msg.iter().map(|m| ValType::from(m.clone())));

            chip.poseidon.layout(
                &mut layouter.namespace(|| "Poseidon hash (encrypted_msg)"),
                &[poseidon_message.into()],
                0,
            )?
        };

        let encrypted_msg_hash = match &encrypted_msg_hash
            .get_inner_tensor()
            .map_err(|_| plonk::Error::Synthesis)?[0]
        {
            ValType::PrevAssigned(v) => v.clone(),
            _ => panic!("poseidon hash should be an assigned value"),
        };

        Ok(encrypted_msg_hash)
    }

    /// Hash the secret key to be used as a public input.
    pub(crate) fn verify_sk_hash(
        &self,
        mut layouter: impl Layouter<Fr>,
        config: &ElGamalConfig,
        sk: &AssignedCell<Fr, Fr>,
    ) -> Result<AssignedCell<Fr, Fr>, plonk::Error> {
        let chip = ElGamalChip::new(config.clone());

        // compute dh = poseidon_hash(randomness*pk)
        let sk_hash = {
            let poseidon_message =
                Tensor::from([ValType::from(sk.clone()), ValType::from(sk.clone())].into_iter());

            chip.poseidon.layout(
                &mut layouter.namespace(|| "Poseidon hash (sk)"),
                &[poseidon_message.into()],
                0,
            )?
        };

        let sk_hash = match &sk_hash
            .get_inner_tensor()
            .map_err(|_| plonk::Error::Synthesis)?[0]
        {
            ValType::PrevAssigned(v) => v.clone(),
            _ => panic!("poseidon hash should be an assigned value"),
        };

        Ok(sk_hash)
    }

    pub(crate) fn verify_secret(
        &self,
        mut layouter: impl Layouter<Fr>,
        config: &ElGamalConfig,
        sk: &AssignedCell<Fr, Fr>,
    ) -> Result<[AssignedPoint<Fq, Fr, NUMBER_OF_LIMBS, BIT_LEN_LIMB>; 2], plonk::Error> {
        let mut chip = ElGamalChip::new(config.clone());

        let g = G1Affine::generator();

        let variables = match self.variables {
            Some(ref variables) => variables,
            None => panic!("variables not loaded"),
        };

        // compute s = randomness*pk
        let s = variables.pk.mul(variables.r).to_affine();
        let c1 = g.mul(variables.r).to_affine();

        layouter.assign_region(
            || "obtain_s",
            |region| {
                let offset = 0;
                let ctx = &mut RegionCtx::new(region, offset);

                chip.ecc
                    .assign_aux_generator(ctx, Value::known(variables.aux_generator))?;
                chip.ecc.assign_aux(ctx, variables.window_size, 1)?;

                let s = chip.ecc.assign_point(ctx, Value::known(s)).unwrap();
                // compute c1 = randomness*generator
                let c1 = chip.ecc.assign_point(ctx, Value::known(c1)).unwrap();

                let s_from_sk = chip.ecc.mul(ctx, &c1, sk, variables.window_size).unwrap();

                chip.ecc.assert_equal(ctx, &s, &s_from_sk)?;

                Ok([s, c1])
            },
        )
    }

    pub(crate) fn verify_encryption(
        &self,
        mut layouter: impl Layouter<Fr>,
        config: &ElGamalConfig,
        m: &AssignedCell<Fr, Fr>,
        s: &AssignedPoint<Fq, Fr, NUMBER_OF_LIMBS, BIT_LEN_LIMB>,
    ) -> Result<AssignedCell<Fr, Fr>, plonk::Error> {
        let chip = ElGamalChip::new(config.clone());

        // compute dh = poseidon_hash(randomness*pk)
        let dh = {
            let poseidon_message = Tensor::from(
                [
                    ValType::from(s.x().native().clone()),
                    ValType::from(s.y().native().clone()),
                ]
                .into_iter(),
            );

            chip.poseidon.layout(
                &mut layouter.namespace(|| "Poseidon hasher"),
                &[poseidon_message.into()],
                0,
            )?
        };

        let dh = match &dh.get_inner_tensor().map_err(|_| plonk::Error::Synthesis)?[0] {
            ValType::PrevAssigned(v) => v.clone(),
            _ => panic!("poseidon hash should be an assigned value"),
        };

        // compute c2 = poseidon_hash(nk, rho) + psi.
        let c2 = chip.add.add(
            layouter.namespace(|| "c2 = poseidon_hash(randomness*pk) + m"),
            &dh,
            m,
        )?;

        Ok(c2)
    }
}

impl Module<Fr> for ElGamalGadget {
    type Config = ElGamalConfig;
    type InputAssignments = (Vec<AssignedCell<Fr, Fr>>, AssignedCell<Fr, Fr>);
    type RunInputs = (Vec<Fr>, ElGamalVariables);
    type Params = ();

    fn new(config: Self::Config) -> Self {
        Self {
            config,
            variables: None,
        }
    }

    fn configure(meta: &mut ConstraintSystem<Fr>, _: Self::Params) -> Self::Config {
        ElGamalChip::configure(meta)
    }

    fn name(&self) -> &'static str {
        "ElGamal"
    }

    fn instance_increment_input(&self) -> Vec<usize> {
        // in order
        // 1. c1, sk_hash, c2_hash
        vec![4]
    }

    fn run(input: Self::RunInputs) -> Result<Vec<Vec<Fr>>, Box<dyn std::error::Error>> {
        let start_time = instant::Instant::now();

        let (input, var) = input;
        let len = input.len();

        let cipher = Self::encrypt(var.pk, input, var.r);
        // keep 1 empty (maingate instance variable).
        let mut public_inputs: Vec<Vec<Fr>> = vec![];
        public_inputs.extend(Self::get_instances(&cipher, Self::hash_sk(var.sk)));

        log::trace!("run (N={:?}) took: {:?}", len, start_time.elapsed());

        Ok(public_inputs)
    }

    fn layout_inputs(
        &self,
        layouter: &mut impl Layouter<Fr>,
        inputs: &[ValTensor<Fr>],
    ) -> Result<Self::InputAssignments, Error> {
        assert_eq!(inputs.len(), 2);
        let message = inputs[0].clone();
        let message_offset = message
            .get_inner_tensor()
            .map_err(|_| Error::Synthesis)?
            .len();
        let sk = inputs[1].clone();

        let start_time = instant::Instant::now();
        let (msg_var, sk_var) = layouter.assign_region(
            || "plaintext",
            |mut region| {
                let msg_var: Result<Vec<AssignedCell<Fr, Fr>>, Error> = match &message {
                    ValTensor::Value { inner: v, .. } => v
                        .iter()
                        .enumerate()
                        .map(|(i, value)| match value {
                            ValType::Value(v) => region.assign_advice(
                                || format!("load message_{}", i),
                                self.config.plaintext_col,
                                i,
                                || *v,
                            ),
                            ValType::PrevAssigned(v) | ValType::AssignedConstant(v, ..) => {
                                Ok(v.clone())
                            }
                            ValType::Constant(f) => region.assign_advice_from_constant(
                                || format!("load message_{}", i),
                                self.config.plaintext_col,
                                i,
                                *f,
                            ),
                            e => panic!("wrong input type {:?}, must be previously assigned", e),
                        })
                        .collect(),
                    ValTensor::Instance {
                        dims,
                        inner: col,
                        idx,
                        initial_offset,
                        ..
                    } => {
                        // this should never ever fail
                        let num_elems = dims[*idx].iter().product::<usize>();
                        (0..num_elems)
                            .map(|i| {
                                region.assign_advice_from_instance(
                                    || "pub input anchor",
                                    *col,
                                    initial_offset + i,
                                    self.config.plaintext_col,
                                    i,
                                )
                            })
                            .collect()
                    }
                };

                let sk = match sk.get_inner_tensor().unwrap()[0] {
                    ValType::Value(v) => v,
                    _ => panic!("wrong input type"),
                };

                let sk_var = region.assign_advice(
                    || "sk",
                    self.config.plaintext_col,
                    message_offset,
                    || sk,
                )?;

                Ok((msg_var?, sk_var))
            },
        )?;
        let duration = start_time.elapsed();
        log::trace!("layout inputs took: {:?}", duration);

        Ok((msg_var, sk_var))
    }

    fn layout(
        &self,
        layouter: &mut impl Layouter<Fr>,
        inputs: &[ValTensor<Fr>],
        row_offset: usize,
    ) -> Result<ValTensor<Fr>, Error> {
        let start_time = instant::Instant::now();

        // if all equivalent to 0, then we are in the first row of the circuit
        if !self.config.initialized {
            self.config.config_range(layouter).unwrap();
        }

        let (msg_var, sk_var) = self.layout_inputs(layouter, inputs)?;

        let [s, c1] = self.verify_secret(
            layouter.namespace(|| "verify_secret"),
            &self.config,
            &sk_var,
        )?;

        // Force the public input to be the hash of the secret key so that we can ascertain decryption can happen
        let sk_hash = self.verify_sk_hash(
            layouter.namespace(|| "verify_sk_hash"),
            &self.config,
            &sk_var,
        )?;

        layouter
            .constrain_instance(
                c1.x().native().cell(),
                self.config.instance,
                C1_X + row_offset,
            )
            .and(layouter.constrain_instance(
                c1.y().native().cell(),
                self.config.instance,
                C1_Y + row_offset,
            ))
            .and(layouter.constrain_instance(
                sk_hash.cell(),
                self.config.instance,
                SK_H + row_offset,
            ))?;

        let c2: Result<Vec<AssignedCell<Fr, Fr>>, _> = msg_var
            .iter()
            .map(|m| {
                self.verify_encryption(
                    layouter.namespace(|| "verify_encryption"),
                    &self.config,
                    m,
                    &s,
                )
            })
            .collect();

        let c2 = c2?;

        let c2_hash = self.verify_encrypted_msg_hash(
            layouter.namespace(|| "verify_c2_hash"),
            &self.config,
            &c2,
        )?;

        layouter.constrain_instance(c2_hash.cell(), self.config.instance, C2_H + row_offset)?;

        let mut assigned_input: Tensor<ValType<Fr>> =
            msg_var.iter().map(|e| ValType::from(e.clone())).into();

        assigned_input.reshape(inputs[0].dims());

        log::trace!(
            "layout (N={:?}) took: {:?}",
            msg_var.len(),
            start_time.elapsed()
        );

        Ok(assigned_input.into())
    }

    fn num_rows(input_len: usize) -> usize {
        // this was determined by running the circuit and looking at the number of constraints
        // in the test called hash_for_a_range_of_input_sizes, then regressing in python to find the slope
        // ```python
        // import numpy as np
        // x = [1, 2, 3, 512, 513, 514]
        // y = [75424, 75592, 75840, 161017, 161913, 162000]
        // def fit_above(x, y) :
        //     x0, y0 = x[0] - 1, y[0]
        //     x -= x0
        //     y -= y0
        //     def error_function_2(b, x, y) :
        //         a = np.min((y - b) / x)
        //         return np.sum((y - a * x - b)**2)
        //     b = scipy.optimize.minimize(error_function_2, [0], args=(x, y)).x[0]
        //     a = np.max((y - b) / x)
        //     return a, b - a * x0 + y0
        // a, b = fit_above(x, y)
        // plt.plot(x, y, 'o')
        // plt.plot(x, a*x + b, '-')
        // plt.show()
        // for (x_i, y_i) in zip(x,y):
        // assert y_i <= a*x_i + b
        // print(a, b)
        // ```
        const NUM_CONSTRAINTS_SLOPE: usize = 196;
        const NUM_CONSTRAINTS_INTERCEPT: usize = 75257;

        // check if even or odd
        input_len * NUM_CONSTRAINTS_SLOPE + NUM_CONSTRAINTS_INTERCEPT
    }
}

#[cfg(test)]
mod tests {
    use crate::circuit::modules::ModulePlanner;

    use super::*;
    use ark_std::test_rng;
    use halo2_proofs::{dev::MockProver, plonk::Circuit};

    struct EncryptionCircuit {
        message: ValTensor<Fr>,
        variables: ElGamalVariables,
    }

    impl Circuit<Fr> for EncryptionCircuit {
        type Config = ElGamalConfig;
        type FloorPlanner = ModulePlanner;
        type Params = ();

        fn without_witnesses(&self) -> Self {
            let empty_val: Vec<ValType<Fr>> = vec![Value::<Fr>::unknown().into()];
            let message: Tensor<ValType<Fr>> = empty_val.into_iter().into();

            let variables = ElGamalVariables::default();

            Self {
                message: message.into(),
                variables,
            }
        }

        fn configure(meta: &mut ConstraintSystem<Fr>) -> ElGamalConfig {
            ElGamalGadget::configure(meta, ())
        }

        fn synthesize(
            &self,
            config: ElGamalConfig,
            mut layouter: impl Layouter<Fr>,
        ) -> Result<(), Error> {
            let mut chip = ElGamalGadget::new(config);
            chip.load_variables(self.variables.clone());
            let sk: Tensor<ValType<Fr>> =
                Tensor::new(Some(&[Value::known(self.variables.sk).into()]), &[1]).unwrap();
            chip.layout(&mut layouter, &[self.message.clone(), sk.into()], 0)?;
            Ok(())
        }
    }

    #[test]
    // this is for backwards compatibility with the old format
    fn test_variables_serialization_round_trip() {
        let mut rng = test_rng();

        let var = ElGamalVariables::gen_random(&mut rng);

        let mut buf = vec![];
        serde_json::to_writer(&mut buf, &var).unwrap();

        let var2 = serde_json::from_reader(&buf[..]).unwrap();

        assert_eq!(var, var2);
    }

    #[test]
    pub fn test_encrypt_decrypt() {
        let mut rng = test_rng();

        let var = ElGamalVariables::gen_random(&mut rng);

        let mut msg = vec![];
        //
        for _ in 0..32 {
            msg.push(Fr::random(&mut rng));
        }

        let cipher = ElGamalGadget::encrypt(var.pk, msg.clone(), var.r);

        let decrypted_msg = ElGamalGadget::decrypt(&cipher, var.sk);

        assert_eq!(decrypted_msg, msg);
    }

    #[test]
    pub fn test_circuit() {
        let mut rng = test_rng();

        let var = ElGamalVariables::gen_random(&mut rng);

        let mut msg = vec![];
        //
        for _ in 0..2 {
            msg.push(Fr::random(&mut rng));
        }

        let run_inputs = (msg.clone(), var.clone());
        let public_inputs: Vec<Vec<Fr>> = ElGamalGadget::run(run_inputs).unwrap();

        let message: Tensor<ValType<Fr>> = msg.into_iter().map(|m| Value::known(m).into()).into();

        let circuit = EncryptionCircuit {
            message: message.into(),
            variables: var,
        };

        let res = MockProver::run(17, &circuit, public_inputs).unwrap();
        res.assert_satisfied_par();
    }

    #[test]
    #[ignore]
    pub fn test_circuit_range_of_input_sizes() {
        let mut rng = test_rng();

        #[cfg(not(target_arch = "wasm32"))]
        env_logger::init();

        //
        for i in [1, 2, 3, 512, 513, 514, 1024] {
            println!("i is {} ----------------------------------------", i);

            let var = ElGamalVariables::gen_random(&mut rng);
            let mut msg = vec![];
            for _ in 0..i {
                msg.push(Fr::random(&mut rng));
            }

            let run_inputs = (msg.clone(), var.clone());
            let public_inputs: Vec<Vec<Fr>> = ElGamalGadget::run(run_inputs).unwrap();

            let message: Tensor<ValType<Fr>> =
                msg.into_iter().map(|m| Value::known(m).into()).into();

            let circuit = EncryptionCircuit {
                message: message.into(),
                variables: var,
            };

            let res = MockProver::run(19, &circuit, public_inputs).unwrap();
            res.assert_satisfied_par();
        }
    }
}
