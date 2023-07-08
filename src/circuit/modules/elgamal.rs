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
use halo2_gadgets::poseidon::{
    primitives::{self as poseidon, ConstantLength},
    Hash as PoseidonHash, Pow5Chip as PoseidonChip, Pow5Config as PoseidonConfig,
};
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

use super::Module;

// Absolute offsets for public inputs.
const C1_X: usize = 0;
const C1_Y: usize = 1;
const SK_H: usize = 2;

///
const NUMBER_OF_LIMBS: usize = 4;
const BIT_LEN_LIMB: usize = 64;
/// The number of instance columns used by the ElGamal circuit.
pub const NUM_INSTANCE_COLUMNS: usize = 3;

type CircuitHash = PoseidonHash<Fr, PoseidonChip<Fr, 2, 1>, PoseidonSpec, ConstantLength<2>, 2, 1>;

#[derive(Debug)]
/// A chip implementing ElGamal encryption.
pub struct ElGamalChip {
    /// The configuration for this chip.
    config: ElGamalConfig,
    /// The ECC chip.
    ecc: BaseFieldEccChip<G1Affine, NUMBER_OF_LIMBS, BIT_LEN_LIMB>,
    /// The Poseidon hash chip.
    poseidon: PoseidonChip<Fr, 2, 1>,
    /// The addition chip.
    add: AddChip,
}

#[derive(Debug, Clone)]
/// Configuration for the ElGamal chip.
pub struct ElGamalConfig {
    main_gate_config: MainGateConfig,
    range_config: RangeConfig,
    poseidon_config: PoseidonConfig<Fr, 2, 1>,
    add_config: AddConfig,
    plaintext_col: Column<Advice>,
    ciphertext_c1_exp_col: Column<Instance>,
    ciphertext_c2_exp_col: Column<Instance>,
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
            poseidon: PoseidonChip::construct(p.poseidon_config.clone()),
            add: AddChip::construct(p.add_config.clone()),
            config: p,
        }
    }

    /// Configure the chip.
    fn configure(meta: &mut ConstraintSystem<Fr>) -> ElGamalConfig {
        let main_gate_config = MainGate::<Fr>::configure(meta);
        let advices = main_gate_config.advices();

        let fixed_columns = [
            meta.fixed_column(),
            meta.fixed_column(),
            meta.fixed_column(),
            meta.fixed_column(),
        ];

        meta.enable_constant(fixed_columns[3]);

        let rc_a = fixed_columns[0..2].try_into().unwrap();
        let rc_b = fixed_columns[2..4].try_into().unwrap();

        let rns = Rns::<Fq, Fr, NUMBER_OF_LIMBS, BIT_LEN_LIMB>::construct();

        let overflow_bit_lens = rns.overflow_lengths();
        let composition_bit_lens = vec![BIT_LEN_LIMB / NUMBER_OF_LIMBS];

        let range_config = RangeChip::<Fr>::configure(
            meta,
            &main_gate_config,
            composition_bit_lens,
            overflow_bit_lens,
        );

        let poseidon_config = PoseidonChip::configure::<PoseidonSpec>(
            meta,
            advices[1..3].try_into().unwrap(),
            advices[0],
            rc_a,
            rc_b,
        );

        let add_config = AddChip::configure(meta, advices[0], advices[1], advices[2]);

        let ciphertext_c1_exp_col = meta.instance_column();
        meta.enable_equality(ciphertext_c1_exp_col);

        let ciphertext_c2_exp_col = meta.instance_column();
        meta.enable_equality(ciphertext_c2_exp_col);

        let plaintext_col = advices[1];

        ElGamalConfig {
            poseidon_config,
            main_gate_config,
            range_config,
            add_config,
            plaintext_col,
            ciphertext_c1_exp_col,
            ciphertext_c2_exp_col,
        }
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
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
        //  and a public key mapped to the G2 curve:: pk=sk.G2
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

#[derive(Debug, Clone)]
/// A gadget implementing ElGamal encryption.
pub struct ElGamalGadget {
    /// The configuration for this gadget.
    config: ElGamalConfig,
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
    pub fn encrypt(pk: G1Affine, msg: Vec<Fr>, r: Fr) -> (G1, Vec<Fr>) {
        let g = G1Affine::generator();
        let c1 = g.mul(&r);

        let coords = pk.mul(&r).to_affine().coordinates().unwrap();

        let x = Integer::from_fe(*coords.x(), Self::rns());
        let y = Integer::from_fe(*coords.y(), Self::rns());

        let hasher = poseidon::Hash::<Fr, PoseidonSpec, ConstantLength<2>, 2, 1>::init();
        let dh = hasher.hash([x.native(), y.native()]); // this is Fq now :( (we need Fr)

        let mut c2 = vec![];

        for m in &msg {
            c2.push(m + dh);
        }

        (c1, c2)
    }

    /// Hash the secret key to be used as a public input.
    pub fn hash_sk(sk: Fr) -> Fr {
        let hasher = poseidon::Hash::<Fr, PoseidonSpec, ConstantLength<2>, 2, 1>::init();
        // this is Fq now :( (we need Fr)
        hasher.hash([sk, sk])
    }

    /// Decrypt a ciphertext using the secret key.
    pub fn decrypt(cipher: &(G1, Vec<Fr>), sk: Fr) -> Vec<Fr> {
        let c1 = cipher.0;
        let c2 = cipher.1.clone();

        let s = c1.mul(sk).to_affine().coordinates().unwrap();

        let x = Integer::from_fe(*s.x(), Self::rns());
        let y = Integer::from_fe(*s.y(), Self::rns());

        let hasher = poseidon::Hash::<Fr, PoseidonSpec, ConstantLength<2>, 2, 1>::init();
        let dh = hasher.hash([x.native(), y.native()]); // this is Fq now :( (we need Fr)

        let mut msg = vec![];
        for encrypted_m in &c2 {
            msg.push(encrypted_m - dh);
        }

        msg
    }

    /// Get the public inputs for the circuit.
    pub fn get_instances(cipher: &(G1, Vec<Fr>), sk_hash: Fr) -> Vec<Vec<Fr>> {
        let mut c1_and_sk = cipher
            .0
            .to_affine()
            .coordinates()
            .map(|c| {
                let x = Integer::from_fe(*c.x(), Self::rns());
                let y = Integer::from_fe(*c.y(), Self::rns());

                vec![x.native(), y.native()]
            })
            .unwrap();

        c1_and_sk.push(sk_hash);

        vec![c1_and_sk, cipher.1.clone()]
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
            let poseidon_hasher =
                CircuitHash::init(chip.poseidon, layouter.namespace(|| "Poseidon hasher"))?;
            poseidon_hasher.hash(
                layouter.namespace(|| "Poseidon hash (sk)"),
                [sk.clone(), sk.clone()],
            )?
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

        let (s, c1) = layouter.assign_region(
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

                Ok((s, c1))
            },
        )?;
        Ok([s, c1])
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
            let poseidon_message = [s.x().native().clone(), s.y().native().clone()];
            let poseidon_hasher =
                CircuitHash::init(chip.poseidon, layouter.namespace(|| "Poseidon hasher"))?;
            poseidon_hasher.hash(
                layouter.namespace(|| "Poseidon hash (randomness*pk)"),
                poseidon_message,
            )?
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

    fn new(config: Self::Config) -> Self {
        Self {
            config,
            variables: None,
        }
    }

    fn configure(meta: &mut ConstraintSystem<Fr>) -> Self::Config {
        ElGamalChip::configure(meta)
    }

    fn name(&self) -> &'static str {
        "ElGamal"
    }

    fn instance_increment_input(&self, var_len: Vec<usize>) -> Vec<usize> {
        // in order
        // 1. empty maingate instance
        // 2. c1, sk_hash
        // 3. c2
        vec![0, 0, var_len[0]]
    }

    fn instance_increment_module(&self) -> Vec<usize> {
        // in order
        // 1. empty maingate instance
        // 2. c1, sk_hash
        // 3. c2
        vec![0, 3, 0]
    }

    fn run(input: Self::RunInputs) -> Result<Vec<Vec<Fr>>, Box<dyn std::error::Error>> {
        let start_time = instant::Instant::now();

        let (input, var) = input;
        let len = input.len();

        let cipher = Self::encrypt(var.pk, input, var.r);
        // keep 1 empty (maingate instance variable).
        let mut public_inputs: Vec<Vec<Fr>> = vec![vec![]];
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
                            ValType::PrevAssigned(v) => Ok(v.clone()),
                            _ => panic!("wrong input type, must be previously assigned"),
                        })
                        .collect(),
                    ValTensor::Instance {
                        inner: col, dims, ..
                    } => {
                        // this should never ever fail
                        let num_elems = dims.iter().product::<usize>();
                        (0..num_elems)
                            .map(|i| {
                                region.assign_advice_from_instance(
                                    || "pub input anchor",
                                    *col,
                                    i,
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
                    message.len(),
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
        row_offsets: Vec<usize>,
    ) -> Result<ValTensor<Fr>, Error> {
        let start_time = instant::Instant::now();

        // if all equivalent to 0, then we are in the first row of the circuit
        if row_offsets.iter().all(|&x| x == 0) {
            self.config.config_range(layouter)?;
        }

        let (msg_var, sk_var) = self.layout_inputs(layouter, inputs)?;

        let [s, c1] = self.verify_secret(
            layouter.namespace(|| "verify_secret"),
            &self.config,
            &sk_var,
        )?;

        for (i, m) in msg_var.iter().enumerate() {
            let c2 = self.verify_encryption(
                layouter.namespace(|| "verify_encryption"),
                &self.config,
                m,
                &s,
            )?;

            layouter.constrain_instance(
                c2.cell(),
                self.config.ciphertext_c2_exp_col,
                i + row_offsets[2],
            )?;
        }

        // Force the public input to be the hash of the secret key so that we can ascertain decryption can happen
        let sk_hash = self.verify_sk_hash(
            layouter.namespace(|| "verify_sk_hash"),
            &self.config,
            &sk_var,
        )?;

        layouter
            .constrain_instance(
                c1.x().native().cell(),
                self.config.ciphertext_c1_exp_col,
                C1_X + row_offsets[1],
            )
            .and(layouter.constrain_instance(
                c1.y().native().cell(),
                self.config.ciphertext_c1_exp_col,
                C1_Y + row_offsets[1],
            ))
            .and(layouter.constrain_instance(
                sk_hash.cell(),
                self.config.ciphertext_c1_exp_col,
                SK_H + row_offsets[1],
            ))?;

        let assigned_input: Tensor<ValType<Fr>> =
            msg_var.iter().map(|e| ValType::from(e.clone())).into();

        log::trace!(
            "layout (N={:?}) took: {:?}",
            msg_var.len(),
            start_time.elapsed()
        );

        Ok(assigned_input.into())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use ark_std::test_rng;
    use halo2_proofs::{circuit::SimpleFloorPlanner, dev::MockProver, plonk::Circuit};

    struct EncryptytionCircuit {
        message: ValTensor<Fr>,
        variables: ElGamalVariables,
    }

    impl Circuit<Fr> for EncryptytionCircuit {
        type Config = ElGamalConfig;
        type FloorPlanner = SimpleFloorPlanner;
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
            ElGamalGadget::configure(meta)
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
            chip.layout(
                &mut layouter,
                &[self.message.clone(), sk.into()],
                vec![0; NUM_INSTANCE_COLUMNS],
            )?;
            Ok(())
        }
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

        let circuit = EncryptytionCircuit {
            message: message.into(),
            variables: var,
        };

        let res = MockProver::run(17, &circuit, public_inputs).unwrap();
        res.assert_satisfied_par();
    }
}
