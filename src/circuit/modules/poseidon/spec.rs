//! Specification for rate 4 Poseidon using the BN256 curve.
//! Patterned after [halo2_gadgets::poseidon::primitives::P128Pow5T3]

use halo2_gadgets::poseidon::primitives::*;
use halo2_proofs::arithmetic::Field;
use halo2_proofs::halo2curves::bn256::Fr as Fp;

///
#[derive(Debug, Clone, Copy)]
pub struct PoseidonSpec;

pub(crate) type Mds<Fp, const T: usize> = [[Fp; T]; T];

impl Spec<Fp, 15, 14> for PoseidonSpec {
    fn full_rounds() -> usize {
        8
    }

    fn partial_rounds() -> usize {
        60
    }

    fn sbox(val: Fp) -> Fp {
        val.pow_vartime(&[5])
    }

    fn secure_mds() -> usize {
        unimplemented!()
    }

    fn constants() -> (Vec<[Fp; 15]>, Mds<Fp, 15>, Mds<Fp, 15>) {
        (
            super::rate15_params::ROUND_CONSTANTS[..].to_vec(),
            super::rate15_params::MDS,
            super::rate15_params::MDS_INV,
        )
    }
}
