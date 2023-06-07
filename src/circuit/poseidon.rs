/*
An easy-to-use implementation of the Poseidon Hash in the form of a Halo2 Chip. While the Poseidon Hash function
is already implemented in halo2_gadgets, there is no wrapper chip that makes it easy to use in other circuits.
Thanks to https://github.com/summa-dev/summa-solvency/blob/master/src/chips/poseidon/hash.rs for the inspiration (and also helping us understand how to use this).
*/

use halo2_proofs::circuit::Value;
use halo2curves::{bn256::Fr, ff::Field};

use self::{primitives::ConstantLength, spec::PoseidonSpec};
///
pub mod base;
///
pub mod grain;
///
pub mod mds;
///
pub mod pow5;
///
pub mod primitives;
///
pub mod rate15_params;
///
pub mod spec;

///
pub fn witness_hash<const L: usize>(
    message: Vec<Fr>,
) -> Result<Value<Fr>, Box<dyn std::error::Error>> {
    let mut hash_inputs = message.clone();
    // do the Tree dance baby
    while hash_inputs.len() > 1 {
        let mut hashes: Vec<Fr> = vec![];
        for block in hash_inputs.chunks(L) {
            let mut block = block.to_vec();
            let remainder = block.len() % L;
            if remainder != 0 {
                block.extend(vec![Fr::ZERO; L - remainder].iter());
            }
            let hash = primitives::Hash::<_, PoseidonSpec, ConstantLength<L>, 15, 14>::init()
                .hash(block.clone().try_into().unwrap());
            hashes.push(hash);
        }
        hash_inputs = hashes;
    }

    let output = hash_inputs[0];

    Ok(Value::known(output))
}
