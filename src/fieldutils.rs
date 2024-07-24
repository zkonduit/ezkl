use halo2_proofs::arithmetic::Field;
/// Utilities for converting from Halo2 PrimeField types to integers (and vice-versa).
use halo2curves::ff::PrimeField;

/// Integer representation of a PrimeField element.
pub type IntegerRep = i128;

/// Converts an i64 to a PrimeField element.
pub fn integer_rep_to_felt<F: PrimeField>(x: IntegerRep) -> F {
    if x >= 0 {
        F::from_u128(x as u128)
    } else {
        -F::from_u128(x.saturating_neg() as u128)
    }
}

/// Converts a PrimeField element to an f64.
pub fn felt_to_f64<F: PrimeField + PartialOrd + Field>(x: F) -> f64 {
    if x > F::from_u128(IntegerRep::MAX as u128) {
        let rep = (-x).to_repr();
        let negtmp: &[u8] = rep.as_ref();
        let lower_128: u128 = u128::from_le_bytes(negtmp[..16].try_into().unwrap());
        -(lower_128 as f64)
    } else {
        let rep = (x).to_repr();
        let tmp: &[u8] = rep.as_ref();
        let lower_128: u128 = u128::from_le_bytes(tmp[..16].try_into().unwrap());
        lower_128 as f64
    }
}

/// Converts a PrimeField element to an i64.
pub fn felt_to_integer_rep<F: PrimeField + PartialOrd + Field>(x: F) -> IntegerRep {
    if x > F::from_u128(IntegerRep::MAX as u128) {
        let rep = (-x).to_repr();
        let negtmp: &[u8] = rep.as_ref();
        let lower_128: u128 = u128::from_le_bytes(negtmp[..16].try_into().unwrap());
        -(lower_128 as IntegerRep)
    } else {
        let rep = (x).to_repr();
        let tmp: &[u8] = rep.as_ref();
        let lower_128: u128 = u128::from_le_bytes(tmp[..16].try_into().unwrap());
        lower_128 as IntegerRep
    }
}

#[cfg(test)]
mod test {

    use super::*;
    use halo2curves::pasta::Fp as F;

    #[test]
    fn test_conv() {
        let res: F = integer_rep_to_felt(-15);
        assert_eq!(res, -F::from(15));

        let res: F = integer_rep_to_felt(2_i128.pow(17));
        assert_eq!(res, F::from(131072));

        let res: F = integer_rep_to_felt(-15);
        assert_eq!(res, -F::from(15));

        let res: F = integer_rep_to_felt(2_i128.pow(17));
        assert_eq!(res, F::from(131072));
    }

    #[test]
    fn felttointegerrep() {
        for x in -(2_i128.pow(16))..(2_i128.pow(16)) {
            let fieldx: F = integer_rep_to_felt::<F>(x);
            let xf: i128 = felt_to_integer_rep::<F>(fieldx);
            assert_eq!(x, xf);
        }
    }
}
