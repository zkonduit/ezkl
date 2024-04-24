use halo2_proofs::arithmetic::Field;
/// Utilities for converting from Halo2 PrimeField types to integers (and vice-versa).
use halo2curves::ff::PrimeField;

/// Converts an i32 to a PrimeField element.
pub fn i32_to_felt<F: PrimeField>(x: i32) -> F {
    if x >= 0 {
        F::from(x as u64)
    } else {
        -F::from(x.unsigned_abs() as u64)
    }
}

/// Converts an i64 to a PrimeField element.
pub fn i64_to_felt<F: PrimeField>(x: i64) -> F {
    if x >= 0 {
        F::from_u128(x as u128)
    } else {
        -F::from_u128((-x) as u128)
    }
}

/// Converts a PrimeField element to an i32.
pub fn felt_to_i32<F: PrimeField + PartialOrd + Field>(x: F) -> i32 {
    if x > F::from(i32::MAX as u64) {
        let rep = (-x).to_repr();
        let negtmp: &[u8] = rep.as_ref();
        let lower_32 = u32::from_le_bytes(negtmp[..4].try_into().unwrap());
        -(lower_32 as i32)
    } else {
        let rep = (x).to_repr();
        let tmp: &[u8] = rep.as_ref();
        let lower_32 = u32::from_le_bytes(tmp[..4].try_into().unwrap());
        lower_32 as i32
    }
}

/// Converts a PrimeField element to an f64.
pub fn felt_to_f64<F: PrimeField + PartialOrd + Field>(x: F) -> f64 {
    if x > F::from_u128(i64::MAX as u128) {
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
pub fn felt_to_i64<F: PrimeField + PartialOrd + Field>(x: F) -> i64 {
    if x > F::from_u128(i64::MAX as u128) {
        let rep = (-x).to_repr();
        let negtmp: &[u8] = rep.as_ref();
        let lower_128: u128 = u128::from_le_bytes(negtmp[..16].try_into().unwrap());
        -(lower_128 as i64)
    } else {
        let rep = (x).to_repr();
        let tmp: &[u8] = rep.as_ref();
        let lower_128: u128 = u128::from_le_bytes(tmp[..16].try_into().unwrap());
        lower_128 as i64
    }
}

#[cfg(test)]
mod test {

    use super::*;
    use halo2curves::pasta::Fp as F;

    #[test]
    fn test_conv() {
        let res: F = i32_to_felt(-15i32);
        assert_eq!(res, -F::from(15));

        let res: F = i32_to_felt(2_i32.pow(17));
        assert_eq!(res, F::from(131072));

        let res: F = i64_to_felt(-15i64);
        assert_eq!(res, -F::from(15));

        let res: F = i64_to_felt(2_i64.pow(17));
        assert_eq!(res, F::from(131072));
    }

    #[test]
    fn felttoi32() {
        for x in -(2i32.pow(16))..(2i32.pow(16)) {
            let fieldx: F = i32_to_felt::<F>(x);
            let xf: i32 = felt_to_i32::<F>(fieldx);
            assert_eq!(x, xf);
        }
    }

    #[test]
    fn felttoi64() {
        for x in -(2i64.pow(20))..(2i64.pow(20)) {
            let fieldx: F = i64_to_felt::<F>(x);
            let xf: i64 = felt_to_i64::<F>(fieldx);
            assert_eq!(x, xf);
        }
    }
}
