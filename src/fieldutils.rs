/// Utilities for converting from Halo2 Field types to integers (and vice-versa).
use halo2_proofs::arithmetic::FieldExt;

/// Converts an i32 to a Field element.
pub fn i32_to_felt<F: FieldExt>(x: i32) -> F {
    if x >= 0 {
        F::from(x as u64)
    } else {
        -F::from((-x) as u64)
    }
}

/// Converts an i32 to a Field element.
pub fn i128_to_felt<F: FieldExt>(x: i128) -> F {
    if x >= 0 {
        F::from_u128(x as u128)
    } else {
        -F::from_u128((-x) as u128)
    }
}

/// Converts a Field element to an i32.
pub fn felt_to_i32<F: FieldExt>(x: F) -> i32 {
    if x > F::from(2_u64.pow(16)) {
        -((-x).get_lower_32() as i32)
    } else {
        x.get_lower_32() as i32
    }
}

/// Converts a Field element to an i32.
pub fn felt_to_i128<F: FieldExt>(x: F) -> i128 {
    if x > F::from_u128(2_u128.pow(64)) {
        -((-x).get_lower_128() as i128)
    } else {
        x.get_lower_128() as i128
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

        let res: F = i128_to_felt(-15i128);
        assert_eq!(res, -F::from(15));
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
    fn felttoi128() {
        for x in -(2i128.pow(20))..(2i128.pow(20)) {
            let fieldx: F = i128_to_felt::<F>(x);
            let xf: i128 = felt_to_i128::<F>(fieldx);
            assert_eq!(x, xf);
        }
    }
}
