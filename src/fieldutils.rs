use halo2_proofs::arithmetic::FieldExt;

pub fn i32toF<F: FieldExt>(x: i32) -> F {
    if x >= 0 {
        F::from(x as u64)
    } else {
        -F::from((-x) as u64)
    }
}

mod test {
    use super::*;
    use halo2_proofs::pasta::Fp as F;

    #[test]
    fn test_conv() {
        let res: F = i32toF(-15i32);
        assert_eq!(res, -F::from(15));
    }
}
