use halo2_proofs::arithmetic::FieldExt;

pub fn i32tofelt<F: FieldExt>(x: i32) -> F {
    if x >= 0 {
        F::from(x as u64)
    } else {
        -F::from((-x) as u64)
    }
}

fn felt_to_u16<F: FieldExt>(x: F) -> u16 {
    x.get_lower_32() as u16
}

fn felt_to_u32<F: FieldExt>(x: F) -> u32 {
    x.get_lower_32() as u32
}

pub fn felt_to_i32<F: FieldExt>(x: F) -> i32 {
    if x > F::from(65536) {
        -(felt_to_u32(-x) as i32)
    } else {
        felt_to_u32(x) as i32
    }
}

mod test {
    use super::*;
    use halo2curves::pasta::Fp as F;

    #[test]
    fn test_conv() {
        let res: F = i32tofelt(-15i32);
        assert_eq!(res, -F::from(15));
    }

    #[test]
    fn felttoi32() {
        for x in -(2i32.pow(15))..(2i32.pow(15)) {
            let fieldx: F = i32tofelt::<F>(x);
            let xf: i32 = felt_to_i32::<F>(fieldx);
            assert_eq!(x, xf);
        }
    }
}
