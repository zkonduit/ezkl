use crate::tensor::{Tensor, TensorType};
use halo2_proofs::{
    arithmetic::FieldExt,
    circuit::{AssignedCell, Region, Value},
    plonk::{Advice, Assigned, Column},
};

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

// just an internal function to help assign. should not be used outside module
pub fn assign_advice_tensor<F: Clone + FieldExt + TensorType>(
    input: &mut Tensor<Value<Assigned<F>>>,
    region: &mut Region<'_, F>,
    name: &str,
    columns: &[Column<Advice>],
    offset: usize,
) -> Result<Tensor<AssignedCell<Assigned<F>, F>>, halo2_proofs::plonk::Error> {
    assert!(input.len() <= 2);
    let dims = input.dims();
    let mut eq = Vec::new();
    if dims.len() == 1 {
        for i in 0..dims[0] {
            let v = region.assign_advice(
                || format!("{}", name),
                columns[0],
                offset + i,
                || input.get(&[i]),
            )?;
            eq.push(v);
        }
    } else if dims.len() == 2 {
        for (i, col) in columns.iter().enumerate().take(dims[0]) {
            for j in 0..dims[1] {
                let weight = region.assign_advice(
                    || format!("{}", name),
                    *col,
                    offset + j,
                    || input.get(&[i, j]),
                )?;
                eq.push(weight);
            }
        }
    } else {
        panic!("should never reach here")
    }
    Ok(Tensor::new(Some(&eq), dims).unwrap())
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
