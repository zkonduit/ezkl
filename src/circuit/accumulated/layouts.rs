use super::*;
/// Assigns variables to the regions created when calling `configure`.
/// # Arguments
/// * `values` - The explicit values to the operations.
/// * `layouter` - A Halo2 Layouter.
pub fn dot<F: FieldExt + TensorType>(
    config: &mut BaseConfig<F>,
    layouter: &mut impl Layouter<F>,
    values: &[ValTensor<F>; 2],
    offset: usize,
) -> Result<ValTensor<F>, Box<dyn Error>> {
    if values.len() != config.inputs.len() {
        return Err(Box::new(CircuitError::DimMismatch(
            "accum dot layout".to_string(),
        )));
    }

    let t = match layouter.assign_region(
        || "assign inputs",
        |mut region| {
            let mut inputs = vec![];
            for (i, input) in values.iter().enumerate() {
                let inp = utils::value_muxer(
                    &config.inputs[i],
                    &{
                        let res = config.inputs[i].assign(&mut region, offset, input)?;
                        res.map(|e| e.value_field().evaluate())
                    },
                    input,
                );
                inputs.push(inp);
            }

            // Now we can assign the dot product
            let accumulated_dot = accumulated::dot(&inputs)
                .expect("accum poly: dot op failed")
                .into();
            let output = config
                .output
                .assign(&mut region, offset, &accumulated_dot)?;

            for i in 0..inputs[0].len() {
                let (_, y) = config.inputs[0].cartesian_coord(i);
                if y == 0 {
                    config
                        .selectors
                        .get(&BaseOp::InitDot)
                        .unwrap()
                        .enable(&mut region, offset + y)?;
                } else {
                    config
                        .selectors
                        .get(&BaseOp::Dot)
                        .unwrap()
                        .enable(&mut region, offset + y)?;
                }
            }

            // last element is the result
            Ok(output
                .get_slice(&[output.len() - 1..output.len()])
                .expect("accum poly: failed to fetch last elem"))
        },
    ) {
        Ok(a) => a,
        Err(e) => {
            return Err(Box::new(e));
        }
    };

    Ok(ValTensor::from(t))
}

/// Assigns variables to the regions created when calling `configure`.
/// # Arguments
/// * `values` - The explicit values to the operations.
/// * `layouter` - A Halo2 Layouter.
pub fn matmul<F: FieldExt + TensorType>(
    config: &mut BaseConfig<F>,
    layouter: &mut impl Layouter<F>,
    values: &[ValTensor<F>; 2],
    offset: usize,
) -> Result<ValTensor<F>, Box<dyn Error>> {
    if values.len() != 2 {
        return Err(Box::new(CircuitError::DimMismatch(
            "accum matmul layout".to_string(),
        )));
    };

    let mut a = values[0].clone();
    let mut b = values[1].clone();
    b.transpose_2d()?;

    let num_a_repeats = b.dims()[0];
    let num_b_tiles = a.dims()[1];
    let b_row_len = b.dims()[1];

    a.repeat_rows(num_a_repeats)?;
    b.tile(num_b_tiles)?;

    let t = match layouter.assign_region(
        || "assign inputs",
        |mut region| {
            let mut inputs = vec![];

            for (i, elem) in vec![a.clone(), b.clone()].iter().enumerate() {
                let inp = utils::value_muxer(
                    &config.inputs[i],
                    &{
                        let res = config.inputs[i].assign(&mut region, offset, elem)?;
                        res.map(|e| e.value_field().evaluate())
                    },
                    elem,
                );
                inputs.push(inp);
            }

            // remove any repeats from the assignment
            if num_a_repeats > 1 {
                let dims = inputs[0].dims().to_vec();
                inputs[0].reshape(&[dims[0], dims[1..].iter().product()]);
                let mut rm_dup = vec![];
                for i in 0..dims[0] {
                    rm_dup.push(inputs[0].get_slice(&[i..i + 1, 0..dims[1]]).unwrap());
                }
                inputs[0] = Tensor::new(Some(&rm_dup), &[rm_dup.len()])
                    .unwrap()
                    .combine()
                    .unwrap();
            }

            inputs[0].reshape(values[0].dims());

            // transpose it back to its normal shape
            inputs[1] = inputs[1].get_slice(&[0..1]).unwrap();
            inputs[1].reshape(&[values[1].dims()[1], values[1].dims()[0]]);
            inputs[1].transpose_2d().unwrap();

            // now perform matrix multiplication on the processed tensors
            let accumulated_matmul =
                accumulated::matmul(&vec![inputs[0].clone(), inputs[1].clone()])
                    .expect("accum poly: matmul op failed");

            let output = config
                .output
                .assign(&mut region, offset, &accumulated_matmul.into())?;

            // these selectors map from
            for i in 0..a.dims().iter().product::<usize>() {
                let (_, y) = config.inputs[0].cartesian_coord(i);
                if (i) % b_row_len > 0 {
                    config
                        .selectors
                        .get(&BaseOp::Dot)
                        .unwrap()
                        .enable(&mut region, offset + y)?;
                } else {
                    config
                        .selectors
                        .get(&BaseOp::InitDot)
                        .unwrap()
                        .enable(&mut region, offset + y)?;
                }
            }

            let dims = output.dims();
            let mut last_dims = vec![];

            for d in &dims[0..dims.len() - 1] {
                last_dims.push(0..*d);
            }
            let script_len = dims.last().unwrap();
            last_dims.push(script_len - 1..*script_len);

            let mut last_elem = output
                .get_slice(&last_dims)
                .expect("accum poly: failed to fetch last elem");

            last_elem.reshape(&[values[0].dims()[0], values[1].dims()[1]]);
            // Now we can assign the matmul op
            Ok(last_elem)
        },
    ) {
        Ok(a) => a,
        Err(e) => {
            return Err(Box::new(e));
        }
    };

    Ok(ValTensor::from(t))
}

/// Assigns variables to the regions created when calling `configure`.
/// # Arguments
/// * `values` - The explicit values to the operations.
/// * `layouter` - A Halo2 Layouter.
pub fn affine<F: FieldExt + TensorType>(
    config: &mut BaseConfig<F>,
    layouter: &mut impl Layouter<F>,
    values: &[ValTensor<F>; 3],
    offset: usize,
) -> Result<ValTensor<F>, Box<dyn Error>> {
    let (kernel, bias, mut input) = (values[0].clone(), values[1].clone(), values[2].clone());

    input.pad_row_ones()?;
    let params = kernel.append_to_row(bias)?;

    matmul(config, layouter, &[params, input], offset)
}
