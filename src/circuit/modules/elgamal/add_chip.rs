use halo2_proofs::{
    circuit::{AssignedCell, Chip, Layouter},
    plonk::{self, Advice, Column, ConstraintSystem, Constraints, Selector},
    poly::Rotation,
};
use halo2curves::bn256::Fr;
use halo2curves::ff::PrimeField;

/// An instruction set for adding two circuit words (field elements).
pub(crate) trait AddInstruction<F: PrimeField>: Chip<F> {
    /// Constraints `a + b` and returns the sum.
    fn add(
        &self,
        layouter: impl Layouter<F>,
        a: &AssignedCell<F, F>,
        b: &AssignedCell<F, F>,
    ) -> Result<AssignedCell<F, F>, plonk::Error>;
}

#[derive(Clone, Debug)]
pub struct AddConfig {
    a: Column<Advice>,
    b: Column<Advice>,
    c: Column<Advice>,
    q_add: Selector,
}

/// A chip implementing a single addition constraint `c = a + b` on a single row.
#[derive(Clone, Debug)]
pub struct AddChip {
    config: AddConfig,
}

impl Chip<Fr> for AddChip {
    type Config = AddConfig;
    type Loaded = ();

    fn config(&self) -> &Self::Config {
        &self.config
    }

    fn loaded(&self) -> &Self::Loaded {
        &()
    }
}

impl AddChip {
    pub fn configure(
        meta: &mut ConstraintSystem<Fr>,
        a: Column<Advice>,
        b: Column<Advice>,
        c: Column<Advice>,
    ) -> AddConfig {
        let q_add = meta.selector();
        meta.create_gate("Field element addition: c = a + b", |meta| {
            let q_add = meta.query_selector(q_add);
            let a = meta.query_advice(a, Rotation::cur());
            let b = meta.query_advice(b, Rotation::cur());
            let c = meta.query_advice(c, Rotation::cur());

            Constraints::with_selector(q_add, Some(a + b - c))
        });

        AddConfig { a, b, c, q_add }
    }

    pub fn construct(config: AddConfig) -> Self {
        Self { config }
    }
}

impl AddInstruction<Fr> for AddChip {
    fn add(
        &self,
        mut layouter: impl Layouter<Fr>,
        a: &AssignedCell<Fr, Fr>,
        b: &AssignedCell<Fr, Fr>,
    ) -> Result<AssignedCell<Fr, Fr>, plonk::Error> {
        layouter.assign_region(
            || "c = a + b",
            |mut region| {
                self.config.q_add.enable(&mut region, 0)?;

                a.copy_advice(|| "copy a", &mut region, self.config.a, 0)?;
                b.copy_advice(|| "copy b", &mut region, self.config.b, 0)?;

                let scalar_val = a.value().zip(b.value()).map(|(a, b)| a + b);
                region.assign_advice(|| "c", self.config.c, 0, || scalar_val)
            },
        )
    }
}
