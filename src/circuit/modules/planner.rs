use std::collections::HashMap;
use std::fmt;
use std::marker::PhantomData;
use std::{cmp, panic};

use halo2curves::ff::Field;

use halo2_proofs::{
    circuit::{
        layouter::{RegionColumn, RegionLayouter, RegionShape, SyncDeps, TableLayouter},
        Cell, Layouter, Region, RegionIndex, RegionStart, Table, Value,
    },
    plonk::{
        Advice, Any, Assigned, Assignment, Challenge, Circuit, Column, Error, Fixed, FloorPlanner,
        Instance, Selector, TableColumn,
    },
};
use log::{debug, trace};

/// A simple [`FloorPlanner`] that performs minimal optimizations.
#[derive(Debug)]
pub struct ModulePlanner;

impl FloorPlanner for ModulePlanner {
    fn synthesize<F: Field, CS: Assignment<F> + SyncDeps, C: Circuit<F>>(
        cs: &mut CS,
        circuit: &C,
        config: C::Config,
        constants: Vec<Column<Fixed>>,
    ) -> Result<(), Error> {
        let layouter = ModuleLayouter::new(cs, constants)?;
        circuit.synthesize(config, layouter)
    }
}
///
pub type ModuleIdx = usize;
///
pub type RegionIdx = usize;

/// A [`Layouter`] for a circuit with multiple modules.
pub struct ModuleLayouter<'a, F: Field, CS: Assignment<F> + 'a> {
    cs: &'a mut CS,
    constants: Vec<Column<Fixed>>,
    /// Stores the starting row for each region.
    regions: HashMap<ModuleIdx, HashMap<RegionIdx, RegionStart>>,
    /// Stores the starting row for each region.
    region_idx: HashMap<RegionIdx, ModuleIdx>,
    /// Stores the first empty row for each column.
    columns: HashMap<(ModuleIdx, RegionColumn), usize>,
    /// Stores the table fixed columns.
    table_columns: Vec<TableColumn>,
    _marker: PhantomData<F>,
    /// current module
    current_module: usize,
    /// num_constants
    total_constants: usize,
}

impl<'a, F: Field, CS: Assignment<F> + 'a> fmt::Debug for ModuleLayouter<'a, F, CS> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("ModuleLayouter")
            .field("regions", &self.regions)
            .field("columns", &self.columns)
            .field("total_constants", &self.total_constants)
            .finish()
    }
}

impl<'a, F: Field, CS: Assignment<F>> ModuleLayouter<'a, F, CS> {
    /// Creates a new module layouter.
    pub fn new(cs: &'a mut CS, constants: Vec<Column<Fixed>>) -> Result<Self, Error> {
        let ret = ModuleLayouter {
            cs,
            constants,
            regions: HashMap::from([(0, HashMap::default()), (1, HashMap::default())]),
            columns: HashMap::default(),
            region_idx: HashMap::default(),
            table_columns: vec![],
            current_module: 0,
            total_constants: 0,
            _marker: PhantomData,
        };
        Ok(ret)
    }
}

impl<'a, F: Field, CS: Assignment<F> + 'a + SyncDeps> Layouter<F> for ModuleLayouter<'a, F, CS> {
    type Root = Self;

    fn assign_region<A, AR, N, NR>(&mut self, name: N, mut assignment: A) -> Result<AR, Error>
    where
        A: FnMut(Region<'_, F>) -> Result<AR, Error>,
        N: Fn() -> NR,
        NR: Into<String>,
    {
        // if the name contains the required substring we increment the current module idx
        if Into::<String>::into(name()).contains("_new_module") {
            self.current_module = self.regions.keys().max().unwrap_or(&0) + 1;
        } else if Into::<String>::into(name()).contains("_enter_module_") {
            let index = Into::<String>::into(name())
                .split("_enter_module_")
                .last()
                .unwrap_or_else(|| panic!("Invalid module name"))
                .parse::<usize>()
                .unwrap_or_else(|_| panic!("Invalid module name"));
            assert!(self.regions.contains_key(&index), "module does not exist");
            self.current_module = index;
        }

        let region_index = self.region_idx.len();
        self.region_idx.insert(region_index, self.current_module);

        // Get shape of the region.
        let mut shape = RegionShape::new(region_index.into());
        {
            let region: &mut dyn RegionLayouter<F> = &mut shape;
            assignment(region.into())?;
        }

        // Modules are stacked horizontally across new columns -- THIS ASSUMES THE MODULES HAVE NON OVERLAPPING COLUMNS.
        let region_start = match self.regions.get_mut(&self.current_module) {
            Some(v) => {
                let mut region_start = 0;
                for column in shape.columns().iter() {
                    region_start = cmp::max(
                        region_start,
                        self.columns
                            .get(&(self.current_module, *column))
                            .cloned()
                            .unwrap_or(0),
                    );
                }

                v.insert(region_index, region_start.into());
                region_start
            }
            None => {
                let map = HashMap::from([(region_index, 0.into())]);
                self.regions.insert(self.current_module, map);
                0
            }
        };

        // Update column usage information.
        for column in shape.columns() {
            self.columns.insert(
                (self.current_module, *column),
                region_start + shape.row_count(),
            );
        }

        // Assign region cells.
        self.cs.enter_region(name);
        let mut region = ModuleLayouterRegion::new(self, region_index.into());
        let result = {
            let region: &mut dyn RegionLayouter<F> = &mut region;
            assignment(region.into())
        }?;
        let constants_to_assign = region.constants;
        self.cs.exit_region();

        // Assign constants. For the simple floor planner, we assign constants in order in
        // the first `constants` column.
        if self.constants.is_empty() {
            if !constants_to_assign.is_empty() {
                return Err(Error::NotEnoughColumnsForConstants);
            }
        } else {
            let constants_column = self.constants[0];

            for (constant, advice) in constants_to_assign {
                self.cs.assign_fixed(
                    || format!("Constant({:?})", constant.evaluate()),
                    constants_column,
                    self.total_constants,
                    || Value::known(constant),
                )?;

                let region_module = self.region_idx[&advice.region_index];

                self.cs.copy(
                    constants_column.into(),
                    self.total_constants,
                    advice.column,
                    *self.regions[&region_module][&advice.region_index] + advice.row_offset,
                )?;
                self.total_constants += 1;
            }
        }

        trace!("region {} assigned", region_index);
        trace!("total_constants: {:?}", self.total_constants);
        let max_row_index = self
            .columns
            .iter()
            .filter(|((module, _), _)| *module == self.current_module)
            .map(|(_, row)| *row)
            .max()
            .unwrap_or(0);
        trace!("max_row_index: {:?}", max_row_index);

        Ok(result)
    }

    fn assign_table<A, N, NR>(&mut self, name: N, mut assignment: A) -> Result<(), Error>
    where
        A: FnMut(Table<'_, F>) -> Result<(), Error>,
        N: Fn() -> NR,
        NR: Into<String>,
    {
        // Maintenance hazard: there is near-duplicate code in `v1::AssignmentPass::assign_table`.
        // Assign table cells.
        self.cs.enter_region(name);
        let mut table =
            halo2_proofs::circuit::SimpleTableLayouter::new(self.cs, &self.table_columns);
        {
            let table: &mut dyn TableLayouter<F> = &mut table;
            assignment(table.into())
        }?;
        let default_and_assigned = table.default_and_assigned;
        self.cs.exit_region();

        // Check that all table columns have the same length `first_unused`,
        // and all cells up to that length are assigned.
        let first_unused = {
            match default_and_assigned
                .values()
                .map(|(_, assigned)| {
                    if assigned.iter().all(|b| *b) {
                        Some(assigned.len())
                    } else {
                        None
                    }
                })
                .reduce(|acc, item| match (acc, item) {
                    (Some(a), Some(b)) if a == b => Some(a),
                    _ => None,
                }) {
                Some(Some(len)) => len,
                _ => return Err(Error::Synthesis), // TODO better error
            }
        };

        // Record these columns so that we can prevent them from being used again.
        for column in default_and_assigned.keys() {
            self.table_columns.push(*column);
        }

        for (col, (default_val, _)) in default_and_assigned {
            // default_val must be Some because we must have assigned
            // at least one cell in each column, and in that case we checked
            // that all cells up to first_unused were assigned.
            self.cs
                .fill_from_row(col.inner(), first_unused, default_val.unwrap())?;
        }

        Ok(())
    }

    fn constrain_instance(
        &mut self,
        cell: Cell,
        instance: Column<Instance>,
        row: usize,
    ) -> Result<(), Error> {
        let module_idx = self.region_idx[&cell.region_index];

        self.cs.copy(
            cell.column,
            *self.regions[&module_idx][&cell.region_index] + cell.row_offset,
            instance.into(),
            row,
        )
    }

    fn get_challenge(&self, challenge: Challenge) -> Value<F> {
        self.cs.get_challenge(challenge)
    }

    fn get_root(&mut self) -> &mut Self::Root {
        self
    }

    fn push_namespace<NR, N>(&mut self, name_fn: N)
    where
        NR: Into<String>,
        N: FnOnce() -> NR,
    {
        self.cs.push_namespace(name_fn)
    }

    fn pop_namespace(&mut self, gadget_name: Option<String>) {
        self.cs.pop_namespace(gadget_name)
    }
}

struct ModuleLayouterRegion<'r, 'a, F: Field, CS: Assignment<F> + 'a> {
    layouter: &'r mut ModuleLayouter<'a, F, CS>,
    region_index: RegionIndex,
    /// Stores the constants to be assigned, and the cells to which they are copied.
    constants: Vec<(Assigned<F>, Cell)>,
}

impl<'r, 'a, F: Field, CS: Assignment<F> + 'a> fmt::Debug for ModuleLayouterRegion<'r, 'a, F, CS> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("ModuleLayouterRegion")
            .field("layouter", &self.layouter)
            .field("region_index", &self.region_index)
            .finish()
    }
}

impl<'r, 'a, F: Field, CS: Assignment<F> + 'a> ModuleLayouterRegion<'r, 'a, F, CS> {
    fn new(layouter: &'r mut ModuleLayouter<'a, F, CS>, region_index: RegionIndex) -> Self {
        ModuleLayouterRegion {
            layouter,
            region_index,
            constants: vec![],
        }
    }
}

impl<'r, 'a, F: Field, CS: Assignment<F> + 'a + SyncDeps> SyncDeps
    for ModuleLayouterRegion<'r, 'a, F, CS>
{
}

impl<'r, 'a, F: Field, CS: Assignment<F> + 'a + SyncDeps> RegionLayouter<F>
    for ModuleLayouterRegion<'r, 'a, F, CS>
{
    fn enable_selector<'v>(
        &'v mut self,
        annotation: &'v (dyn Fn() -> String + 'v),
        selector: &Selector,
        offset: usize,
    ) -> Result<(), Error> {
        let module_idx = self.layouter.region_idx[&self.region_index];
        self.layouter.cs.enable_selector(
            annotation,
            selector,
            *self.layouter.regions[&module_idx][&self.region_index] + offset,
        )
    }

    fn name_column<'v>(
        &'v mut self,
        annotation: &'v (dyn Fn() -> String + 'v),
        column: Column<Any>,
    ) {
        self.layouter.cs.annotate_column(annotation, column);
    }

    fn assign_advice<'v>(
        &'v mut self,
        annotation: &'v (dyn Fn() -> String + 'v),
        column: Column<Advice>,
        offset: usize,
        to: &'v mut (dyn FnMut() -> Value<Assigned<F>> + 'v),
    ) -> Result<Cell, Error> {
        let module_idx = self.layouter.region_idx[&self.region_index];

        self.layouter.cs.assign_advice(
            annotation,
            column,
            *self.layouter.regions[&module_idx][&self.region_index] + offset,
            to,
        )?;

        Ok(Cell {
            region_index: self.region_index,
            row_offset: offset,
            column: column.into(),
        })
    }

    fn assign_advice_from_constant<'v>(
        &'v mut self,
        annotation: &'v (dyn Fn() -> String + 'v),
        column: Column<Advice>,
        offset: usize,
        constant: Assigned<F>,
    ) -> Result<Cell, Error> {
        let advice =
            self.assign_advice(annotation, column, offset, &mut || Value::known(constant))?;
        self.constrain_constant(advice, constant)?;

        Ok(advice)
    }

    fn assign_advice_from_instance<'v>(
        &mut self,
        annotation: &'v (dyn Fn() -> String + 'v),
        instance: Column<Instance>,
        row: usize,
        advice: Column<Advice>,
        offset: usize,
    ) -> Result<(Cell, Value<F>), Error> {
        let value = self.layouter.cs.query_instance(instance, row)?;

        let cell = self.assign_advice(annotation, advice, offset, &mut || value.to_field())?;
        let module_idx = self.layouter.region_idx[&cell.region_index];

        self.layouter.cs.copy(
            cell.column,
            *self.layouter.regions[&module_idx][&cell.region_index] + cell.row_offset,
            instance.into(),
            row,
        )?;

        Ok((cell, value))
    }

    fn assign_fixed<'v>(
        &'v mut self,
        annotation: &'v (dyn Fn() -> String + 'v),
        column: Column<Fixed>,
        offset: usize,
        to: &'v mut (dyn FnMut() -> Value<Assigned<F>> + 'v),
    ) -> Result<Cell, Error> {
        let module_idx = self.layouter.region_idx[&self.region_index];

        self.layouter.cs.assign_fixed(
            annotation,
            column,
            *self.layouter.regions[&module_idx][&self.region_index] + offset,
            to,
        )?;

        Ok(Cell {
            region_index: self.region_index,
            row_offset: offset,
            column: column.into(),
        })
    }

    fn constrain_constant(&mut self, cell: Cell, constant: Assigned<F>) -> Result<(), Error> {
        self.constants.push((constant, cell));
        Ok(())
    }

    fn constrain_equal(&mut self, left: Cell, right: Cell) -> Result<(), Error> {
        let left_module = self.layouter.region_idx[&left.region_index];
        let right_module = self.layouter.region_idx[&right.region_index];

        self.layouter.cs.copy(
            left.column,
            *self.layouter.regions[&left_module][&left.region_index] + left.row_offset,
            right.column,
            *self.layouter.regions[&right_module][&right.region_index] + right.row_offset,
        )?;

        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use halo2curves::pasta::vesta;

    use super::ModulePlanner;
    use halo2_proofs::{
        dev::MockProver,
        plonk::{Advice, Circuit, Column, Error},
    };

    #[test]
    fn not_enough_columns_for_constants() {
        struct MyCircuit {}

        impl Circuit<vesta::Scalar> for MyCircuit {
            type Config = Column<Advice>;
            type FloorPlanner = ModulePlanner;
            type Params = ();

            fn without_witnesses(&self) -> Self {
                MyCircuit {}
            }

            fn configure(
                meta: &mut halo2_proofs::plonk::ConstraintSystem<vesta::Scalar>,
            ) -> Self::Config {
                meta.advice_column()
            }

            fn synthesize(
                &self,
                config: Self::Config,
                mut layouter: impl halo2_proofs::circuit::Layouter<vesta::Scalar>,
            ) -> Result<(), halo2_proofs::plonk::Error> {
                layouter.assign_region(
                    || "assign constant",
                    |mut region| {
                        region.assign_advice_from_constant(
                            || "one",
                            config,
                            0,
                            vesta::Scalar::one(),
                        )
                    },
                )?;

                Ok(())
            }
        }

        let circuit = MyCircuit {};
        assert!(matches!(
            MockProver::run(3, &circuit, vec![]).unwrap_err(),
            Error::NotEnoughColumnsForConstants,
        ));
    }
}
