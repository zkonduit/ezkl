//! The Poseidon algebraic hash function.

use std::convert::TryInto;
use std::fmt;
use std::marker::PhantomData;

use halo2_proofs::{
    circuit::{AssignedCell, Chip, Region},
    plonk::Error,
};
use halo2curves::ff::Field;
use halo2curves::ff::PrimeField;

use super::primitives::{Absorbing, ConstantLength, Domain, Spec, SpongeMode, Squeezing};

pub(crate) type State<F, const T: usize> = [F; T];

/// A word from the padded input to a Poseidon sponge.
#[derive(Clone, Debug)]
pub enum PaddedWord<F: Field> {
    /// A message word provided by the prover.
    Message(AssignedCell<F, F>),
    /// A padding word, that will be fixed in the circuit parameters.
    Padding(F),
}

/// The set of circuit instructions required to use the Poseidon permutation.
pub trait PoseidonInstructions<F: Field, S: Spec<F, T, RATE>, const T: usize, const RATE: usize>:
    Chip<F>
{
    /// Variable representing the word over which the Poseidon permutation operates.
    type Word: Clone + fmt::Debug + From<AssignedCell<F, F>> + Into<AssignedCell<F, F>>;

    /// Applies the Poseidon permutation to the given state.
    fn permute(
        &self,
        region: &mut Region<F>,
        initial_state: &State<Self::Word, T>,
        offset: &mut usize,
    ) -> Result<State<Self::Word, T>, Error>;
}

/// The set of circuit instructions required to use the [`Sponge`] and [`Hash`] gadgets.
///
/// [`Hash`]: self::Hash
pub trait PoseidonSpongeInstructions<
    F: Field,
    S: Spec<F, T, RATE>,
    D: Domain<F, RATE>,
    const T: usize,
    const RATE: usize,
>: PoseidonInstructions<F, S, T, RATE>
{
    /// Returns the initial empty state for the given domain.
    fn initial_state(
        &self,
        region: &mut Region<F>,
        offset: &mut usize,
    ) -> Result<State<Self::Word, T>, Error>;

    /// Adds the given input to the state.
    fn add_input(
        &self,
        region: &mut Region<F>,
        initial_state: &State<Self::Word, T>,
        input: &Absorbing<PaddedWord<F>, RATE>,
    ) -> Result<State<Self::Word, T>, Error>;

    /// Extracts sponge output from the given state.
    fn get_output(state: &State<Self::Word, T>) -> Squeezing<Self::Word, RATE>;
}

/// A word over which the Poseidon permutation operates.
#[derive(Debug)]
pub struct Word<
    F: Field,
    PoseidonChip: PoseidonInstructions<F, S, T, RATE>,
    S: Spec<F, T, RATE>,
    const T: usize,
    const RATE: usize,
> {
    inner: PoseidonChip::Word,
}

impl<
        F: Field,
        PoseidonChip: PoseidonInstructions<F, S, T, RATE>,
        S: Spec<F, T, RATE>,
        const T: usize,
        const RATE: usize,
    > Word<F, PoseidonChip, S, T, RATE>
{
    /// The word contained in this gadget.
    pub fn inner(&self) -> PoseidonChip::Word {
        self.inner.clone()
    }

    /// Construct a [`Word`] gadget from the inner word.
    pub fn from_inner(inner: PoseidonChip::Word) -> Self {
        Self { inner }
    }
}

fn poseidon_sponge<
    F: Field,
    PoseidonChip: PoseidonSpongeInstructions<F, S, D, T, RATE>,
    S: Spec<F, T, RATE>,
    D: Domain<F, RATE>,
    const T: usize,
    const RATE: usize,
>(
    chip: &PoseidonChip,
    region: &mut Region<F>,
    state: &mut State<PoseidonChip::Word, T>,
    input: Option<&Absorbing<PaddedWord<F>, RATE>>,
    offset: &mut usize,
) -> Result<Squeezing<PoseidonChip::Word, RATE>, Error> {
    if let Some(input) = input {
        *state = chip.add_input(region, state, input)?;
    }
    *state = chip.permute(region, state, offset)?;
    Ok(PoseidonChip::get_output(state))
}

/// A Poseidon sponge.
#[derive(Debug)]
pub struct Sponge<
    F: Field,
    PoseidonChip: PoseidonSpongeInstructions<F, S, D, T, RATE>,
    S: Spec<F, T, RATE>,
    M: SpongeMode,
    D: Domain<F, RATE>,
    const T: usize,
    const RATE: usize,
> {
    chip: PoseidonChip,
    mode: M,
    state: State<PoseidonChip::Word, T>,
    _marker: PhantomData<D>,
}

impl<
        F: Field,
        PoseidonChip: PoseidonSpongeInstructions<F, S, D, T, RATE>,
        S: Spec<F, T, RATE>,
        D: Domain<F, RATE>,
        const T: usize,
        const RATE: usize,
    > Sponge<F, PoseidonChip, S, Absorbing<PaddedWord<F>, RATE>, D, T, RATE>
{
    /// Constructs a new duplex sponge for the given Poseidon specification.
    pub fn new(
        chip: PoseidonChip,
        region: &mut Region<F>,
        offset: &mut usize,
    ) -> Result<Self, Error> {
        chip.initial_state(region, offset).map(|state| Sponge {
            chip,
            mode: Absorbing(
                (0..RATE)
                    .map(|_| None)
                    .collect::<Vec<_>>()
                    .try_into()
                    .unwrap(),
            ),
            state,
            _marker: PhantomData::default(),
        })
    }

    /// Absorbs an element into the sponge.
    pub fn absorb(
        &mut self,
        region: &mut Region<F>,
        value: PaddedWord<F>,
        offset: &mut usize,
    ) -> Result<(), Error> {
        for entry in self.mode.0.iter_mut() {
            if entry.is_none() {
                *entry = Some(value);
                return Ok(());
            }
        }

        // We've already absorbed as many elements as we can
        let _ = poseidon_sponge(
            &self.chip,
            region,
            &mut self.state,
            Some(&self.mode),
            offset,
        )?;
        self.mode = Absorbing::init_with(value);

        Ok(())
    }

    /// Transitions the sponge into its squeezing state.
    #[allow(clippy::type_complexity)]
    pub fn finish_absorbing(
        mut self,
        region: &mut Region<F>,
        offset: &mut usize,
    ) -> Result<Sponge<F, PoseidonChip, S, Squeezing<PoseidonChip::Word, RATE>, D, T, RATE>, Error>
    {
        let mode = poseidon_sponge(
            &self.chip,
            region,
            &mut self.state,
            Some(&self.mode),
            offset,
        )?;

        Ok(Sponge {
            chip: self.chip,
            mode,
            state: self.state,
            _marker: PhantomData::default(),
        })
    }
}

impl<
        F: Field,
        PoseidonChip: PoseidonSpongeInstructions<F, S, D, T, RATE>,
        S: Spec<F, T, RATE>,
        D: Domain<F, RATE>,
        const T: usize,
        const RATE: usize,
    > Sponge<F, PoseidonChip, S, Squeezing<PoseidonChip::Word, RATE>, D, T, RATE>
{
    /// Squeezes an element from the sponge.
    pub fn squeeze(
        &mut self,
        region: &mut Region<F>,
        offset: &mut usize,
    ) -> Result<AssignedCell<F, F>, Error> {
        loop {
            for entry in self.mode.0.iter_mut() {
                if let Some(inner) = entry.take() {
                    return Ok(inner.into());
                }
            }

            // We've already squeezed out all available elements
            self.mode = poseidon_sponge(&self.chip, region, &mut self.state, None, offset)?;
        }
    }
}

/// A Poseidon hash function, built around a sponge.
#[derive(Debug)]
pub struct Hash<
    F: Field,
    PoseidonChip: PoseidonSpongeInstructions<F, S, D, T, RATE>,
    S: Spec<F, T, RATE>,
    D: Domain<F, RATE>,
    const T: usize,
    const RATE: usize,
> {
    sponge: Sponge<F, PoseidonChip, S, Absorbing<PaddedWord<F>, RATE>, D, T, RATE>,
}

impl<
        F: Field,
        PoseidonChip: PoseidonSpongeInstructions<F, S, D, T, RATE>,
        S: Spec<F, T, RATE>,
        D: Domain<F, RATE>,
        const T: usize,
        const RATE: usize,
    > Hash<F, PoseidonChip, S, D, T, RATE>
{
    /// Initializes a new hasher.
    pub fn init(
        chip: PoseidonChip,
        region: &mut Region<F>,
        offset: &mut usize,
    ) -> Result<Self, Error> {
        Sponge::new(chip, region, offset).map(|sponge| Hash { sponge })
    }
}

impl<
        F: PrimeField,
        PoseidonChip: PoseidonSpongeInstructions<F, S, ConstantLength<L>, T, RATE>,
        S: Spec<F, T, RATE>,
        const T: usize,
        const RATE: usize,
        const L: usize,
    > Hash<F, PoseidonChip, S, ConstantLength<L>, T, RATE>
{
    /// Hashes the given input.
    pub fn hash(
        mut self,
        region: &mut Region<F>,
        message: [AssignedCell<F, F>; L],
        offset: &mut usize,
    ) -> Result<AssignedCell<F, F>, Error> {
        for (_, value) in message
            .into_iter()
            .map(PaddedWord::Message)
            .chain(<ConstantLength<L> as Domain<F, RATE>>::padding(L).map(PaddedWord::Padding))
            .enumerate()
        {
            self.sponge.absorb(region, value, offset)?;
        }
        self.sponge
            .finish_absorbing(region, offset)?
            .squeeze(region, offset)
    }
}
