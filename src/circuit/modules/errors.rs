use halo2_proofs::plonk::Error as PlonkError;
use thiserror::Error;

/// Error type for the circuit module
#[derive(Error, Debug)]
pub enum ModuleError {
    /// Halo 2 error
    #[error("Halo 2 error: {0}")]
    Halo2Error(#[from] PlonkError),
    /// Wrong input type for a module
    #[error("Wrong input type {0} must be {1}")]
    WrongInputType(String, String),
    /// A constant was not previously assigned
    #[error("Constant was not previously assigned")]
    ConstantNotAssigned,
    /// Input length is wrong
    #[error("Input length is wrong {0}")]
    InputWrongLength(usize),
}

impl From<ModuleError> for PlonkError {
    fn from(_e: ModuleError) -> PlonkError {
        PlonkError::Synthesis
    }
}
