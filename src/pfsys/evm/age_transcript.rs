use super::*;
use halo2_proofs::transcript::{Challenge255, Transcript};
use snark_verifier::loader::native::NativeLoader;

/// Age-optimized transcript based on Poseidon with reduced parameters
pub struct AgeTranscript<L, S>(PoseidonTranscript<L, S>);

impl<L: EcPointLoader<G1Affine>, S: Clone> AgeTranscript<L, S> {
    pub fn new(state: S) -> Self {
        // Use reduced-round Poseidon internally
        const AGE_R_P: usize = 30; // Even further reduced for age verification
        
        let inner = PoseidonTranscript::<L, S>::new(state);
        Self(inner)
    }
}

// Implement Transcript trait (delegating to inner with optimized parameters)
impl<L: EcPointLoader<G1Affine>, S: Clone> Transcript<G1Affine, Challenge255<G1Affine>>
    for AgeTranscript<L, S>
{
    // Implementation delegating to inner PoseidonTranscript with appropriate methods
    // ...
} 