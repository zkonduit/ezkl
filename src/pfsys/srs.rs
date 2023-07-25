use halo2_proofs::poly::commitment::CommitmentScheme;
use halo2_proofs::poly::commitment::Params;
use halo2_proofs::poly::commitment::ParamsProver;
use log::info;
use std::error::Error;
use std::fs::File;
use std::io::BufReader;
use std::path::PathBuf;

/// for now we use the urls of the powers of tau ceremony from <https://github.com/han0110/halo2-kzg-srs>
pub const PUBLIC_SRS_URL: &str =
    "https://trusted-setup-halo2kzg.s3.eu-central-1.amazonaws.com/perpetual-powers-of-tau-raw-";

/// Helper function for generating SRS. Only use for testing
pub fn gen_srs<Scheme: CommitmentScheme>(k: u32) -> Scheme::ParamsProver {
    Scheme::ParamsProver::new(k)
}

/// Loads the [CommitmentScheme::ParamsVerifier] at `path`.
pub fn load_srs<Scheme: CommitmentScheme>(
    path: PathBuf,
) -> Result<Scheme::ParamsVerifier, Box<dyn Error>> {
    info!("loading srs from {:?}", path);
    let f = File::open(path)?;
    let mut reader = BufReader::new(f);
    Params::<'_, Scheme::Curve>::read(&mut reader).map_err(Box::<dyn Error>::from)
}
