#[cfg(all(target_arch = "wasm32", target_os = "unknown"))]
#[cfg(test)]
mod wasm32 {
    use ark_std::test_rng;
    use ezkl::circuit::modules::elgamal::ElGamalVariables;
    use ezkl::circuit::modules::poseidon::spec::{PoseidonSpec, POSEIDON_RATE, POSEIDON_WIDTH};
    use ezkl::circuit::modules::poseidon::PoseidonChip;
    use ezkl::circuit::modules::Module;
    use ezkl::graph::modules::POSEIDON_LEN_GRAPH;
    use ezkl::graph::GraphWitness;
    use ezkl::pfsys::Snark;
    use ezkl::wasm::{
        elgamalDecrypt, elgamalEncrypt, elgamalGenRandom, genWitness, poseidonHash, prove,
        vecU64ToFelt, vecU64ToFloat, vecU64ToInt, verify,
    };
    use halo2curves::bn256::{Fr, G1Affine};
    use rand::rngs::StdRng;
    use rand::SeedableRng;
    #[cfg(feature = "web")]
    pub use wasm_bindgen_rayon::init_thread_pool;
    use wasm_bindgen_test::*;

    wasm_bindgen_test::wasm_bindgen_test_configure!(run_in_browser);

    pub const KZG_PARAMS: &[u8] = include_bytes!("../tests/wasm/kzg");
    pub const CIRCUIT_PARAMS: &[u8] = include_bytes!("../tests/wasm/settings.json");
    pub const VK: &[u8] = include_bytes!("../tests/wasm/test.key");
    pub const PK: &[u8] = include_bytes!("../tests/wasm/test.provekey");
    pub const WITNESS: &[u8] = include_bytes!("../tests/wasm/test.witness.json");
    pub const PROOF: &[u8] = include_bytes!("../tests/wasm/test.proof");
    pub const NETWORK: &[u8] = include_bytes!("../tests/wasm/network.compiled");
    pub const INPUT: &[u8] = include_bytes!("../tests/wasm/input.json");

    #[wasm_bindgen_test]
    async fn verify_field_serialization_roundtrip() {
        for i in 0..32 {
            let field_element = Fr::from(i);
            let serialized = serde_json::to_vec(&field_element).unwrap();
            let clamped = wasm_bindgen::Clamped(serialized);
            let scale = 2;
            let floating_point = vecU64ToFloat(clamped.clone(), scale);
            assert_eq!(floating_point, (i as f64) / 4.0);

            let integer: i128 = serde_json::from_slice(&vecU64ToInt(clamped.clone())).unwrap();
            assert_eq!(integer, i as i128);

            let hex_string = format!("{:?}", field_element);
            let returned_string = vecU64ToFelt(clamped);
            assert_eq!(hex_string, returned_string);
        }
    }

    #[wasm_bindgen_test]
    async fn verify_elgamal_gen_random_wasm() {
        // Generate a seed value
        let seed = [0u8; 32];

        // Convert the seed to a wasm-friendly format
        let wasm_seed = wasm_bindgen::Clamped(seed.to_vec());

        // Use the seed to generate ElGamal variables via WASM function
        let wasm_output = elgamalGenRandom(wasm_seed);

        let wasm_vars: ElGamalVariables = serde_json::from_slice(&wasm_output[..]).unwrap();

        // Use the same seed to generate ElGamal variables directly
        let mut rng_from_seed = StdRng::from_seed(seed);
        let direct_vars = ElGamalVariables::gen_random(&mut rng_from_seed);

        // Check if both variables are the same
        assert_eq!(direct_vars, wasm_vars)
    }

    #[wasm_bindgen_test]
    async fn verify_elgamal_wasm() {
        let mut rng = test_rng();

        let var = ElGamalVariables::gen_random(&mut rng);

        let mut message: Vec<Fr> = vec![];
        for i in 0..32 {
            message.push(Fr::from(i as u64));
        }

        let pk = serde_json::to_vec(&var.pk).unwrap();
        let message_ser = serde_json::to_vec(&message).unwrap();
        let r = serde_json::to_vec(&var.r).unwrap();

        let cipher = elgamalEncrypt(
            wasm_bindgen::Clamped(pk.clone()),
            wasm_bindgen::Clamped(message_ser.clone()),
            wasm_bindgen::Clamped(r.clone()),
        );

        let sk = serde_json::to_vec(&var.sk).unwrap();

        let decrypted_message =
            elgamalDecrypt(wasm_bindgen::Clamped(cipher), wasm_bindgen::Clamped(sk));

        let decrypted_message: Vec<Fr> = serde_json::from_slice(&decrypted_message[..]).unwrap();

        assert_eq!(message, decrypted_message)
    }

    #[wasm_bindgen_test]
    async fn verify_hash() {
        let mut message: Vec<Fr> = vec![];
        for i in 0..32 {
            message.push(Fr::from(i as u64));
        }

        let message_ser = serde_json::to_vec(&message).unwrap();

        let hash = poseidonHash(wasm_bindgen::Clamped(message_ser));
        let hash: Vec<Vec<Fr>> = serde_json::from_slice(&hash[..]).unwrap();

        let reference_hash =
            PoseidonChip::<PoseidonSpec, POSEIDON_WIDTH, POSEIDON_RATE, POSEIDON_LEN_GRAPH>::run(
                message.clone(),
            )
            .unwrap();

        assert_eq!(hash, reference_hash)
    }

    #[wasm_bindgen_test]
    async fn verify_gen_witness() {
        let witness = genWitness(
            wasm_bindgen::Clamped(NETWORK.to_vec()),
            wasm_bindgen::Clamped(INPUT.to_vec()),
            wasm_bindgen::Clamped(CIRCUIT_PARAMS.to_vec()),
        );

        let witness: GraphWitness = serde_json::from_slice(&witness[..]).unwrap();

        let reference_witness: GraphWitness = serde_json::from_slice(&WITNESS).unwrap();
        // should not fail
        assert_eq!(witness, reference_witness);
    }

    #[wasm_bindgen_test]
    async fn verify_pass() {
        let value = verify(
            wasm_bindgen::Clamped(PROOF.to_vec()),
            wasm_bindgen::Clamped(VK.to_vec()),
            wasm_bindgen::Clamped(CIRCUIT_PARAMS.to_vec()),
            wasm_bindgen::Clamped(KZG_PARAMS.to_vec()),
        );
        assert!(value);
    }

    #[wasm_bindgen_test]
    async fn verify_fail() {
        let og_proof: Snark<Fr, G1Affine> = serde_json::from_slice(&PROOF).unwrap();

        let proof: Snark<Fr, G1Affine> = Snark {
            proof: vec![0; 32],
            protocol: og_proof.protocol,
            instances: vec![vec![Fr::from(0); 32]],
            transcript_type: ezkl::pfsys::TranscriptType::EVM,
        };
        let proof = serde_json::to_string(&proof).unwrap().into_bytes();

        let value = verify(
            wasm_bindgen::Clamped(proof),
            wasm_bindgen::Clamped(VK.to_vec()),
            wasm_bindgen::Clamped(CIRCUIT_PARAMS.to_vec()),
            wasm_bindgen::Clamped(KZG_PARAMS.to_vec()),
        );
        // should fail
        assert!(!value);
    }

    #[wasm_bindgen_test]
    async fn prove_pass() {
        // prove
        let proof = prove(
            wasm_bindgen::Clamped(WITNESS.to_vec()),
            wasm_bindgen::Clamped(PK.to_vec()),
            wasm_bindgen::Clamped(NETWORK.to_vec()),
            wasm_bindgen::Clamped(CIRCUIT_PARAMS.to_vec()),
            wasm_bindgen::Clamped(KZG_PARAMS.to_vec()),
        );
        assert!(proof.len() > 0);

        let value = verify(
            wasm_bindgen::Clamped(proof.to_vec()),
            wasm_bindgen::Clamped(VK.to_vec()),
            wasm_bindgen::Clamped(CIRCUIT_PARAMS.to_vec()),
            wasm_bindgen::Clamped(KZG_PARAMS.to_vec()),
        );
        // should not fail
        assert!(value);
    }
}
