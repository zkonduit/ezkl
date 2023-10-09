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
    use ezkl::wasm::{
        bufferToVecOfVecU64, elgamalDecrypt, elgamalEncrypt, elgamalGenRandom, genWitness,
        poseidonHash, u8_array_to_u128_le, vecU64ToFelt, vecU64ToFloat, vecU64ToInt, encodeVerifierCalldata
    };
    use halo2curves::bn256::{Fr, G1Affine};
    use halo2_solidity_verifier::encode_calldata;
    use rand::rngs::StdRng;
    use rand::SeedableRng;
    use ezkl::pfsys;
    use snark_verifier::util::arithmetic::PrimeField;
    #[cfg(feature = "web")]
    pub use wasm_bindgen_rayon::init_thread_pool;
    use wasm_bindgen_test::*;

    wasm_bindgen_test::wasm_bindgen_test_configure!(run_in_browser);

    pub const WITNESS: &[u8] = include_bytes!("../tests/wasm/test.witness.json");
    pub const NETWORK: &[u8] = include_bytes!("../tests/wasm/test_network.compiled");
    pub const INPUT: &[u8] = include_bytes!("../tests/wasm/input.json");
    pub const PROOF: &[u8] = include_bytes!("../tests/wasm/test.proof");

    #[wasm_bindgen_test]
    async fn verify_encode_verifier_calldata() {

        let ser_proof = wasm_bindgen::Clamped(PROOF.to_vec());

        // with no vk address
        let calldata = encodeVerifierCalldata(ser_proof.clone(), None)
            .map_err(|_| "failed")
            .unwrap();

        let snark: pfsys::Snark<Fr, G1Affine> = serde_json::from_slice(&PROOF).unwrap();
        let flattened_instances = snark.instances.into_iter().flatten();
        let reference_calldata = 
            encode_calldata(
                None,
                &snark.proof,
                &flattened_instances.clone().collect::<Vec<_>>()
            );
        assert_eq!(calldata, reference_calldata);
        // with vk address
        let vk_address = hex::decode("0000000000000000000000000000000000000000").unwrap();

        let vk_address: [u8; 20] = {
            let mut array = [0u8; 20];
            array.copy_from_slice(&vk_address);
            array
        };

        let serialized = serde_json::to_vec(&vk_address).unwrap();
    
        let calldata = encodeVerifierCalldata(ser_proof, Some(serialized))
            .map_err(|_| "failed")
            .unwrap();
        let reference_calldata = 
            encode_calldata(
                Some(vk_address),
                &snark.proof,
                &flattened_instances.collect::<Vec<_>>()
            );
        assert_eq!(calldata, reference_calldata);

    }

    #[wasm_bindgen_test]
    async fn verify_field_serialization_roundtrip() {
        for i in 0..32 {
            let field_element = Fr::from(i);
            let serialized = serde_json::to_vec(&field_element).unwrap();
            let clamped = wasm_bindgen::Clamped(serialized);
            let scale = 2;
            let floating_point = vecU64ToFloat(clamped.clone(), scale)
                .map_err(|_| "failed")
                .unwrap();
            assert_eq!(floating_point, (i as f64) / 4.0);

            let integer: i128 = serde_json::from_slice(
                &vecU64ToInt(clamped.clone()).map_err(|_| "failed").unwrap(),
            )
            .unwrap();
            assert_eq!(integer, i as i128);

            let hex_string = format!("{:?}", field_element);
            let returned_string = vecU64ToFelt(clamped).map_err(|_| "failed").unwrap();
            assert_eq!(hex_string, returned_string);
        }
    }

    #[wasm_bindgen_test]
    async fn verify_buffer_to_field_elements() {
        let string_high = String::from("high");
        let mut buffer = string_high.clone().into_bytes();
        let clamped = wasm_bindgen::Clamped(buffer.clone());

        let field_elements_ser = bufferToVecOfVecU64(clamped).map_err(|_| "failed").unwrap();

        let field_elements: Vec<Fr> = serde_json::from_slice(&field_elements_ser[..]).unwrap();

        buffer.resize(16, 0);

        let reference_int = u8_array_to_u128_le(buffer.try_into().unwrap());

        let reference_field_element_high = PrimeField::from_u128(reference_int);

        assert_eq!(field_elements[0], reference_field_element_high);

        // length 16 string (divisible by 16 so doesn't need padding)
        let string_sample = String::from("a sample string!");
        let buffer = string_sample.clone().into_bytes();
        let clamped = wasm_bindgen::Clamped(buffer.clone());

        let field_elements_ser = bufferToVecOfVecU64(clamped).map_err(|_| "failed").unwrap();

        let field_elements: Vec<Fr> = serde_json::from_slice(&field_elements_ser[..]).unwrap();

        let reference_int = u8_array_to_u128_le(buffer.try_into().unwrap());

        let reference_field_element_sample = PrimeField::from_u128(reference_int);

        assert_eq!(field_elements[0], reference_field_element_sample);

        let string_concat = string_sample + &string_high;

        let buffer = string_concat.into_bytes();
        let clamped = wasm_bindgen::Clamped(buffer.clone());

        let field_elements_ser = bufferToVecOfVecU64(clamped).map_err(|_| "failed").unwrap();

        let field_elements: Vec<Fr> = serde_json::from_slice(&field_elements_ser[..]).unwrap();

        assert_eq!(field_elements[0], reference_field_element_sample);
        assert_eq!(field_elements[1], reference_field_element_high);
    }

    #[wasm_bindgen_test]
    async fn verify_elgamal_gen_random_wasm() {
        // Generate a seed value
        let seed = [0u8; 32];

        // Convert the seed to a wasm-friendly format
        let wasm_seed = wasm_bindgen::Clamped(seed.to_vec());

        // Use the seed to generate ElGamal variables via WASM function
        let wasm_output = elgamalGenRandom(wasm_seed).map_err(|_| "failed").unwrap();

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
        )
        .map_err(|_| "failed")
        .unwrap();

        let sk = serde_json::to_vec(&var.sk).unwrap();

        let decrypted_message =
            elgamalDecrypt(wasm_bindgen::Clamped(cipher), wasm_bindgen::Clamped(sk))
                .map_err(|_| "failed")
                .unwrap();

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

        let hash = poseidonHash(wasm_bindgen::Clamped(message_ser))
            .map_err(|_| "failed")
            .unwrap();
        let hash: Vec<Vec<Fr>> = serde_json::from_slice(&hash[..]).unwrap();

        let reference_hash =
            PoseidonChip::<PoseidonSpec, POSEIDON_WIDTH, POSEIDON_RATE, POSEIDON_LEN_GRAPH>::run(
                message.clone(),
            )
            .map_err(|_| "failed")
            .unwrap();

        assert_eq!(hash, reference_hash)
    }

    #[wasm_bindgen_test]
    async fn verify_gen_witness() {
        let witness = genWitness(
            wasm_bindgen::Clamped(NETWORK.to_vec()),
            wasm_bindgen::Clamped(INPUT.to_vec()),
        )
        .map_err(|_| "failed")
        .unwrap();

        let witness: GraphWitness = serde_json::from_slice(&witness[..]).unwrap();

        let reference_witness: GraphWitness = serde_json::from_slice(&WITNESS).unwrap();
        // should not fail
        assert_eq!(witness, reference_witness);
    }
}
