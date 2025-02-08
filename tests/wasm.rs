#[cfg(all(target_arch = "wasm32", target_os = "unknown"))]
#[cfg(test)]
mod wasm32 {
    use ezkl::bindings::wasm::{
        bufferToVecOfFelt, compiledCircuitValidation, encodeVerifierCalldata, feltToBigEndian,
        feltToFloat, feltToInt, feltToLittleEndian, genPk, genVk, genWitness, inputValidation,
        kzgCommit, pkValidation, poseidonHash, proofValidation, prove, settingsValidation,
        srsValidation, u8_array_to_u128_le, verify, verifyAggr, vkValidation, witnessValidation,
    };
    use ezkl::circuit::modules::polycommit::PolyCommitChip;
    use ezkl::circuit::modules::poseidon::spec::{PoseidonSpec, POSEIDON_RATE, POSEIDON_WIDTH};
    use ezkl::circuit::modules::poseidon::PoseidonChip;
    use ezkl::circuit::modules::Module;
    use ezkl::graph::GraphCircuit;
    use ezkl::graph::{GraphSettings, GraphWitness};
    use ezkl::pfsys;
    use halo2_proofs::plonk::VerifyingKey;
    use halo2_proofs::poly::kzg::commitment::KZGCommitmentScheme;
    use halo2_proofs::poly::kzg::commitment::ParamsKZG;
    use halo2_solidity_verifier::encode_calldata;
    use halo2curves::bn256::Bn256;
    use halo2curves::bn256::{Fr, G1Affine};
    use snark_verifier::util::arithmetic::PrimeField;
    #[cfg(feature = "web")]
    pub use wasm_bindgen_rayon::init_thread_pool;
    use wasm_bindgen_test::*;

    wasm_bindgen_test::wasm_bindgen_test_configure!(run_in_browser);

    pub const WITNESS: &[u8] = include_bytes!("assets/witness.json");
    pub const NETWORK_COMPILED: &[u8] = include_bytes!("assets/model.compiled");
    pub const NETWORK: &[u8] = include_bytes!("assets/network.onnx");
    pub const INPUT: &[u8] = include_bytes!("assets/input.json");
    pub const PROOF: &[u8] = include_bytes!("assets/proof.json");
    pub const PROOF_AGGR: &[u8] = include_bytes!("assets/proof_aggr.json");
    pub const SETTINGS: &[u8] = include_bytes!("assets/settings.json");
    pub const PK: &[u8] = include_bytes!("assets/pk.key");
    pub const VK: &[u8] = include_bytes!("assets/vk.key");
    pub const VK_AGGR: &[u8] = include_bytes!("assets/vk_aggr.key");
    pub const SRS: &[u8] = include_bytes!("assets/kzg");
    pub const SRS1: &[u8] = include_bytes!("assets/kzg1.srs");

    #[wasm_bindgen_test]
    async fn can_verify_aggr() {
        let value = verifyAggr(
            wasm_bindgen::Clamped(PROOF_AGGR.to_vec()),
            wasm_bindgen::Clamped(VK_AGGR.to_vec()),
            21,
            wasm_bindgen::Clamped(SRS1.to_vec()),
            "kzg",
        )
        .map_err(|_| "failed")
        .unwrap();

        // should not fail
        assert!(value);
    }

    #[wasm_bindgen_test]
    async fn verify_encode_verifier_calldata() {
        let ser_proof = wasm_bindgen::Clamped(PROOF.to_vec());

        // with no vk address
        let calldata = encodeVerifierCalldata(ser_proof.clone(), None)
            .map_err(|_| "failed")
            .unwrap();

        let snark: pfsys::Snark<Fr, G1Affine> = serde_json::from_slice(&PROOF).unwrap();
        let flattened_instances = snark.instances.into_iter().flatten();
        let reference_calldata = encode_calldata(
            None,
            &snark.proof,
            &flattened_instances.clone().collect::<Vec<_>>(),
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
        let reference_calldata = encode_calldata(
            Some(vk_address),
            &snark.proof,
            &flattened_instances.collect::<Vec<_>>(),
        );
        assert_eq!(calldata, reference_calldata);
    }

    #[wasm_bindgen_test]
    fn verify_kzg_commit() {
        // create a vector of field elements Vec<Fr> and assign it to the message variable
        let mut message: Vec<Fr> = vec![];
        for i in 0..32 {
            message.push(Fr::from(i as u64));
        }
        let message_ser = serde_json::to_vec(&message).unwrap();

        let settings: GraphSettings = serde_json::from_slice(&SETTINGS).unwrap();
        let mut reader = std::io::BufReader::new(SRS);
        let params: ParamsKZG<Bn256> =
            halo2_proofs::poly::commitment::Params::<'_, G1Affine>::read(&mut reader).unwrap();
        let mut reader = std::io::BufReader::new(VK);
        let vk = VerifyingKey::<G1Affine>::read::<_, GraphCircuit>(
            &mut reader,
            halo2_proofs::SerdeFormat::RawBytes,
            settings.clone(),
        )
        .unwrap();
        let commitment_ser = kzgCommit(
            wasm_bindgen::Clamped(message_ser),
            wasm_bindgen::Clamped(VK.to_vec()),
            wasm_bindgen::Clamped(SETTINGS.to_vec()),
            wasm_bindgen::Clamped(SRS.to_vec()),
        )
        .map_err(|_| "failed")
        .unwrap();
        let commitment: Vec<halo2curves::bn256::G1Affine> =
            serde_json::from_slice(&commitment_ser[..]).unwrap();
        let reference_commitment = PolyCommitChip::commit::<KZGCommitmentScheme<Bn256>>(
            message,
            (vk.cs().blinding_factors() + 1) as u32,
            &params,
        );

        assert_eq!(commitment, reference_commitment);
    }

    #[wasm_bindgen_test]
    async fn verify_field_serialization_roundtrip() {
        for i in 0..32 {
            let field_element = Fr::from(i);
            let serialized = serde_json::to_vec(&field_element).unwrap();

            let clamped = wasm_bindgen::Clamped(serialized);
            let scale = 2;
            let floating_point = feltToFloat(clamped.clone(), scale)
                .map_err(|_| "failed")
                .unwrap();
            assert_eq!(floating_point, (i as f64) / 4.0);

            let integer: i64 =
                serde_json::from_slice(&feltToInt(clamped.clone()).map_err(|_| "failed").unwrap())
                    .unwrap();
            assert_eq!(integer, i as i64);

            let hex_string = format!("{:?}", field_element.clone());
            let returned_string: String = feltToBigEndian(clamped.clone())
                .map_err(|_| "failed")
                .unwrap();
            assert_eq!(hex_string, returned_string);
            let repr = serde_json::to_string(&field_element).unwrap();
            let little_endian_string: String = serde_json::from_str(&repr).unwrap();
            let returned_string: String =
                feltToLittleEndian(clamped).map_err(|_| "failed").unwrap();
            assert_eq!(little_endian_string, returned_string);
        }
    }

    #[wasm_bindgen_test]
    async fn verify_buffer_to_field_elements() {
        let string_high = String::from("high");
        let mut buffer = string_high.clone().into_bytes();
        let clamped = wasm_bindgen::Clamped(buffer.clone());

        let field_elements_ser = bufferToVecOfFelt(clamped).map_err(|_| "failed").unwrap();

        let field_elements: Vec<Fr> = serde_json::from_slice(&field_elements_ser[..]).unwrap();

        buffer.resize(16, 0);

        let reference_int = u8_array_to_u128_le(buffer.try_into().unwrap());

        let reference_field_element_high = PrimeField::from_u128(reference_int);

        assert_eq!(field_elements[0], reference_field_element_high);

        // length 16 string (divisible by 16 so doesn't need padding)
        let string_sample = String::from("a sample string!");
        let buffer = string_sample.clone().into_bytes();
        let clamped = wasm_bindgen::Clamped(buffer.clone());

        let field_elements_ser = bufferToVecOfFelt(clamped).map_err(|_| "failed").unwrap();

        let field_elements: Vec<Fr> = serde_json::from_slice(&field_elements_ser[..]).unwrap();

        let reference_int = u8_array_to_u128_le(buffer.try_into().unwrap());

        let reference_field_element_sample = PrimeField::from_u128(reference_int);

        assert_eq!(field_elements[0], reference_field_element_sample);

        let string_concat = string_sample + &string_high;

        let buffer = string_concat.into_bytes();
        let clamped = wasm_bindgen::Clamped(buffer.clone());

        let field_elements_ser = bufferToVecOfFelt(clamped).map_err(|_| "failed").unwrap();

        let field_elements: Vec<Fr> = serde_json::from_slice(&field_elements_ser[..]).unwrap();

        assert_eq!(field_elements[0], reference_field_element_sample);
        assert_eq!(field_elements[1], reference_field_element_high);
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
            PoseidonChip::<PoseidonSpec, POSEIDON_WIDTH, POSEIDON_RATE>::run(message.clone())
                .map_err(|_| "failed")
                .unwrap();

        assert_eq!(hash, reference_hash)
    }

    #[wasm_bindgen_test]
    async fn verify_gen_witness() {
        let witness = genWitness(
            wasm_bindgen::Clamped(NETWORK_COMPILED.to_vec()),
            wasm_bindgen::Clamped(INPUT.to_vec()),
        )
        .map_err(|_| "failed")
        .unwrap();

        let witness: GraphWitness = serde_json::from_slice(&witness[..]).unwrap();

        let reference_witness: GraphWitness = serde_json::from_slice(&WITNESS).unwrap();
        // should not fail
        assert_eq!(witness, reference_witness);
    }

    #[wasm_bindgen_test]
    async fn gen_pk_test() {
        let vk = genVk(
            wasm_bindgen::Clamped(NETWORK_COMPILED.to_vec()),
            wasm_bindgen::Clamped(SRS.to_vec()),
            true,
        )
        .map_err(|_| "failed")
        .unwrap();

        let pk = genPk(
            wasm_bindgen::Clamped(vk),
            wasm_bindgen::Clamped(NETWORK_COMPILED.to_vec()),
            wasm_bindgen::Clamped(SRS.to_vec()),
        )
        .map_err(|_| "failed")
        .unwrap();

        assert!(pk.len() > 0);
    }

    #[wasm_bindgen_test]
    async fn gen_vk_test() {
        let vk = genVk(
            wasm_bindgen::Clamped(NETWORK_COMPILED.to_vec()),
            wasm_bindgen::Clamped(SRS.to_vec()),
            true,
        )
        .map_err(|_| "failed")
        .unwrap();

        assert!(vk.len() > 0);
    }

    #[wasm_bindgen_test]
    async fn pk_is_valid_test() {
        let vk = genVk(
            wasm_bindgen::Clamped(NETWORK_COMPILED.to_vec()),
            wasm_bindgen::Clamped(SRS.to_vec()),
            true,
        )
        .map_err(|_| "failed")
        .unwrap();

        let pk = genPk(
            wasm_bindgen::Clamped(vk.clone()),
            wasm_bindgen::Clamped(NETWORK_COMPILED.to_vec()),
            wasm_bindgen::Clamped(SRS.to_vec()),
        )
        .map_err(|_| "failed")
        .unwrap();

        // prove
        let proof = prove(
            wasm_bindgen::Clamped(WITNESS.to_vec()),
            wasm_bindgen::Clamped(pk.clone()),
            wasm_bindgen::Clamped(NETWORK_COMPILED.to_vec()),
            wasm_bindgen::Clamped(SRS.to_vec()),
        )
        .map_err(|_| "failed")
        .unwrap();

        assert!(proof.len() > 0);

        let value = verify(
            wasm_bindgen::Clamped(proof.to_vec()),
            wasm_bindgen::Clamped(vk),
            wasm_bindgen::Clamped(SETTINGS.to_vec()),
            wasm_bindgen::Clamped(SRS.to_vec()),
        )
        .map_err(|_| "failed")
        .unwrap();

        // should not fail
        assert!(value);
    }

    #[wasm_bindgen_test]
    async fn verify_validations() {
        // Run witness validation on network (should fail)
        let witness = witnessValidation(wasm_bindgen::Clamped(NETWORK_COMPILED.to_vec()));
        assert!(witness.is_err());
        // Run witness validation on witness (should pass)
        let witness = witnessValidation(wasm_bindgen::Clamped(WITNESS.to_vec()));
        assert!(witness.is_ok());
        // Run compiled circuit validation on onnx network (should fail)
        let circuit = compiledCircuitValidation(wasm_bindgen::Clamped(NETWORK.to_vec()));
        assert!(circuit.is_err());
        // Run compiled circuit validation on compiled network (should pass)
        let circuit = compiledCircuitValidation(wasm_bindgen::Clamped(NETWORK_COMPILED.to_vec()));
        assert!(circuit.is_ok());
        // Run input validation on witness (should fail)
        let input = inputValidation(wasm_bindgen::Clamped(WITNESS.to_vec()));
        assert!(input.is_err());
        // Run input validation on input (should pass)
        let input = inputValidation(wasm_bindgen::Clamped(INPUT.to_vec()));
        assert!(input.is_ok());
        // Run proof validation on witness (should fail)
        let proof = proofValidation(wasm_bindgen::Clamped(WITNESS.to_vec()));
        assert!(proof.is_err());
        // Run proof validation on proof (should pass)
        let proof = proofValidation(wasm_bindgen::Clamped(PROOF.to_vec()));
        assert!(proof.is_ok());
        // // Run vk validation on SRS (should fail)
        // let vk = vkValidation(
        //     wasm_bindgen::Clamped(SRS.to_vec()),
        //     wasm_bindgen::Clamped(SETTINGS.to_vec())
        // );
        // assert!(vk.is_err());

        // Run vk validation on vk (should pass)
        let vk = vkValidation(
            wasm_bindgen::Clamped(VK.to_vec()),
            wasm_bindgen::Clamped(SETTINGS.to_vec()),
        );
        assert!(vk.is_ok());
        // // Run pk validation on vk (should fail)
        // let pk = pkValidation(
        //     wasm_bindgen::Clamped(VK.to_vec()),
        //     wasm_bindgen::Clamped(SETTINGS.to_vec())
        // );
        // assert!(pk.is_err());
        // Run pk validation on pk (should pass)
        let pk = pkValidation(
            wasm_bindgen::Clamped(PK.to_vec()),
            wasm_bindgen::Clamped(SETTINGS.to_vec()),
        );

        assert!(pk.is_ok());
        // Run settings validation on proof (should fail)
        let settings = settingsValidation(wasm_bindgen::Clamped(PROOF.to_vec()));
        assert!(settings.is_err());
        // Run settings validation on settings (should pass)
        let settings = settingsValidation(wasm_bindgen::Clamped(SETTINGS.to_vec()));
        assert!(settings.is_ok());
        // // Run srs validation on vk (should fail)
        // let srs = srsValidation(
        //     wasm_bindgen::Clamped(VK.to_vec())
        // );
        // assert!(srs.is_err());
        // Run srs validation on srs (should pass)
        let srs = srsValidation(wasm_bindgen::Clamped(SRS.to_vec()));
        assert!(srs.is_ok());
    }
}
