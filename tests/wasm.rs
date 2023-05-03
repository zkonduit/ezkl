#[cfg(all(target_arch = "wasm32", target_os = "unknown"))]
mod wasm32 {
    use ezkl_lib::pfsys::Snarkbytes;
    use ezkl_lib::wasm::verify_wasm;

    use wasm_bindgen_test::*;

    pub const KZG_PARAMS: &[u8] = include_bytes!("../tests/wasm/kzg.params");
    pub const CIRCUIT_PARAMS: &[u8] = include_bytes!("../tests/wasm/circuit.params");
    pub const VK: &[u8] = include_bytes!("../tests/wasm/local.vk");
    pub const PROOF: &[u8] = include_bytes!("../tests/wasm/local.pf");

    #[wasm_bindgen_test]
    fn pass() {
        let proof_js = wasm_bindgen::JsValue::from_serde(&PROOF).unwrap();
        let vk = wasm_bindgen::JsValue::from_serde(&VK).unwrap();
        let circuit_params_ser = wasm_bindgen::JsValue::from_serde(&CIRCUIT_PARAMS).unwrap();
        let params_ser = wasm_bindgen::JsValue::from_serde(&KZG_PARAMS).unwrap();
        let value = verify_wasm(proof_js, vk, circuit_params_ser, params_ser);
        assert!(value);
    }

    #[wasm_bindgen_test]
    fn fail() {
        let proof = Snarkbytes {
            proof: vec![0; 32],
            num_instance: vec![1],
            instances: vec![vec![vec![0_u8; 32]]],
        };
        let proof = serde_json::to_vec(&proof).unwrap();
        let proof_js = wasm_bindgen::JsValue::from_serde(&proof).unwrap();
        let vk = wasm_bindgen::JsValue::from_serde(&VK).unwrap();
        let circuit_params_ser = wasm_bindgen::JsValue::from_serde(&CIRCUIT_PARAMS).unwrap();
        let params_ser = wasm_bindgen::JsValue::from_serde(&KZG_PARAMS).unwrap();
        let value = verify_wasm(proof_js, vk, circuit_params_ser, params_ser);
        // should fail
        assert!(!value);
    }
}
