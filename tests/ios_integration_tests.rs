#[cfg(feature = "ios-bindings")]
uniffi::build_foreign_language_testcases!(
    "tests/ios/can_verify_aggr.swift",
    "tests/ios/verify_gen_witness.swift",
    "tests/ios/gen_pk_test.swift",
    "tests/ios/gen_vk_test.swift",
    "tests/ios/pk_is_valid_test.swift",
    "tests/ios/verify_validations.swift",
    // "tests/ios/verify_encode_verifier_calldata.swift", // TODO - the function requires rust dependencies to test
    // "tests/ios/verify_kzg_commit.swift", // TODO - the function is not exported and requires rust dependencies to test
);