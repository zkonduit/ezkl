// SPDX-License-Identifier: MIT
pragma solidity ^0.8.20;

import "forge-std/Test.sol";
import {console} from "forge-std/console.sol";
import "contracts/AttestData.sol" as AttestData;

contract MockVKA {
    constructor() {}
}

contract MockVerifier {
    bool public shouldVerify;

    constructor(bool _shouldVerify) {
        shouldVerify = _shouldVerify;
    }

    function verifyProof(
        bytes calldata,
        uint256[] calldata
    ) external view returns (bool) {
        require(shouldVerify, "Verification failed");
        return shouldVerify;
    }
}

contract MockVerifierSeperate {
    bool public shouldVerify;

    constructor(bool _shouldVerify) {
        shouldVerify = _shouldVerify;
    }

    function verifyProof(
        address,
        bytes calldata,
        uint256[] calldata
    ) external view returns (bool) {
        require(shouldVerify, "Verification failed");
        return shouldVerify;
    }
}

contract MockTargetContract {
    int256[] public data;

    constructor(int256[] memory _data) {
        data = _data;
    }

    function setData(int256[] memory _data) external {
        data = _data;
    }

    function getData() external view returns (int256[] memory) {
        return data;
    }
}

contract DataAttestationTest is Test {
    AttestData.DataAttestation das;
    MockVerifier verifier;
    MockVerifierSeperate verifierSeperate;
    MockVKA vka;
    MockTargetContract target;
    int256[] mockData = [int256(1e18), -int256(5e17)];
    uint256[] decimals = [18, 18];
    uint256[] bits = [13, 13];
    uint8 instanceOffset = 0;
    bytes callData;

    function setUp() public {
        target = new MockTargetContract(mockData);
        verifier = new MockVerifier(true);
        verifierSeperate = new MockVerifierSeperate(true);
        vka = new MockVKA();

        callData = abi.encodeWithSignature("getData()");

        das = new AttestData.DataAttestation(
            address(target),
            callData,
            decimals,
            bits,
            instanceOffset
        );
    }

    // Fork of mulDivRound which doesn't revert on overflow and returns a boolean instead to indicate overflow
    function mulDivRound(
        uint256 x,
        uint256 y,
        uint256 denominator
    ) public pure returns (uint256 result, bool overflow) {
        unchecked {
            uint256 prod0;
            uint256 prod1;
            assembly {
                let mm := mulmod(x, y, not(0))
                prod0 := mul(x, y)
                prod1 := sub(sub(mm, prod0), lt(mm, prod0))
            }
            uint256 remainder = mulmod(x, y, denominator);
            bool addOne;
            if (remainder * 2 >= denominator) {
                addOne = true;
            }

            if (prod1 == 0) {
                if (addOne) {
                    return ((prod0 / denominator) + 1, false);
                }
                return (prod0 / denominator, false);
            }

            if (denominator > prod1) {
                return (0, true);
            }

            assembly {
                prod1 := sub(prod1, gt(remainder, prod0))
                prod0 := sub(prod0, remainder)
            }

            uint256 twos = denominator & (~denominator + 1);
            assembly {
                denominator := div(denominator, twos)
                prod0 := div(prod0, twos)
                twos := add(div(sub(0, twos), twos), 1)
            }

            prod0 |= prod1 * twos;

            uint256 inverse = (3 * denominator) ^ 2;

            inverse *= 2 - denominator * inverse;
            inverse *= 2 - denominator * inverse;
            inverse *= 2 - denominator * inverse;
            inverse *= 2 - denominator * inverse;
            inverse *= 2 - denominator * inverse;
            inverse *= 2 - denominator * inverse;

            result = prod0 * inverse;
            if (addOne) {
                result += 1;
            }
            return (result, false);
        }
    }
    struct SampleAttestation {
        int256 mockData;
        uint8 decimals;
        uint8 bits;
    }
    function test_fuzzAttestedData(
        SampleAttestation[] memory _attestations
    ) public {
        vm.assume(_attestations.length == 1);
        int256[] memory _mockData = new int256[](1);
        uint256[] memory _decimals = new uint256[](1);
        uint256[] memory _bits = new uint256[](1);
        uint256[] memory _instances = new uint256[](1);
        for (uint256 i = 0; i < 1; i++) {
            SampleAttestation memory attestation = _attestations[i];
            _mockData[i] = attestation.mockData;
            vm.assume(attestation.mockData != type(int256).min); /// Will overflow int256 during negation op
            vm.assume(attestation.decimals < 77); /// Else will exceed uint256 bounds
            vm.assume(attestation.bits < 128); /// Else will exceed EZKL fixed point bounds for int128 type
            bool neg = attestation.mockData < 0;
            if (neg) {
                attestation.mockData = -attestation.mockData;
            }
            (uint256 _result, bool overflow) = mulDivRound(
                uint256(attestation.mockData),
                uint256(1 << attestation.bits),
                uint256(10 ** attestation.decimals)
            );
            vm.assume(!overflow);
            vm.assume(_result < das.HALF_ORDER());
            if (neg) {
                // No possibility of overflow here since output is less than or equal to HALF_ORDER
                // and therefore falls within the max range of int256 without overflow
                vm.assume(-int256(_result) > type(int128).min);
                _instances[i] =
                    uint256(int(das.ORDER()) - int256(_result)) %
                    das.ORDER();
            } else {
                vm.assume(_result < uint128(type(int128).max));
                _instances[i] = _result;
            }
            _decimals[i] = attestation.decimals;
            _bits[i] = attestation.bits;
        }
        // Update the attested data
        target.setData(_mockData);
        // Deploy the new data attestation contract
        AttestData.DataAttestation dasNew = new AttestData.DataAttestation(
            address(target),
            callData,
            _decimals,
            _bits,
            instanceOffset
        );
        bytes memory proof = hex"1234"; // Would normally contain commitments
        bytes memory encoded = abi.encodeWithSignature(
            "verifyProof(bytes,uint256[])",
            proof,
            _instances
        );

        AttestData.DataAttestation.Scalars memory _scalars = AttestData
            .DataAttestation
            .Scalars(10 ** _decimals[0], 1 << _bits[0]);

        int256 output = dasNew.quantizeData(_mockData[0], _scalars);
        console.log("output: ", output);
        uint256 fieldElement = dasNew.toFieldElement(output);
        // output should equal to _instances[0]
        assertEq(fieldElement, _instances[0]);

        bool verificationResult = dasNew.verifyWithDataAttestation(
            address(verifier),
            encoded
        );
        assertTrue(verificationResult);
    }

    // Test deployment parameters
    function testDeployment() public view {
        assertEq(das.contractAddress(), address(target));
        assertEq(das.callData(), abi.encodeWithSignature("getData()"));
        assertEq(das.instanceOffset(), instanceOffset);

        AttestData.DataAttestation.Scalars memory scalar = das.getScalars(0);
        assertEq(scalar.decimals, 1e18);
        assertEq(scalar.bits, 1 << 13);
    }

    // Test quantizeData function
    function testQuantizeData() public view {
        AttestData.DataAttestation.Scalars memory scalar = das.getScalars(0);

        int256 positive = das.quantizeData(1e18, scalar);
        assertEq(positive, int256(scalar.bits));

        int256 negative = das.quantizeData(-1e18, scalar);
        assertEq(negative, -int256(scalar.bits));

        // Test rounding
        int half = int(0.5e18 / scalar.bits);
        int256 rounded = das.quantizeData(half, scalar);
        assertEq(rounded, 1);
    }

    // Test staticCall functionality
    function testStaticCall() public view {
        bytes memory result = das.staticCall(
            address(target),
            abi.encodeWithSignature("getData()")
        );
        int256[] memory decoded = abi.decode(result, (int256[]));
        assertEq(decoded[0], mockData[0]);
        assertEq(decoded[1], mockData[1]);
    }

    // Test attestData validation
    function testAttestDataSuccess() public view {
        uint256[] memory instances = new uint256[](2);
        AttestData.DataAttestation.Scalars memory scalar = das.getScalars(0);
        instances[0] = das.toFieldElement(int(scalar.bits));
        instances[1] = das.toFieldElement(-int(scalar.bits >> 1));
        das.attestData(instances); // Should not revert
    }

    function testAttestDataFailure() public {
        uint256[] memory instances = new uint256[](2);
        instances[0] = das.toFieldElement(1e18); // Incorrect value
        instances[1] = das.toFieldElement(5e17);

        vm.expectRevert("Public input does not match");
        das.attestData(instances);
    }

    // Test full verification flow
    function testSuccessfulVerification() public view {
        // Prepare valid instances
        uint256[] memory instances = new uint256[](2);
        AttestData.DataAttestation.Scalars memory scalar = das.getScalars(0);
        instances[0] = das.toFieldElement(int(scalar.bits));
        instances[1] = das.toFieldElement(-int(scalar.bits >> 1));

        // Create valid calldata (mock)
        bytes memory proof = hex"1234"; // Would normally contain commitments
        bytes memory encoded = abi.encodeWithSignature(
            "verifyProof(bytes,uint256[])",
            proof,
            instances
        );
        bytes memory encoded_vka = abi.encodeWithSignature(
            "verifyProof(address,bytes,uint256[])",
            address(vka),
            proof,
            instances
        );

        bool result = das.verifyWithDataAttestation(address(verifier), encoded);
        assertTrue(result);
        result = das.verifyWithDataAttestation(
            address(verifierSeperate),
            encoded_vka
        );
        assertTrue(result);
    }

    function testLoadInstances() public view {
        uint256[] memory instances = new uint256[](2);
        AttestData.DataAttestation.Scalars memory scalar = das.getScalars(0);
        instances[0] = das.toFieldElement(int(scalar.bits));
        instances[1] = das.toFieldElement(-int(scalar.bits >> 1));

        // Create valid calldata (mock)
        bytes memory proof = hex"1234"; // Would normally contain commitments
        bytes memory encoded = abi.encodeWithSignature(
            "verifyProof(bytes,uint256[])",
            proof,
            instances
        );
        bytes memory encoded_vka = abi.encodeWithSignature(
            "verifyProof(address,bytes,uint256[])",
            address(vka),
            proof,
            instances
        );

        // Load encoded instances from calldata
        uint256[] memory extracted_instances_calldata = das
            .getInstancesCalldata(encoded);
        assertEq(extracted_instances_calldata[0], instances[0]);
        assertEq(extracted_instances_calldata[1], instances[1]);
        // Load encoded instances from memory
        uint256[] memory extracted_instances_memory = das.getInstancesMemory(
            encoded
        );
        assertEq(extracted_instances_memory[0], instances[0]);
        assertEq(extracted_instances_memory[1], instances[1]);
        // Load encoded with vk instances from calldata
        uint256[] memory extracted_instances_calldata_vk = das
            .getInstancesCalldata(encoded_vka);
        assertEq(extracted_instances_calldata_vk[0], instances[0]);
        assertEq(extracted_instances_calldata_vk[1], instances[1]);
        // Load encoded with vk instances from memory
        uint256[] memory extracted_instances_memory_vk = das.getInstancesMemory(
            encoded_vka
        );
        assertEq(extracted_instances_memory_vk[0], instances[0]);
        assertEq(extracted_instances_memory_vk[1], instances[1]);
    }

    function testInvalidCommitments() public {
        // Create calldata with invalid commitments
        bytes memory invalidProof = hex"5678";
        uint256[] memory instances = new uint256[](2);
        AttestData.DataAttestation.Scalars memory scalar = das.getScalars(0);
        instances[0] = das.toFieldElement(int(scalar.bits));
        instances[1] = das.toFieldElement(-int(scalar.bits >> 1));
        bytes memory encoded = abi.encodeWithSignature(
            "verifyProof(bytes,uint256[])",
            invalidProof,
            instances
        );

        vm.expectRevert("Invalid KZG commitments");
        das.verifyWithDataAttestation(address(verifier), encoded);
    }

    function testInvalidVerifier() public {
        MockVerifier invalidVerifier = new MockVerifier(false);
        uint256[] memory instances = new uint256[](2);
        AttestData.DataAttestation.Scalars memory scalar = das.getScalars(0);
        instances[0] = das.toFieldElement(int(scalar.bits));
        instances[1] = das.toFieldElement(-int(scalar.bits >> 1));
        bytes memory encoded = abi.encodeWithSignature(
            "verifyProof(bytes,uint256[])",
            hex"1234",
            instances
        );

        vm.expectRevert("low-level call to verifier failed");
        das.verifyWithDataAttestation(address(invalidVerifier), encoded);
    }

    // Test edge cases
    function testZeroValueQuantization() public view {
        AttestData.DataAttestation.Scalars memory scalar = das.getScalars(0);
        int256 zero = das.quantizeData(0, scalar);
        assertEq(zero, 0);
    }

    function testOverflowProtection() public {
        int256 order = int(
            uint256(
                0x30644e72e131a029b85045b68181585d2833e84879b9709143e1f593f0000001
            )
        );
        // int256 half_order = int(order >> 1);
        AttestData.DataAttestation.Scalars memory scalar = AttestData
            .DataAttestation
            .Scalars(1, 1 << 2);

        vm.expectRevert("Overflow field modulus");
        das.quantizeData(order, scalar); // Value that would overflow
    }

    function testInvalidFunctionSignature() public {
        uint256[] memory instances = new uint256[](2);
        AttestData.DataAttestation.Scalars memory scalar = das.getScalars(0);
        instances[0] = das.toFieldElement(int(scalar.bits));
        instances[1] = das.toFieldElement(-int(scalar.bits >> 1));
        bytes memory encoded_invalid_sig = abi.encodeWithSignature(
            "verifyProofff(bytes,uint256[])",
            hex"1234",
            instances
        );

        vm.expectRevert("Invalid function signature");
        das.verifyWithDataAttestation(address(verifier), encoded_invalid_sig);
    }
}
