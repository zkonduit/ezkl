// SPDX-License-Identifier: MIT
pragma solidity ^0.8.17;

contract Verifier {

    /**
     * @notice EZKL P value
     * @dev In order to prevent the verifier from accepting two version of the same pubInput, n and the quantity (n + P),  where n + P <= 2^256, we require that all instances are stricly less than P.
     * @dev The reason for this is that the assmebly code of the verifier performs all arithmetic operations modulo P and as a consequence can't distinguish between n and n + P values.
     */

    uint256 constant SIZE_LIMIT = uint256(0x30644e72e131a029b85045b68181585d2833e84879b9709143e1f593f0000001); 

    function verify(
        uint256[] calldata instances,
        bytes calldata proof
    ) public view returns (bool) {
        bool success = true;
        bytes32[] memory transcript;
        for (uint i = 0; i < instances.length; i++) {
            require(instances[i] < SIZE_LIMIT);
        }
        assembly { /* This is where the proof verification happens*/ }
        return success;
    }
}
