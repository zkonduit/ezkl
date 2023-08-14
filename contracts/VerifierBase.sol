// SPDX-License-Identifier: MIT
pragma solidity ^0.8.17;

contract Verifier {

    /**
     * @notice EZKL P value
     * @dev In order to prevent the verifier from accepting two version of the same pubInput, n and the quantity (n + P),  where n + P <= 2^256, we require that all instances are stricly less than P.
     * @dev The reason for this is that the assmebly code of the verifier performs all arithmetic operations modulo P and as a consequence can't distinguish between n and n + P values.
     */

    uint256 constant SIZE_LIMIT = 21888242871839275222246405745257275088696311157297823662689037894645226208583; 

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
