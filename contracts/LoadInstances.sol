// SPDX-License-Identifier: MIT
pragma solidity ^0.8.20;
contract LoadInstances {
    /**
     * @dev Parse the instances array from the Hal2Verifier encoded calldata. 
     * @notice must pass encoded bytes from memory
     * @param encoded - The data returned from the account calls.
     */
    function getInstancesMemory(
        bytes memory encoded
    ) internal pure returns (uint256[] memory instances) {
        bytes4 funcSig;
        uint256 instances_offset;
        uint256 instances_length;
        assembly {
            // fetch function sig. Either `verifyProof(bytes,uint256[])` or `verifyProof(address,bytes,uint256[])`
            funcSig := mload(add(encoded, 0x20))

            // Fetch instances offset which is 4 + 32 + 32 bytes away from
            // start of encoded for `verifyProof(bytes,uint256[])`,
            // and 4 + 32 + 32 +32 away for `verifyProof(address,bytes,uint256[])`

            instances_offset := mload(
                add(encoded, add(0x44, mul(0x20, eq(funcSig, 0xaf83a18d))))
            )

            instances_length := mload(add(add(encoded, 0x24), instances_offset))
        }
        instances = new uint256[](instances_length); // Allocate memory for the instances array.
        assembly {
            // Now instances points to the start of the array data
            // (right after the length field).
            for {
                let i := 0x20
            } lt(i, mul(instances_length, 0x20)) {
                i := add(i, 0x20)
            } {
                mstore(
                    add(instances, i),
                    mload(add(add(encoded, add(i, 0x24)), instances_offset))
                )
            }
        }
    }
    /**
     * @dev Parse the instances array from the Hal2Verifier encoded calldata. 
     * @notice must pass encoded bytes from calldata
     * @param encoded - The data returned from the account calls.
     */
    function getInstancesCalldata(
        bytes calldata encoded
    ) internal pure returns (uint256[] memory instances) {
        bytes4 funcSig;
        uint256 instances_offset;
        uint256 instances_length;
        assembly {
            // fetch function sig. Either `verifyProof(bytes,uint256[])` or `verifyProof(address,bytes,uint256[])`
            funcSig := calldataload(encoded.offset)

            // Fetch instances offset which is 4 + 32 + 32 bytes away from
            // start of encoded for `verifyProof(bytes,uint256[])`,
            // and 4 + 32 + 32 +32 away for `verifyProof(address,bytes,uint256[])`

            instances_offset := calldataload(
                add(
                    encoded.offset,
                    add(0x24, mul(0x20, eq(funcSig, 0xaf83a18d)))
                )
            )

            instances_length := calldataload(add(add(encoded.offset, 0x04), instances_offset))
        }
        instances = new uint256[](instances_length); // Allocate memory for the instances array.
        assembly{
            // Now instances points to the start of the array data
            // (right after the length field).

            for {
                let i := 0x20
            } lt(i, mul(instances_length, 0x20)) {
                i := add(i, 0x20)
            } {
                mstore(
                    add(instances, i),
                    calldataload(
                        add(add(encoded.offset, add(i, 0x04)), instances_offset)
                    )
                )
            }
        }
    }
}