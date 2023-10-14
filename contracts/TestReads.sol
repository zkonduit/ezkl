// SPDX-License-Identifier: UNLICENSED
pragma solidity ^0.8.17;

contract TestReads {
    int[] public arr;

    constructor(int256[] memory _numbers) {
        for (uint256 i = 0; i < _numbers.length; i++) {
            arr.push(_numbers[i]);
        }
    }
}
