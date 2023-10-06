// SPDX-License-Identifier: MIT
pragma solidity ^0.8.20;

// This contract serves as a Data Attestation Verifier for the EZKL model.
// It is designed to read and attest to instances of proofs generated from a specified circuit.
// It is particularly constructed to read only int256 data from specified on-chain contracts' view functions.

// Overview of the contract functionality:
// 1. Initialization: Through the constructor, it sets up the contract calls that the EZKL model will read from.
// 2. Data Quantization: Quantizes the returned data into a scaled fixed-point representation. See the `quantizeData` method for details.
// 3. Static Calls: Makes static calls to fetch data from other contracts. See the `staticCall` method.
// 4. Field Element Conversion: The fixed-point representation is then converted into a field element modulo P using the `toFieldElement` method.
// 5. Data Attestation: The `attestData` method validates that the public instances match the data fetched and processed by the contract.
// 6. Proof Verification: The `verifyWithDataAttestation` method parses the instances out of the encoded calldata and calls the `attestData` method to validate the public instances,
//  then calls the `verifyProof` method to verify the proof on the verifier.

contract DataAttestation {
    /**
     * @notice Struct used to make view only calls to accounts to fetch the data that EZKL reads from.
     * @param the address of the account to make calls to
     * @param the abi encoded function calls to make to the `contractAddress`
     */
    struct AccountCall {
        address contractAddress;
        mapping(uint256 => bytes) callData;
        mapping(uint256 => uint256) decimals;
        uint callCount;
    }
    AccountCall[] public accountCalls;

    uint[] public scales;

    address public admin;

    /**
     * @notice EZKL P value 
     * @dev In order to prevent the verifier from accepting two version of the same pubInput, n and the quantity (n + P),  where n + P <= 2^256, we require that all instances are stricly less than P. a
     * @dev The reason for this is that the assmebly code of the verifier performs all arithmetic operations modulo P and as a consequence can't distinguish between n and n + P.
     */
    uint256 constant ORDER = uint256(0x30644e72e131a029b85045b68181585d2833e84879b9709143e1f593f0000001); 

    uint256 constant INPUT_CALLS = 0;

    uint256 constant OUTPUT_CALLS = 0;

    uint8 public instanceOffset;

    /**
     * @dev Initialize the contract with account calls the EZKL model will read from.
     * @param _contractAddresses - The calls to all the contracts EZKL reads storage from.
     * @param _callData - The abi encoded function calls to make to the `contractAddress` that EZKL reads storage from.
     */
    constructor(
        address[] memory _contractAddresses,
        bytes[][] memory _callData,
        uint256[][] memory _decimals,
        uint[] memory _scales,
        uint8 _instanceOffset,
        address _admin
    ) {
        admin = _admin;
        for (uint i; i < _scales.length; i++) {
            scales.push(1 << _scales[i]);
        }
        populateAccountCalls(_contractAddresses, _callData, _decimals);
        instanceOffset = _instanceOffset;
    }

    function updateAdmin(address _admin) external {
        require(msg.sender == admin, "Only admin can update admin");
        if(_admin == address(0)) {
            revert();
        }
        admin = _admin;
    }

    function updateAccountCalls(
        address[] memory _contractAddresses,
        bytes[][] memory _callData,
        uint256[][] memory _decimals
    ) external {
        require(msg.sender == admin, "Only admin can update instanceOffset");
        populateAccountCalls(_contractAddresses, _callData, _decimals);
    }

    function populateAccountCalls(
        address[] memory _contractAddresses,
        bytes[][] memory _callData,
        uint256[][] memory _decimals
    ) internal {
        require(
            _contractAddresses.length == _callData.length &&
                accountCalls.length == _contractAddresses.length,
            "Invalid input length"
        );
        require(
            _decimals.length == _contractAddresses.length,
            "Invalid number of decimals"
        );
        // fill in the accountCalls storage array
        uint counter = 0;
        for (uint256 i = 0; i < _contractAddresses.length; i++) {
            AccountCall storage accountCall = accountCalls[i];
            accountCall.contractAddress = _contractAddresses[i];
            accountCall.callCount = _callData[i].length;
            for (uint256 j = 0; j < _callData[i].length; j++) {
                accountCall.callData[j] = _callData[i][j];
                accountCall.decimals[j] = 10 ** _decimals[i][j];
            }
            // count the total number of storage reads across all of the accounts
            counter += _callData[i].length;
        }
        require(counter == INPUT_CALLS + OUTPUT_CALLS, "Invalid number of calls");
    }

    function mulDiv(
        uint256 x,
        uint256 y,
        uint256 denominator
    ) internal pure returns (uint256 result) {
        unchecked {
            uint256 prod0;
            uint256 prod1;
            assembly {
                let mm := mulmod(x, y, not(0))
                prod0 := mul(x, y)
                prod1 := sub(sub(mm, prod0), lt(mm, prod0))
            }

            if (prod1 == 0) {
                return prod0 / denominator;
            }

            require(denominator > prod1, "Math: mulDiv overflow");

            uint256 remainder;
            assembly {
                remainder := mulmod(x, y, denominator)
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
            return result;
        }
    }
    /**
     * @dev Quantize the data returned from the account calls to the scale used by the EZKL model.
     * @param data - The data returned from the account calls.
     * @param decimals - The number of decimals the data returned from the account calls has (for floating point representation).
     * @param scale - The scale used to convert the floating point value into a fixed point value. 
     */
    function quantizeData(
        bytes memory data,
        uint256 decimals,
        uint256 scale
    ) internal pure returns (int256 quantized_data) {
        int x = abi.decode(data, (int256));
        bool neg = x < 0;
        if (neg) x = -x;
        uint output = mulDiv(uint256(x), scale, decimals);
        if (mulmod(uint256(x), scale, decimals) * 2 >= decimals) {
            output += 1;
        }
        // In the interest of keeping feature parity with the quantization done on the EZKL cli,
        // we set the fixed point value type to be int128. Any value greater than that will throw an error
        // as it does on the EZKL cli.
        require(output <= uint128(type(int128).max), "Significant bit truncation");
        quantized_data = neg ? -int256(output): int256(output);
    }
    /**
     * @dev Make a static call to the account to fetch the data that EZKL reads from.
     * @param target - The address of the account to make calls to.
     * @param data  - The abi encoded function calls to make to the `contractAddress` that EZKL reads storage from.
     * @return The data returned from the account calls. (Must come from either a view or pure function. Will throw an error otherwise)
     */
    function staticCall(
        address target,
        bytes memory data
    ) internal view returns (bytes memory) {
        (bool success, bytes memory returndata) = target.staticcall(data);
        if (success) {
            if (returndata.length == 0) {
                require(
                    target.code.length > 0,
                    "Address: call to non-contract"
                );
            }
            return returndata;
        } else {
            revert("Address: low-level call failed");
        }
    }
    /**
     * @dev Convert the fixed point quantized data into a field element.
     * @param x - The quantized data.
     * @return field_element - The field element.
     */
    function toFieldElement(int256 x) internal pure returns (uint256 field_element) {
        // The casting down to uint256 is safe because the order is about 2^254, and the value
        // of x ranges of -2^127 to 2^127, so x + int(ORDER) is always positive.
        return uint256(x + int(ORDER)) % ORDER;
    }

    /**
     * @dev Make the account calls to fetch the data that EZKL reads from and attest to the data.
     * @param instances - The public instances to the proof (the data in the proof that publicly accessible to the verifier).
     */
    function attestData(uint256[] memory instances) internal view {
        require(
            instances.length >= INPUT_CALLS + OUTPUT_CALLS,
            "Invalid public inputs length"
        );
        uint256 _accountCount = accountCalls.length;
        uint counter = 0;
        for (uint8 i = 0; i < _accountCount; ++i) {
            address account = accountCalls[i].contractAddress;
            for (uint8 j = 0; j < accountCalls[i].callCount; j++) {
                bytes memory returnData = staticCall(
                    account,
                    accountCalls[i].callData[j]
                );
                uint256 scale = scales[counter];
                int256 quantized_data = quantizeData(
                    returnData,
                    accountCalls[i].decimals[j],
                    scale
                );
                uint256 field_element = toFieldElement(quantized_data);
                require(
                    field_element == instances[counter + instanceOffset],
                    "Public input does not match"
                );
                counter++;
            }
        }
    }

    function verifyWithDataAttestation(
        address verifier,
        bytes memory encoded
    ) public view returns (bool) {
        require(verifier.code.length > 0,"Address: call to non-contract");
        bytes4 fnSelector;
        uint256[] memory instances;
        bytes memory paramData = new bytes(encoded.length - 4);
        assembly {
            /* 
                4 (fun sig) + 
                32 (verifier address) + 
                32 (offset encoded) + 
                32 (length encoded) = 100 bytes = 0x64
            */
            fnSelector := calldataload(0x64)

            mstore(add(paramData, 0x20), sub(mload(add(encoded, 0x20)), 4))
            for {
                let i := 0
            } lt(i, sub(mload(encoded), 4)) {
                i := add(i, 0x20)
            } {
                mstore(add(paramData, add(0x20, i)), mload(add(encoded, add(0x24, i))))
            }
        }
        if (fnSelector == 0xaf83a18d) {
            // abi decode verifyProof(address,bytes,uint256[])
            (,,instances) = abi.decode(paramData, (address, bytes, uint256[]));
        } else {
            // abi decode verifyProof(bytes,uint256[])
            (,instances) = abi.decode(paramData, (bytes, uint256[]));
        }
        attestData(instances);
        
        // static call the verifier contract to verify the proof
        (bool success, bytes memory returndata) = verifier.staticcall(encoded);

        if (success) {
            return abi.decode(returndata, (bool));
        } else {
            revert("low-level call to verifier failed");
        }
        
    }
}
