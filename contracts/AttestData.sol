// SPDX-License-Identifier: MIT
pragma solidity ^0.8.17;

contract DataAttestationVerifier {
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

    uint public constant INPUT_SCALE = 1 << 0;
    uint[] public outputScales;


    /**
     * @notice EZKL P value 
     * @dev In order to prevent the verifier from accepting two version of the same pubInput, n and the quantity (n + P),  where n + P <= 2^256, we require that all instances are stricly less than P. a
     * @dev The reason for this is that the assmebly code of the verifier performs all arithmetic operations modulo P and as a consequence can't distinguish between n and n + P.
     */
    uint256 constant SIZE_LIMIT = 21888242871839275222246405745257275088696311157297823662689037894645226208583; 

    uint256 constant INPUT_CALLS = 0;

    uint256 constant OUTPUT_CALLS = 0;

    uint8 public instanceOffset;

    //bytes[] constant output_scales = bytes(0x000);

    /**
     * @dev Initialize the contract with account calls the EZKL model will read from.
     * @param _contractAddresses - The calls to all the contracts EZKL reads storage from.
     * @param _callData - The abi encoded function calls to make to the `contractAddress` that EZKL reads storage from.
     */
    constructor(
        address[] memory _contractAddresses,
        bytes[][] memory _callData,
        uint256[][] memory _decimals,
        uint[] memory _outputScales,
        uint8 _instanceOffset
    ) {
        for (uint i; i < _outputScales.length; i++) {
            outputScales.push(1 << _outputScales[i]);
        }
        require(
            _contractAddresses.length == _callData.length &&
                accountCalls.length == _contractAddresses.length,
            "Invalid input length"
        );
        require(
            _decimals.length == _contractAddresses.length,
            "Invalid number of decimals"
        );
        uint total_calls = INPUT_CALLS + OUTPUT_CALLS;
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
        require(counter == total_calls, "Invalid number of calls");

        instanceOffset = _instanceOffset;
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

    function quantize_data(
        bytes memory data,
        uint256 decimals,
        uint256 scale
    ) internal pure returns (uint128 quantized_data) {
        uint x = abi.decode(data, (uint256));
        uint output = mulDiv(x, scale, decimals);
        if (mulmod(x, scale, decimals) * 2 >= decimals) {
            output += 1;
        }
        require(output < SIZE_LIMIT, "QuantizeData: overflow");
        quantized_data = uint128(output);
    }

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

    function attestData(uint256[] calldata instances) internal view {
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
                uint256 scale = INPUT_SCALE;
                if (counter >= INPUT_CALLS) {
                    scale = outputScales[counter - INPUT_CALLS];
                }
                uint256 quantized_data = quantize_data(
                    returnData,
                    accountCalls[i].decimals[j],
                    scale
                );
                require(
                    quantized_data == instances[counter + instanceOffset],
                    "Public input does not match"
                );
                counter++;
            }
        }
    }

    function verifyWithDataAttestation(
        uint256[] calldata instances,
        bytes calldata proof
    ) public view returns (bool) {
        bool success = true;
        bytes32[] memory transcript;
        attestData(instances);
        assembly { /* This is where the proof verification happens*/ }
        return success;
    }
}
