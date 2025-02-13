use crate::graph::input::{CallToAccount, Calls, CallsToAccount, FileSourceInner, GraphData};
use crate::graph::modules::POSEIDON_INSTANCES;
use crate::graph::DataSource;
use crate::graph::GraphSettings;
use crate::pfsys::evm::EvmVerificationError;
use crate::pfsys::Snark;
use alloy::contract::CallBuilder;
use alloy::core::primitives::Address as H160;
use alloy::core::primitives::Bytes;
use alloy::core::primitives::U256;
use alloy::dyn_abi::abi::token::{DynSeqToken, PackedSeqToken, WordToken};
use alloy::dyn_abi::abi::TokenSeq;
// use alloy::providers::Middleware;
use alloy::json_abi::JsonAbi;
use alloy::node_bindings::Anvil;
use alloy::primitives::ruint::ParseError;
use alloy::primitives::{ParseSignedError, B256, I256};
use alloy::providers::fillers::{
    ChainIdFiller, FillProvider, GasFiller, JoinFill, NonceFiller, SignerFiller,
};
use alloy::providers::network::{Ethereum, EthereumSigner};
use alloy::providers::ProviderBuilder;
use alloy::providers::{Identity, Provider, RootProvider};
use alloy::rpc::types::eth::TransactionInput;
use alloy::rpc::types::eth::TransactionRequest;
use alloy::signers::k256::ecdsa;
use alloy::signers::wallet::{LocalWallet, WalletError};
use alloy::sol as abigen;
use alloy::transports::http::Http;
use alloy::transports::{RpcError, TransportErrorKind};
use foundry_compilers::artifacts::Settings as SolcSettings;
use foundry_compilers::error::{SolcError, SolcIoError};
use foundry_compilers::Solc;
use halo2_solidity_verifier::encode_calldata;
use halo2curves::bn256::{Fr, G1Affine};
use halo2curves::group::ff::PrimeField;
use itertools::Itertools;
use log::{debug, info, warn};
use reqwest::Client;
use std::path::PathBuf;
use std::str::FromStr;
use std::sync::Arc;

const ANVIL_DEFAULT_PRIVATE_KEY: &str =
    "ac0974bec39a17e36ba4a6b4d238ff944bacb478cbed5efcae784d7bf4f2ff80";

pub const DEFAULT_ANVIL_ENDPOINT: &str = "http://localhost:8545";

// Generate contract bindings OUTSIDE the functions so they are part of library
abigen!(
    #[allow(missing_docs)]
    #[sol(rpc, bytecode="60806040523461012d576102008038038061001981610131565b9283398101602090818382031261012d5782516001600160401b039384821161012d57019080601f8301121561012d5781519384116100fb5760059184831b908480610066818501610131565b80988152019282010192831161012d5784809101915b83831061011d57505050505f5b835181101561010f578281831b850101515f54680100000000000000008110156100fb5760018101805f558110156100e7575f8052845f2001555f1981146100d357600101610089565b634e487b7160e01b5f52601160045260245ffd5b634e487b7160e01b5f52603260045260245ffd5b634e487b7160e01b5f52604160045260245ffd5b60405160a990816101578239f35b825181529181019185910161007c565b5f80fd5b6040519190601f01601f191682016001600160401b038111838210176100fb5760405256fe60808060405260043610156011575f80fd5b5f90813560e01c6371e5ee5f146025575f80fd5b34606f576020366003190112606f576004358254811015606b5782602093527f290decd9548b62a8d60345a988386fc84ba6bc95484008f6362f93160ef3e56301548152f35b8280fd5b5080fdfea2646970667358221220dc28d7ff0d25a49f74c6b97a87c7c6039ee98d715c0f61be72cc4d180d40a41e64736f6c63430008140033")]
    contract TestReads {
        int[] public arr;

        constructor(int256[] memory _numbers) {
            for (uint256 i = 0; i < _numbers.length; i++) {
                arr.push(_numbers[i]);
            }
        }
    }
);
abigen!(
    #[allow(missing_docs)]
    #[sol(rpc)]
    DataAttestationMulti,
    "./abis/DataAttestationMulti.json"
);
abigen!(
    #[allow(missing_docs)]
    #[sol(rpc)]
    DataAttestationSingle,
    "./abis/DataAttestationSingle.json"
);
abigen!(
    #[allow(missing_docs)]
    #[sol(rpc, bytecode="608060405234801561000f575f80fd5b506115598061001d5f395ff3fe608060405234801561000f575f80fd5b506004361061004a575f3560e01c80636b6dd58c1461004e5780639e564bbc1461007e578063b404abab146100ae578063d3dc6d1f146100de575b5f80fd5b61006860048036038101906100639190610987565b61010e565b6040516100759190610acf565b60405180910390f35b61009860048036038101906100939190610bcd565b61027f565b6040516100a59190610acf565b60405180910390f35b6100c860048036038101906100c39190610d5b565b610405565b6040516100d59190610e59565b60405180910390f35b6100f860048036038101906100f39190610f6f565b610554565b6040516101059190610e59565b60405180910390f35b60605f84806020019051810190610125919061105f565b9050805167ffffffffffffffff8111156101425761014161076c565b5b6040519080825280602002602001820160405280156101705781602001602082028036833780820191505090505b5091505f5b8151811015610276575f828281518110610192576101916110a6565b5b602002602001015190505f808212905080156101b557816101b290611100565b91505b5f87600a6101c39190611275565b90505f8785815181106101d9576101d86110a6565b5b60200260200101516001901b90505f6101f3858385610639565b90508260028480610207576102066112bf565b5b84880961021491906112ec565b1061022957600181610226919061132d565b90505b83610234578061023f565b8061023e90611100565b5b888781518110610252576102516110a6565b5b6020026020010181815250505050505050808061026e90611360565b915050610175565b50509392505050565b6060835167ffffffffffffffff81111561029c5761029b61076c565b5b6040519080825280602002602001820160405280156102ca5781602001602082028036833780820191505090505b5090505f5b84518110156103fd575f8582815181106102ec576102eb6110a6565b5b602002602001015180602001905181019061030791906113a7565b90505f80821290508015610322578161031f90611100565b91505b5f868481518110610336576103356110a6565b5b6020026020010151600a61034a9190611275565b90505f8685815181106103605761035f6110a6565b5b60200260200101516001901b90505f61037a858385610639565b9050826002848061038e5761038d6112bf565b5b84880961039b91906112ec565b106103b0576001816103ad919061132d565b90505b836103bb57806103c6565b806103c590611100565b5b8787815181106103d9576103d86110a6565b5b602002602001018181525050505050505080806103f590611360565b9150506102cf565b509392505050565b6060815167ffffffffffffffff8111156104225761042161076c565b5b6040519080825280602002602001820160405280156104505781602001602082028036833780820191505090505b5090505f5b825181101561054e575f838281518110610472576104716110a6565b5b6020026020010151121580156104c257507f30644e72e131a029b85045b68181585d2833e84879b9709143e1f593f00000018382815181106104b7576104b66110a6565b5b602002602001015111155b610501576040517f08c379a00000000000000000000000000000000000000000000000000000000081526004016104f89061142c565b60405180910390fd5b828181518110610514576105136110a6565b5b602002602001015182828151811061052f5761052e6110a6565b5b602002602001018181525050808061054690611360565b915050610455565b50919050565b6060815167ffffffffffffffff8111156105715761057061076c565b5b60405190808252806020026020018201604052801561059f5781602001602082028036833780820191505090505b5090505f5b8251811015610633577f30644e72e131a029b85045b68181585d2833e84879b9709143e1f593f0000001808483815181106105e2576105e16110a6565b5b602002602001015160070b6105f7919061144a565b610601919061148b565b828281518110610614576106136110a6565b5b602002602001018181525050808061062b90611360565b9150506105a4565b50919050565b5f805f80198587098587029250828110838203039150505f810361067157838281610667576106666112bf565b5b049250505061073c565b8084116106b3576040517f08c379a00000000000000000000000000000000000000000000000000000000081526004016106aa90611505565b60405180910390fd5b5f8486880990508281118203915080830392505f60018619018616905080860495508084049350600181825f0304019050808302841793505f600287600302189050808702600203810290508087026002038102905080870260020381029050808702600203810290508087026002038102905080870260020381029050808502955050505050505b9392505050565b5f604051905090565b5f80fd5b5f80fd5b5f80fd5b5f80fd5b5f601f19601f8301169050919050565b7f4e487b71000000000000000000000000000000000000000000000000000000005f52604160045260245ffd5b6107a28261075c565b810181811067ffffffffffffffff821117156107c1576107c061076c565b5b80604052505050565b5f6107d3610743565b90506107df8282610799565b919050565b5f67ffffffffffffffff8211156107fe576107fd61076c565b5b6108078261075c565b9050602081019050919050565b828183375f83830152505050565b5f61083461082f846107e4565b6107ca565b9050828152602081018484840111156108505761084f610758565b5b61085b848285610814565b509392505050565b5f82601f83011261087757610876610754565b5b8135610887848260208601610822565b91505092915050565b5f819050919050565b6108a281610890565b81146108ac575f80fd5b50565b5f813590506108bd81610899565b92915050565b5f67ffffffffffffffff8211156108dd576108dc61076c565b5b602082029050602081019050919050565b5f80fd5b5f6109046108ff846108c3565b6107ca565b90508083825260208201905060208402830185811115610927576109266108ee565b5b835b81811015610950578061093c88826108af565b845260208401935050602081019050610929565b5050509392505050565b5f82601f83011261096e5761096d610754565b5b813561097e8482602086016108f2565b91505092915050565b5f805f6060848603121561099e5761099d61074c565b5b5f84013567ffffffffffffffff8111156109bb576109ba610750565b5b6109c786828701610863565b93505060206109d8868287016108af565b925050604084013567ffffffffffffffff8111156109f9576109f8610750565b5b610a058682870161095a565b9150509250925092565b5f81519050919050565b5f82825260208201905092915050565b5f819050602082019050919050565b5f819050919050565b610a4a81610a38565b82525050565b5f610a5b8383610a41565b60208301905092915050565b5f602082019050919050565b5f610a7d82610a0f565b610a878185610a19565b9350610a9283610a29565b805f5b83811015610ac2578151610aa98882610a50565b9750610ab483610a67565b925050600181019050610a95565b5085935050505092915050565b5f6020820190508181035f830152610ae78184610a73565b905092915050565b5f67ffffffffffffffff821115610b0957610b0861076c565b5b602082029050602081019050919050565b5f610b2c610b2784610aef565b6107ca565b90508083825260208201905060208402830185811115610b4f57610b4e6108ee565b5b835b81811015610b9657803567ffffffffffffffff811115610b7457610b73610754565b5b808601610b818982610863565b85526020850194505050602081019050610b51565b5050509392505050565b5f82601f830112610bb457610bb3610754565b5b8135610bc4848260208601610b1a565b91505092915050565b5f805f60608486031215610be457610be361074c565b5b5f84013567ffffffffffffffff811115610c0157610c00610750565b5b610c0d86828701610ba0565b935050602084013567ffffffffffffffff811115610c2e57610c2d610750565b5b610c3a8682870161095a565b925050604084013567ffffffffffffffff811115610c5b57610c5a610750565b5b610c678682870161095a565b9150509250925092565b5f67ffffffffffffffff821115610c8b57610c8a61076c565b5b602082029050602081019050919050565b610ca581610a38565b8114610caf575f80fd5b50565b5f81359050610cc081610c9c565b92915050565b5f610cd8610cd384610c71565b6107ca565b90508083825260208201905060208402830185811115610cfb57610cfa6108ee565b5b835b81811015610d245780610d108882610cb2565b845260208401935050602081019050610cfd565b5050509392505050565b5f82601f830112610d4257610d41610754565b5b8135610d52848260208601610cc6565b91505092915050565b5f60208284031215610d7057610d6f61074c565b5b5f82013567ffffffffffffffff811115610d8d57610d8c610750565b5b610d9984828501610d2e565b91505092915050565b5f81519050919050565b5f82825260208201905092915050565b5f819050602082019050919050565b610dd481610890565b82525050565b5f610de58383610dcb565b60208301905092915050565b5f602082019050919050565b5f610e0782610da2565b610e118185610dac565b9350610e1c83610dbc565b805f5b83811015610e4c578151610e338882610dda565b9750610e3e83610df1565b925050600181019050610e1f565b5085935050505092915050565b5f6020820190508181035f830152610e718184610dfd565b905092915050565b5f67ffffffffffffffff821115610e9357610e9261076c565b5b602082029050602081019050919050565b5f8160070b9050919050565b610eb981610ea4565b8114610ec3575f80fd5b50565b5f81359050610ed481610eb0565b92915050565b5f610eec610ee784610e79565b6107ca565b90508083825260208201905060208402830185811115610f0f57610f0e6108ee565b5b835b81811015610f385780610f248882610ec6565b845260208401935050602081019050610f11565b5050509392505050565b5f82601f830112610f5657610f55610754565b5b8135610f66848260208601610eda565b91505092915050565b5f60208284031215610f8457610f8361074c565b5b5f82013567ffffffffffffffff811115610fa157610fa0610750565b5b610fad84828501610f42565b91505092915050565b5f81519050610fc481610c9c565b92915050565b5f610fdc610fd784610c71565b6107ca565b90508083825260208201905060208402830185811115610fff57610ffe6108ee565b5b835b8181101561102857806110148882610fb6565b845260208401935050602081019050611001565b5050509392505050565b5f82601f83011261104657611045610754565b5b8151611056848260208601610fca565b91505092915050565b5f602082840312156110745761107361074c565b5b5f82015167ffffffffffffffff81111561109157611090610750565b5b61109d84828501611032565b91505092915050565b7f4e487b71000000000000000000000000000000000000000000000000000000005f52603260045260245ffd5b7f4e487b71000000000000000000000000000000000000000000000000000000005f52601160045260245ffd5b5f61110a82610a38565b91507f8000000000000000000000000000000000000000000000000000000000000000820361113c5761113b6110d3565b5b815f039050919050565b5f8160011c9050919050565b5f808291508390505b600185111561119b57808604811115611177576111766110d3565b5b60018516156111865780820291505b808102905061119485611146565b945061115b565b94509492505050565b5f826111b3576001905061126e565b816111c0575f905061126e565b81600181146111d657600281146111e05761120f565b600191505061126e565b60ff8411156111f2576111f16110d3565b5b8360020a915084821115611209576112086110d3565b5b5061126e565b5060208310610133831016604e8410600b84101617156112445782820a90508381111561123f5761123e6110d3565b5b61126e565b6112518484846001611152565b92509050818404811115611268576112676110d3565b5b81810290505b9392505050565b5f61127f82610890565b915061128a83610890565b92506112b77fffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffff84846111a4565b905092915050565b7f4e487b71000000000000000000000000000000000000000000000000000000005f52601260045260245ffd5b5f6112f682610890565b915061130183610890565b925082820261130f81610890565b91508282048414831517611326576113256110d3565b5b5092915050565b5f61133782610890565b915061134283610890565b925082820190508082111561135a576113596110d3565b5b92915050565b5f61136a82610890565b91507fffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffff820361139c5761139b6110d3565b5b600182019050919050565b5f602082840312156113bc576113bb61074c565b5b5f6113c984828501610fb6565b91505092915050565b5f82825260208201905092915050565b7f496e76616c6964206669656c6420656c656d656e7400000000000000000000005f82015250565b5f6114166015836113d2565b9150611421826113e2565b602082019050919050565b5f6020820190508181035f8301526114438161140a565b9050919050565b5f61145482610a38565b915061145f83610a38565b92508282019050828112155f8312168382125f841215161715611485576114846110d3565b5b92915050565b5f61149582610890565b91506114a083610890565b9250826114b0576114af6112bf565b5b828206905092915050565b7f4d6174683a206d756c446976206f766572666c6f7700000000000000000000005f82015250565b5f6114ef6015836113d2565b91506114fa826114bb565b602082019050919050565b5f6020820190508181035f83015261151c816114e3565b905091905056fea264697066735822122074a33a643064f165fcff9d861d52fd29ef856895dd6c118671a9f5406d87310f64736f6c63430008140033")]
    contract QuantizeData {
        /**
         * @notice EZKL P value
         * @dev In order to prevent the verifier from accepting two version of the same instance, n and the quantity (n + P),  where n + P <= 2^256, we require that all instances are stricly less than P. a
         * @dev The reason for this is that the assmebly code of the verifier performs all arithmetic operations modulo P and as a consequence can't distinguish between n and n + P.
         */
        uint256 constant ORDER =
            uint256(
                0x30644e72e131a029b85045b68181585d2833e84879b9709143e1f593f0000001
            );

        // /**
        //  * @notice Calculates floor(x * y / denominator) with full precision. Throws if result overflows a uint256 or denominator == 0
        //  * @dev Original credit to Remco Bloemen under MIT license (https://xn--2-umb.com/21/muldiv)
        //  * with further edits by Uniswap Labs also under MIT license.
        //  */
        function mulDiv(
            uint256 x,
            uint256 y,
            uint256 denominator
        ) internal pure returns (uint256 result) {
            unchecked {
                // 512-bit multiply [prod1 prod0] = x * y. Compute the product mod 2^256 and mod 2^256 - 1, then use
                // use the Chinese Remainder Theorem to reconstruct the 512 bit result. The result is stored in two 256
                // variables such that product = prod1 * 2^256 + prod0.
                uint256 prod0; // Least significant 256 bits of the product
                uint256 prod1; // Most significant 256 bits of the product
                assembly {
                    let mm := mulmod(x, y, not(0))
                    prod0 := mul(x, y)
                    prod1 := sub(sub(mm, prod0), lt(mm, prod0))
                }

                // Handle non-overflow cases, 256 by 256 division.
                if (prod1 == 0) {
                    // Solidity will revert if denominator == 0, unlike the div opcode on its own.
                    // The surrounding unchecked block does not change this fact.
                    // See https://docs.soliditylang.org/en/latest/control-structures.html#checked-or-unchecked-arithmetic.
                    return prod0 / denominator;
                }

                // Make sure the result is less than 2^256. Also prevents denominator == 0.
                require(denominator > prod1, "Math: mulDiv overflow");

                ///////////////////////////////////////////////
                // 512 by 256 division.
                ///////////////////////////////////////////////

                // Make division exact by subtracting the remainder from [prod1 prod0].
                uint256 remainder;
                assembly {
                    // Compute remainder using mulmod.
                    remainder := mulmod(x, y, denominator)

                    // Subtract 256 bit number from 512 bit number.
                    prod1 := sub(prod1, gt(remainder, prod0))
                    prod0 := sub(prod0, remainder)
                }

                // Factor powers of two out of denominator and compute largest power of two divisor of denominator. Always >= 1.
                // See https://cs.stackexchange.com/q/138556/92363.

                // Does not overflow because the denominator cannot be zero at this stage in the function.
                uint256 twos = denominator & (~denominator + 1);
                assembly {
                    // Divide denominator by twos.
                    denominator := div(denominator, twos)

                    // Divide [prod1 prod0] by twos.
                    prod0 := div(prod0, twos)

                    // Flip twos such that it is 2^256 / twos. If twos is zero, then it becomes one.
                    twos := add(div(sub(0, twos), twos), 1)
                }

                // Shift in bits from prod1 into prod0.
                prod0 |= prod1 * twos;

                // Invert denominator mod 2^256. Now that denominator is an odd number, it has an inverse modulo 2^256 such
                // that denominator * inv = 1 mod 2^256. Compute the inverse by starting with a seed that is correct for
                // four bits. That is, denominator * inv = 1 mod 2^4.
                uint256 inverse = (3 * denominator) ^ 2;

                // Use the Newton-Raphson iteration to improve the precision. Thanks to Hensel's lifting lemma, this also works
                // in modular arithmetic, doubling the correct bits in each step.
                inverse *= 2 - denominator * inverse; // inverse mod 2^8
                inverse *= 2 - denominator * inverse; // inverse mod 2^16
                inverse *= 2 - denominator * inverse; // inverse mod 2^32
                inverse *= 2 - denominator * inverse; // inverse mod 2^64
                inverse *= 2 - denominator * inverse; // inverse mod 2^128
                inverse *= 2 - denominator * inverse; // inverse mod 2^256

                // Because the division is now exact we can divide by multiplying with the modular inverse of denominator.
                // This will give us the correct result modulo 2^256. Since the preconditions guarantee that the outcome is
                // less than 2^256, this is the final result. We don't need to compute the high bits of the result and prod1
                // is no longer required.
                result = prod0 * inverse;
                return result;
            }
        }

        function quantize_data_multi(
            bytes[] memory data,
            uint256[] memory decimals,
            uint256[] memory scales
        ) external pure returns (int256[] memory quantized_data) {
            quantized_data = new int256[](data.length);
            for (uint i; i < data.length; i++) {
                int x = abi.decode(data[i], (int256));
                bool neg = x < 0;
                if (neg) x = -x;
                uint denom = 10 ** decimals[i];
                uint scale = 1 << scales[i];
                uint output = mulDiv(uint256(x), scale, denom);
                if (mulmod(uint256(x), scale, denom) * 2 >= denom) {
                    output += 1;
                }

                quantized_data[i] = neg ? -int256(output) : int256(output);
            }
        }

        function quantize_data_single(
            bytes memory data,
            uint256 memory decimals,
            uint256[] memory scales
        ) external pure returns (int256[] memory quantized_data) {
            int[] memory _data = abi.decode(data, (int256[]));
            quantized_data = new int256[](_data.length);
            for (uint i; i < _data.length; i++) {
                int x = _data[i];
                bool neg = x < 0;
                if (neg) x = -x;
                uint denom = 10 ** decimals;
                uint scale = 1 << scales[i];
                uint output = mulDiv(uint256(x), scale, denom);
                if (mulmod(uint256(x), scale, denom) * 2 >= denom) {
                    output += 1;
                }

                quantized_data[i] = neg ? -int256(output) : int256(output);
            }
        }

        function to_field_element(
            int64[] memory quantized_data
        ) public pure returns (uint256[] memory output) {
            output = new uint256[](quantized_data.length);
            for (uint i; i < quantized_data.length; i++) {
                output[i] = uint256(quantized_data[i] + int(ORDER)) % ORDER;
            }
        }

        function check_is_valid_field_element(
            int256[] memory quantized_data
        ) public pure returns (uint256[] memory output) {
            output = new uint256[](quantized_data.length);
            for (uint i; i < quantized_data.length; i++) {
                // assert it is a positive number and less than ORDER
                require(quantized_data[i] >= 0 && uint256(quantized_data[i]) <= ORDER, "Invalid field element");
                output[i] = uint256(quantized_data[i]);
            }
        }
    }
);

#[derive(Debug, thiserror::Error)]
pub enum EthError {
    #[error("a transport error occurred: {0}")]
    Transport(#[from] RpcError<TransportErrorKind>),
    #[error("a contract error occurred: {0}")]
    Contract(#[from] alloy::contract::Error),
    #[error("a wallet error occurred: {0}")]
    Wallet(#[from] WalletError),
    #[error("failed to parse url {0}")]
    UrlParse(String),
    #[error("evm verification error: {0}")]
    EvmVerification(#[from] EvmVerificationError),
    #[error("Private key must be in hex format, 64 chars, without 0x prefix")]
    PrivateKeyFormat,
    #[error("failed to parse hex: {0}")]
    HexParse(#[from] hex::FromHexError),
    #[error("ecdsa error: {0}")]
    Ecdsa(#[from] ecdsa::Error),
    #[error("failed to load graph data")]
    GraphData,
    #[error("failed to load graph settings")]
    GraphSettings,
    #[error("io error: {0}")]
    Io(#[from] std::io::Error),
    #[error("Data source for either input_data or output_data must be OnChain")]
    OnChainDataSource,
    #[error("failed to parse signed integer: {0}")]
    SignedIntegerParse(#[from] ParseSignedError),
    #[error("failed to parse unsigned integer: {0}")]
    UnSignedIntegerParse(#[from] ParseError),
    #[error("updateAccountCalls should have failed")]
    UpdateAccountCalls,
    #[error("ethabi error: {0}")]
    EthAbi(#[from] ethabi::Error),
    #[error("conversion error: {0}")]
    Conversion(#[from] std::convert::Infallible),
    // Constructor arguments provided but no constructor found
    #[error("constructor arguments provided but no constructor found")]
    NoConstructor,
    #[error("contract not found at path: {0}")]
    ContractNotFound(String),
    #[error("solc error: {0}")]
    Solc(#[from] SolcError),
    #[error("solc io error: {0}")]
    SolcIo(#[from] SolcIoError),
    #[error("svm error: {0}")]
    Svm(String),
    #[error("no contract output found")]
    NoContractOutput,
}

// we have to generate these two contract differently because they are generated dynamically ! and hence the static compilation from above does not suit
const ATTESTDATA_SOL: &str = include_str!("../contracts/AttestData.sol");

pub type EthersClient = Arc<
    FillProvider<
        JoinFill<
            JoinFill<JoinFill<JoinFill<Identity, GasFiller>, NonceFiller>, ChainIdFiller>,
            SignerFiller<EthereumSigner>,
        >,
        RootProvider<Http<Client>>,
        Http<Client>,
        Ethereum,
    >,
>;

pub type ContractFactory<M> = CallBuilder<Http<Client>, Arc<M>, ()>;

/// Return an instance of Anvil and a client for the given RPC URL. If none is provided, a local client is used.
pub async fn setup_eth_backend(
    rpc_url: Option<&str>,
    private_key: Option<&str>,
) -> Result<(EthersClient, alloy::primitives::Address), EthError> {
    // Launch anvil

    let endpoint: String;
    if let Some(rpc_url) = rpc_url {
        endpoint = rpc_url.to_string();
    } else {
        let anvil = Anvil::new()
            .args([
                "--code-size-limit=41943040",
                "--disable-block-gas-limit",
                "-p",
                "8545",
            ])
            .spawn();
        endpoint = anvil.endpoint();
    }

    // Instantiate the wallet
    let wallet: LocalWallet;
    if let Some(private_key) = private_key {
        debug!("using private key {}", private_key);
        if private_key.len() != 64 {
            return Err(EthError::PrivateKeyFormat);
        }
        let private_key_buffer = hex::decode(private_key)?;
        wallet = LocalWallet::from_slice(&private_key_buffer)?;
    } else {
        wallet = LocalWallet::from_str(ANVIL_DEFAULT_PRIVATE_KEY)?;
    }

    let wallet_address = wallet.address();

    // Connect to the network
    let client = Arc::new(
        ProviderBuilder::new()
            .with_recommended_fillers()
            .signer(EthereumSigner::from(wallet))
            .on_http(endpoint.parse().map_err(|_| EthError::UrlParse(endpoint))?),
    );

    let chain_id = client.get_chain_id().await?;
    info!("using chain {}", chain_id);

    Ok((client, wallet_address))
}

///
pub async fn deploy_contract_via_solidity(
    sol_code_path: PathBuf,
    rpc_url: Option<&str>,
    runs: usize,
    private_key: Option<&str>,
    contract_name: &str,
) -> Result<H160, EthError> {
    // anvil instance must be alive at least until the factory completes the deploy
    let (client, _) = setup_eth_backend(rpc_url, private_key).await?;

    let (abi, bytecode, runtime_bytecode) =
        get_contract_artifacts(sol_code_path, contract_name, runs).await?;

    let factory = get_sol_contract_factory(abi, bytecode, runtime_bytecode, client, None::<()>)?;
    let contract = factory.deploy().await?;

    Ok(contract)
}

///
pub async fn deploy_da_verifier_via_solidity(
    settings_path: PathBuf,
    input: String,
    sol_code_path: PathBuf,
    rpc_url: Option<&str>,
    runs: usize,
    private_key: Option<&str>,
) -> Result<H160, EthError> {
    let (client, client_address) = setup_eth_backend(rpc_url, private_key).await?;

    let input = GraphData::from_str(&input).map_err(|_| EthError::GraphData)?;

    let settings = GraphSettings::load(&settings_path).map_err(|_| EthError::GraphSettings)?;

    let mut scales: Vec<u32> = vec![];
    // The data that will be stored in the test contracts that will eventually be read from.
    let mut calls_to_accounts = vec![];
    let mut call_to_account = None;

    let mut instance_shapes = vec![];
    let mut model_instance_offset = 0;

    if settings.run_args.input_visibility.is_hashed() {
        instance_shapes.push(POSEIDON_INSTANCES)
    } else if settings.run_args.input_visibility.is_public() {
        for idx in 0..settings.model_input_scales.len() {
            let shape = &settings.model_instance_shapes[idx];
            instance_shapes.push(shape.iter().product::<usize>());
            model_instance_offset += 1;
        }
    }

    if settings.run_args.param_visibility.is_hashed() {
        return Err(EvmVerificationError::InvalidVisibility.into());
    }

    if settings.run_args.output_visibility.is_hashed() {
        instance_shapes.push(POSEIDON_INSTANCES)
    } else if settings.run_args.output_visibility.is_public() {
        for idx in model_instance_offset..model_instance_offset + settings.model_output_scales.len()
        {
            let shape = &settings.model_instance_shapes[idx];
            instance_shapes.push(shape.iter().product::<usize>());
        }
    }

    let mut instance_idx = 0;
    let mut contract_instance_offset = 0;

    if let DataSource::OnChain(source) = input.input_data {
        if settings.run_args.input_visibility.is_hashed_public() {
            // set scales 1.0
            scales.extend(vec![0; instance_shapes[instance_idx]]);
            instance_idx += 1;
        } else {
            let input_scales = settings.clone().model_input_scales;
            // give each input a scale
            for scale in input_scales {
                scales.extend(vec![scale as u32; instance_shapes[instance_idx]]);
                instance_idx += 1;
            }
        }
        // match statement for enum type of source.calls
        match source.calls {
            Calls::Multiple(calls) => {
                for call in calls {
                    calls_to_accounts.push(call);
                }
            }
            Calls::Single(call) => {
                call_to_account = Some(call);
            }
        }
    } else if let DataSource::File(source) = input.input_data {
        if settings.run_args.input_visibility.is_public() {
            instance_idx += source.len();
            for s in source {
                contract_instance_offset += s.len();
            }
        }
    }

    if let Some(DataSource::OnChain(source)) = input.output_data {
        if settings.run_args.output_visibility.is_hashed_public() {
            // set scales 1.0
            scales.extend(vec![0; instance_shapes[instance_idx]]);
        } else {
            let input_scales = settings.clone().model_output_scales;
            // give each output a scale
            for scale in input_scales {
                scales.extend(vec![scale as u32; instance_shapes[instance_idx]]);
                instance_idx += 1;
            }
        }
        // match statement for enum type of source.calls
        match source.calls {
            Calls::Multiple(calls) => {
                for call in calls {
                    calls_to_accounts.push(call);
                }
            }
            Calls::Single(call) => {
                call_to_account = Some(call);
            }
        }
    }

    match call_to_account {
        Some(call) => {
            deploy_single_da_contract(
                client,
                contract_instance_offset,
                client_address,
                scales,
                call,
                sol_code_path,
                runs,
            )
            .await
        }
        None => {
            deploy_multi_da_contract(
                client,
                contract_instance_offset,
                client_address,
                scales,
                calls_to_accounts,
                sol_code_path,
                runs,
                &settings.clone(),
            )
            .await
        }
    }
}

#[allow(clippy::too_many_arguments)]
async fn deploy_multi_da_contract(
    client: EthersClient,
    contract_instance_offset: usize,
    client_address: alloy::primitives::Address,
    scales: Vec<u32>,
    calls_to_accounts: Vec<CallsToAccount>,
    sol_code_path: PathBuf,
    runs: usize,
    settings: &GraphSettings,
) -> Result<H160, EthError> {
    let (abi, bytecode, runtime_bytecode) =
        get_contract_artifacts(sol_code_path, "DataAttestationMulti", runs).await?;

    let (contract_addresses, call_data, decimals) = if !calls_to_accounts.is_empty() {
        parse_calls_to_accounts(calls_to_accounts)?
    } else {
        // if calls to accounts is empty then we know need to check that atleast there kzg visibility in the settings file
        let kzg_visibility = settings.run_args.input_visibility.is_polycommit()
            || settings.run_args.output_visibility.is_polycommit()
            || settings.run_args.param_visibility.is_polycommit();
        if !kzg_visibility {
            return Err(EthError::OnChainDataSource);
        }
        let factory =
            get_sol_contract_factory::<_, ()>(abi, bytecode, runtime_bytecode, client, None)?;
        let contract = factory.deploy().await?;
        return Ok(contract);
    };

    let factory = get_sol_contract_factory(
        abi,
        bytecode,
        runtime_bytecode,
        client,
        Some((
            // address[] memory _contractAddresses,
            DynSeqToken(
                contract_addresses
                    .iter()
                    .map(|ca| WordToken(ca.into_word()))
                    .collect_vec(),
            ),
            // bytes[][] memory _callData,
            DynSeqToken(
                call_data
                    .iter()
                    .map(|bytes| {
                        DynSeqToken(
                            bytes
                                .iter()
                                .map(|i| PackedSeqToken(i.as_ref()))
                                .collect_vec(),
                        )
                    })
                    .collect::<Vec<_>>(),
            ),
            // uint256[][] memory _decimals,
            DynSeqToken(
                decimals
                    .iter()
                    .map(|ints| {
                        DynSeqToken(ints.iter().map(|i| WordToken(B256::from(*i))).collect_vec())
                    })
                    .collect::<Vec<_>>(),
            ),
            // uint[] memory _scales,
            DynSeqToken(
                scales
                    .into_iter()
                    .map(|i| WordToken(U256::from(i).into()))
                    .collect_vec(),
            ),
            //  uint8 _instanceOffset,
            WordToken(U256::from(contract_instance_offset as u32).into()),
            // address _admin
            WordToken(client_address.into_word()),
        )),
    )?;

    debug!("call_data: {:#?}", call_data);
    debug!("contract_addresses: {:#?}", contract_addresses);
    debug!("decimals: {:#?}", decimals);

    let contract = factory.deploy().await?;

    Ok(contract)
}
async fn deploy_single_da_contract(
    client: EthersClient,
    contract_instance_offset: usize,
    client_address: alloy::primitives::Address,
    scales: Vec<u32>,
    call_to_accounts: CallToAccount,
    sol_code_path: PathBuf,
    runs: usize,
) -> Result<H160, EthError> {
    let (abi, bytecode, runtime_bytecode) =
        get_contract_artifacts(sol_code_path, "DataAttestationSingle", runs).await?;

    let (contract_address, call_data, decimals) = parse_call_to_account(call_to_accounts)?;

    let factory = get_sol_contract_factory(
        abi,
        bytecode,
        runtime_bytecode,
        client,
        Some((
            // address _contractAddress,
            WordToken(contract_address.into_word()),
            // bytes memory _callData,
            PackedSeqToken(call_data.as_ref()),
            // uint256 _decimals,
            WordToken(B256::from(decimals)),
            // uint[] memory _scales,
            DynSeqToken(
                scales
                    .into_iter()
                    .map(|i| WordToken(U256::from(i).into()))
                    .collect_vec(),
            ),
            //  uint8 _instanceOffset,
            WordToken(U256::from(contract_instance_offset as u32).into()),
            // address _admin
            WordToken(client_address.into_word()),
        )),
    )?;

    debug!("call_data: {:#?}", call_data);
    debug!("contract_addresses: {:#?}", contract_address);
    debug!("decimals: {:#?}", decimals);

    let contract = factory.deploy().await?;

    Ok(contract)
}
type ParsedCallsToAccount = (Vec<H160>, Vec<Vec<Bytes>>, Vec<Vec<U256>>);

fn parse_calls_to_accounts(
    calls_to_accounts: Vec<CallsToAccount>,
) -> Result<ParsedCallsToAccount, EthError> {
    let mut contract_addresses = vec![];
    let mut call_data = vec![];
    let mut decimals: Vec<Vec<U256>> = vec![];
    for (i, val) in calls_to_accounts.iter().enumerate() {
        let contract_address_bytes = hex::decode(&val.address)?;
        let contract_address = H160::from_slice(&contract_address_bytes);
        contract_addresses.push(contract_address);
        call_data.push(vec![]);
        decimals.push(vec![]);
        for (call, decimal) in &val.call_data {
            let call_data_bytes = hex::decode(call)?;
            call_data[i].push(Bytes::from(call_data_bytes));
            decimals[i].push(I256::from_dec_str(&decimal.to_string())?.unsigned_abs());
        }
    }
    Ok((contract_addresses, call_data, decimals))
}
type ParsedCallToAccount = (H160, Bytes, U256);

fn parse_call_to_account(call_to_account: CallToAccount) -> Result<ParsedCallToAccount, EthError> {
    let contract_address_bytes = hex::decode(&call_to_account.address)?;
    let contract_address = H160::from_slice(&contract_address_bytes);
    let call_data_bytes = hex::decode(&call_to_account.call_data)?;
    let call_data = Bytes::from(call_data_bytes);
    let decimal = I256::from_dec_str(&call_to_account.decimals.to_string())?.unsigned_abs();
    Ok((contract_address, call_data, decimal))
}

pub async fn update_account_calls(
    addr: H160,
    input: String,
    rpc_url: Option<&str>,
) -> Result<(), EthError> {
    let input = GraphData::from_str(&input).map_err(|_| EthError::GraphData)?;

    // The data that will be stored in the test contracts that will eventually be read from.
    let mut calls_to_accounts = vec![];
    let mut call_to_account = None;

    if let DataSource::OnChain(source) = input.input_data {
        match source.calls {
            Calls::Multiple(calls) => {
                for call in calls {
                    calls_to_accounts.push(call);
                }
            }
            Calls::Single(call) => {
                call_to_account = Some(call);
            }
        }
    }

    if let Some(DataSource::OnChain(source)) = input.output_data {
        match source.calls {
            Calls::Multiple(calls) => {
                for call in calls {
                    calls_to_accounts.push(call);
                }
            }
            Calls::Single(call) => {
                call_to_account = Some(call);
            }
        }
    }

    match call_to_account {
        Some(call) => {
            let (contract_address, call_data, decimals) = parse_call_to_account(call)?;

            let (client, client_address) = setup_eth_backend(rpc_url, None).await?;
            let contract = DataAttestationSingle::new(addr, &client);

            let _ = contract
                .updateAccountCalls(contract_address, call_data.clone(), decimals)
                .from(client_address)
                .send()
                .await?;

            // update contract signer with non admin account
            let contract = DataAttestationSingle::new(addr, client.clone());

            // call to update_account_call should fail
            if (contract
                .updateAccountCalls(contract_address, call_data, decimals)
                .send()
                .await)
                .is_err()
            {
                info!("updateAccountCall failed as expected");
            } else {
                return Err(EthError::UpdateAccountCalls);
            }
        }
        None => {
            let (contract_addresses, call_data, decimals) = if !calls_to_accounts.is_empty() {
                parse_calls_to_accounts(calls_to_accounts)?
            } else {
                return Err(EthError::OnChainDataSource);
            };

            let (client, client_address) = setup_eth_backend(rpc_url, None).await?;
            let contract = DataAttestationMulti::new(addr, &client);

            info!("contract_addresses: {:#?}", contract_addresses);

            let _ = contract
                .updateAccountCalls(
                    contract_addresses.clone(),
                    call_data.clone(),
                    decimals.clone(),
                )
                .from(client_address)
                .send()
                .await?;

            // update contract signer with non admin account
            let contract = DataAttestationMulti::new(addr, client.clone());

            info!("contract_addresses: {:#?}", contract_addresses);

            // call to update_account_calls should fail

            if (contract
                .updateAccountCalls(contract_addresses, call_data, decimals)
                .send()
                .await)
                .is_err()
            {
                info!("updateAccountCalls failed as expected");
            } else {
                return Err(EthError::UpdateAccountCalls);
            }
        }
    }

    Ok(())
}

/// Verify a proof using a Solidity verifier contract
pub async fn verify_proof_via_solidity(
    proof: Snark<Fr, G1Affine>,
    addr: H160,
    addr_vk: Option<H160>,
    rpc_url: Option<&str>,
) -> Result<bool, EthError> {
    let flattened_instances = proof.instances.into_iter().flatten();

    let encoded = encode_calldata(
        addr_vk.as_ref().map(|x| x.0).map(|x| x.0),
        &proof.proof,
        &flattened_instances.collect::<Vec<_>>(),
    );

    debug!("encoded: {:#?}", hex::encode(&encoded));

    let input: TransactionInput = encoded.into();

    let (client, _) = setup_eth_backend(rpc_url, None).await?;
    let tx = TransactionRequest::default().to(addr).input(input);
    debug!("transaction {:#?}", tx);

    let result = client.call(&tx).await;

    if let Err(e) = result {
        return Err(EvmVerificationError::SolidityExecution(e.to_string()).into());
    }
    let result = result?;
    debug!("result: {:#?}", result.to_vec());
    // decode return bytes value into uint8
    let result = result.to_vec().last().ok_or(EthError::NoContractOutput)? == &1u8;
    if !result {
        return Err(EvmVerificationError::InvalidProof.into());
    }

    let gas = client.estimate_gas(&tx).await?;

    info!("estimated verify gas cost: {:#?}", gas);

    // if gas is greater than 30 million warn the user that the gas cost is above ethereum's 30 million block gas limit
    if gas > 30_000_000_u128 {
        warn!(
            "Gas cost of verify transaction is greater than 30 million block gas limit. It will fail on mainnet."
        );
    } else if gas > 15_000_000_u128 {
        warn!(
            "Gas cost of verify transaction is greater than 15 million, the target block size for ethereum"
        );
    }

    Ok(true)
}

fn count_decimal_places(num: f32) -> usize {
    // Convert the number to a string
    let s = num.to_string();

    // Find the decimal point
    match s.find('.') {
        Some(index) => {
            // Count the number of characters after the decimal point
            s[index + 1..].len()
        }
        None => 0,
    }
}

///
pub async fn setup_test_contract<M: 'static + Provider<Http<Client>, Ethereum>>(
    client: Arc<M>,
    data: &[Vec<FileSourceInner>],
) -> Result<(TestReads::TestReadsInstance<Http<Client>, Arc<M>>, Vec<u8>), EthError> {
    let mut decimals = vec![];
    let mut scaled_by_decimals_data = vec![];
    for input in &data[0] {
        if input.is_float() {
            let input = input.to_float() as f32;
            let decimal_places = count_decimal_places(input) as u8;
            let scaled_by_decimals = input * f32::powf(10., decimal_places.into());
            scaled_by_decimals_data.push(I256::from_dec_str(
                &(scaled_by_decimals as i32).to_string(),
            )?);
            decimals.push(decimal_places);
        } else if input.is_field() {
            let input = input.to_field(0);
            let hex_str_fr = format!("{:?}", input);
            // remove the 0x prefix
            let hex_str_fr = &hex_str_fr[2..];
            scaled_by_decimals_data.push(I256::from_raw(U256::from_str_radix(hex_str_fr, 16)?));
            decimals.push(0);
        }
    }

    // Compile the contract
    let contract = TestReads::deploy(client, scaled_by_decimals_data).await?;

    Ok((contract, decimals))
}

/// Verify a proof using a Solidity DataAttestation contract.
/// Used for testing purposes.
pub async fn verify_proof_with_data_attestation(
    proof: Snark<Fr, G1Affine>,
    addr_verifier: H160,
    addr_da: H160,
    addr_vk: Option<H160>,
    rpc_url: Option<&str>,
) -> Result<bool, EthError> {
    use ethabi::{Function, Param, ParamType, StateMutability, Token};

    let mut public_inputs: Vec<U256> = vec![];
    let flattened_instances = proof.instances.into_iter().flatten();

    for val in flattened_instances.clone() {
        let bytes = val.to_repr();
        let u = U256::from_le_slice(bytes.inner().as_slice());
        public_inputs.push(u);
    }

    let encoded_verifier = encode_calldata(
        addr_vk.as_ref().map(|x| x.0).map(|x| x.0),
        &proof.proof,
        &flattened_instances.collect::<Vec<_>>(),
    );

    debug!("encoded: {:#?}", hex::encode(&encoded_verifier));

    debug!("public_inputs: {:#?}", public_inputs);
    debug!("proof: {:#?}", Bytes::from(proof.proof.to_vec()));

    #[allow(deprecated)]
    let func = Function {
        name: "verifyWithDataAttestation".to_owned(),
        inputs: vec![
            Param {
                name: "verifier".to_owned(),
                kind: ParamType::Address,
                internal_type: None,
            },
            Param {
                name: "encoded".to_owned(),
                kind: ParamType::Bytes,
                internal_type: None,
            },
        ],
        outputs: vec![Param {
            name: "success".to_owned(),
            kind: ParamType::Bool,
            internal_type: None,
        }],
        constant: None,
        state_mutability: StateMutability::View,
    };

    let encoded = func.encode_input(&[
        Token::Address(addr_verifier.0 .0.into()),
        Token::Bytes(encoded_verifier),
    ])?;

    debug!("encoded: {:#?}", hex::encode(&encoded));

    let encoded: TransactionInput = encoded.into();

    let (client, _) = setup_eth_backend(rpc_url, None).await?;
    let tx = TransactionRequest::default().to(addr_da).input(encoded);
    debug!("transaction {:#?}", tx);
    info!(
        "estimated verify gas cost: {:#?}",
        client.estimate_gas(&tx).await?
    );

    let result = client.call(&tx).await;
    if let Err(e) = result {
        return Err(EvmVerificationError::SolidityExecution(e.to_string()).into());
    }
    let result = result?;
    debug!("result: {:#?}", result);
    // decode return bytes value into uint8
    let result = result.to_vec().last().ok_or(EthError::NoContractOutput)? == &1u8;
    if !result {
        return Err(EvmVerificationError::InvalidProof.into());
    }

    Ok(true)
}

/// Tests on-chain data storage by deploying a contract that stores the network input and or output
/// data in its storage. It does this by converting the floating point values to integers and storing the
/// the number of decimals of the floating point value on chain.
pub async fn test_on_chain_data<M: 'static + Provider<Http<Client>, Ethereum>>(
    client: Arc<M>,
    data: &[Vec<FileSourceInner>],
) -> Result<Vec<CallsToAccount>, EthError> {
    let (contract, decimals) = setup_test_contract(client, data).await?;

    // Get the encoded call data for each input
    let mut calldata = vec![];
    for (i, _) in data.iter().flatten().enumerate() {
        let builder = contract.arr(U256::from(i));
        let call = builder.calldata();
        // Push (call, decimals) to the calldata vector.
        calldata.push((hex::encode(call), decimals[i]));
    }
    // Instantiate a new CallsToAccount struct
    let calls_to_account = CallsToAccount {
        call_data: calldata,
        address: hex::encode(contract.address().0 .0),
    };
    info!("calls_to_account: {:#?}", calls_to_account);
    Ok(vec![calls_to_account])
}

/// Reads on-chain inputs, returning the raw encoded data returned from making all the calls in on_chain_input_data
pub async fn read_on_chain_inputs_multi<M: 'static + Provider<Http<Client>, Ethereum>>(
    client: Arc<M>,
    address: H160,
    data: &Vec<CallsToAccount>,
) -> Result<(Vec<Bytes>, Vec<u8>), EthError> {
    // Iterate over all on-chain inputs

    let mut fetched_inputs = vec![];
    let mut decimals = vec![];
    for on_chain_data in data {
        // Construct the address
        let contract_address_bytes = hex::decode(&on_chain_data.address)?;
        let contract_address = H160::from_slice(&contract_address_bytes);
        for (call_data, decimal) in &on_chain_data.call_data {
            let call_data_bytes = hex::decode(call_data)?;
            let input: TransactionInput = call_data_bytes.into();

            let tx = TransactionRequest::default()
                .to(contract_address)
                .from(address)
                .input(input);
            debug!("transaction {:#?}", tx);

            let result = client.call(&tx).await?;
            debug!("return data {:#?}", result);
            fetched_inputs.push(result);
            decimals.push(*decimal);
        }
    }
    Ok((fetched_inputs, decimals))
}

/// Reads on-chain inputs, returning the raw encoded data returned from making the single call in on_chain_input_data
/// that returns the array of input data we will attest to.
pub async fn read_on_chain_inputs_single<M: 'static + Provider<Http<Client>, Ethereum>>(
    client: Arc<M>,
    address: H160,
    data: CallToAccount,
) -> Result<(Bytes, u8), EthError> {
    // Iterate over all on-chain inputs
    let contract_address_bytes = hex::decode(&data.address)?;
    let contract_address = H160::from_slice(&contract_address_bytes);
    let call_data_bytes = hex::decode(&data.call_data)?;
    let input: TransactionInput = call_data_bytes.into();
    let tx = TransactionRequest::default()
        .to(contract_address)
        .from(address)
        .input(input);
    debug!("transaction {:#?}", tx);

    let result = client.call(&tx).await?;
    debug!("return data {:#?}", result);
    Ok((result, data.decimals))
}

///
pub async fn evm_quantize_multi<M: 'static + Provider<Http<Client>, Ethereum>>(
    client: Arc<M>,
    scales: Vec<crate::Scale>,
    data: &(Vec<Bytes>, Vec<u8>),
) -> Result<Vec<Fr>, EthError> {
    let contract = QuantizeData::deploy(&client).await?;

    let fetched_inputs = &data.0;
    let decimals = &data.1;

    let fetched_inputs = fetched_inputs
        .iter()
        .map(|x| Result::<_, std::convert::Infallible>::Ok(Bytes::from(x.to_vec())))
        .collect::<Result<Vec<Bytes>, _>>()?;

    let decimals = decimals
        .iter()
        .map(|x| Ok(I256::from_dec_str(&x.to_string())?.unsigned_abs()))
        .collect::<Result<Vec<U256>, ParseSignedError>>()?;

    let scales = scales
        .iter()
        .map(|x| Ok(I256::from_dec_str(&x.to_string())?.unsigned_abs()))
        .collect::<Result<Vec<U256>, ParseSignedError>>()?;

    debug!("scales: {:#?}", scales);
    debug!("decimals: {:#?}", decimals);
    debug!("fetched_inputs: {:#?}", fetched_inputs);

    let results = contract
        .quantize_data_multi(fetched_inputs, decimals, scales)
        .call()
        .await?
        .quantized_data;

    debug!("evm quantization results: {:#?}", results);

    let mut felts = vec![];

    for x in results {
        let felt = match i64::from_str(&x.to_string()) {
            Ok(x) => contract.to_field_element(vec![x]).call().await?.output[0],
            Err(_) => {
                contract
                    .check_is_valid_field_element(vec![x])
                    .call()
                    .await?
                    .output[0]
            }
        };
        felts.push(PrimeField::from_str_vartime(&felt.to_string()).unwrap());
    }

    debug!("evm quantized felts: {:#?}", felts,);
    Ok(felts)
}

pub async fn evm_quantize_single<M: 'static + Provider<Http<Client>, Ethereum>>(
    client: Arc<M>,
    scales: Vec<crate::Scale>,
    data: &Bytes,
    decimals: u8,
) -> Result<Vec<Fr>, EthError> {
    let contract = QuantizeData::deploy(&client).await?;

    let fetched_inputs = data;

    let fetched_inputs =
        Result::<_, std::convert::Infallible>::Ok(Bytes::from(fetched_inputs.to_vec()))?;

    let decimals = I256::from_dec_str(&decimals.to_string())?.unsigned_abs();

    let scales = scales
        .iter()
        .map(|x| Ok(I256::from_dec_str(&x.to_string())?.unsigned_abs()))
        .collect::<Result<Vec<U256>, ParseSignedError>>()?;

    debug!("scales: {:#?}", scales);
    debug!("decimals: {:#?}", decimals);
    debug!("fetched_inputs: {:#?}", fetched_inputs);

    let results = contract
        .quantize_data_single(fetched_inputs, decimals, scales)
        .call()
        .await?
        .quantized_data;

    debug!("evm quantization results: {:#?}", results);

    let mut felts = vec![];

    for x in results {
        let felt = match i64::from_str(&x.to_string()) {
            Ok(x) => contract.to_field_element(vec![x]).call().await?.output[0],
            Err(_) => {
                contract
                    .check_is_valid_field_element(vec![x])
                    .call()
                    .await?
                    .output[0]
            }
        };
        felts.push(PrimeField::from_str_vartime(&felt.to_string()).unwrap());
    }

    debug!("evm quantized felts: {:#?}", felts,);
    Ok(felts)
}

/// Generates the contract factory for a solidity verifier. The factory is used to deploy the contract
fn get_sol_contract_factory<'a, M: 'static + Provider<Http<Client>, Ethereum>, T: TokenSeq<'a>>(
    abi: JsonAbi,
    bytecode: Bytes,
    runtime_bytecode: Bytes,
    client: Arc<M>,
    params: Option<T>,
) -> Result<ContractFactory<M>, EthError> {
    const MAX_RUNTIME_BYTECODE_SIZE: usize = 24577;
    let size = runtime_bytecode.len();
    debug!("runtime bytecode size: {:#?}", size);
    if size > MAX_RUNTIME_BYTECODE_SIZE {
        // `_runtime_bytecode` exceeds the limit
        warn!(
            "Solidity runtime bytecode size is: {:#?},
            which exceeds 24577 bytes spurious dragon limit.
            Contract will fail to deploy on any chain with
            EIP 140 enabled",
            size
        );
    }

    // Encode the constructor args & concatenate with the bytecode if necessary
    let data: Bytes = match (abi.constructor(), params.is_none()) {
        (None, false) => {
            return Err(EthError::NoConstructor);
        }
        (None, true) => bytecode,
        (Some(_), _) => {
            let mut data = bytecode.to_vec();

            if let Some(params) = params {
                let params = alloy::sol_types::abi::encode_sequence(&params);
                data.extend(params);
            }
            data.into()
        }
    };

    Ok(CallBuilder::new_raw_deploy(client, data))
}

/// Compiles a solidity verifier contract and returns the abi, bytecode, and runtime bytecode
pub async fn get_contract_artifacts(
    sol_code_path: PathBuf,
    contract_name: &str,
    runs: usize,
) -> Result<(JsonAbi, Bytes, Bytes), EthError> {
    use foundry_compilers::{
        artifacts::{output_selection::OutputSelection, Optimizer},
        compilers::CompilerInput,
        SolcInput, SHANGHAI_SOLC,
    };

    if !sol_code_path.exists() {
        return Err(EthError::ContractNotFound(
            sol_code_path.to_string_lossy().to_string(),
        ));
    }

    let settings = SolcSettings {
        optimizer: Optimizer {
            enabled: Some(true),
            runs: Some(runs),
            details: None,
        },
        output_selection: OutputSelection::default_output_selection(),
        ..Default::default()
    };

    let input = SolcInput::build(
        std::collections::BTreeMap::from([(
            sol_code_path.clone(),
            foundry_compilers::artifacts::Source::read(sol_code_path)?,
        )]),
        settings,
        &SHANGHAI_SOLC,
    );

    let solc_opt = Solc::find_svm_installed_version(SHANGHAI_SOLC.to_string())?;
    let solc = match solc_opt {
        Some(solc) => solc,
        None => {
            info!("required solc version is missing ... installing");
            Solc::install(&SHANGHAI_SOLC)
                .await
                .map_err(|e| EthError::Svm(e.to_string()))?
        }
    };

    let compiled: foundry_compilers::CompilerOutput = solc.compile(&input[0])?;

    let (abi, bytecode, runtime_bytecode) = match compiled.find(contract_name) {
        Some(c) => c.into_parts_or_default(),
        None => {
            return Err(EthError::ContractNotFound(contract_name.to_string()));
        }
    };

    Ok((abi, bytecode, runtime_bytecode))
}

/// Sets the constants stored in the da verifier
pub fn fix_da_multi_sol(
    input_data: Option<Vec<CallsToAccount>>,
    output_data: Option<Vec<CallsToAccount>>,
    commitment_bytes: Option<Vec<u8>>,
) -> Result<String, EthError> {
    let mut accounts_len = 0;
    let mut contract = ATTESTDATA_SOL.to_string();

    // fill in the quantization params and total calls
    // as constants to the contract to save on gas
    if let Some(input_data) = &input_data {
        let input_calls: usize = input_data.iter().map(|v| v.call_data.len()).sum();
        accounts_len = input_data.len();
        contract = contract.replace(
            "uint256 constant INPUT_CALLS = 0;",
            &format!("uint256 constant INPUT_CALLS = {};", input_calls),
        );
    }
    if let Some(output_data) = &output_data {
        let output_calls: usize = output_data.iter().map(|v| v.call_data.len()).sum();
        accounts_len += output_data.len();
        contract = contract.replace(
            "uint256 constant OUTPUT_CALLS = 0;",
            &format!("uint256 constant OUTPUT_CALLS = {};", output_calls),
        );
    }
    contract = contract.replace("AccountCall[]", &format!("AccountCall[{}]", accounts_len));

    // The case where a combination of on-chain data source + kzg commit is provided.
    if commitment_bytes.is_some() && !commitment_bytes.as_ref().unwrap().is_empty() {
        let commitment_bytes = commitment_bytes.as_ref().unwrap();
        let hex_string = hex::encode(commitment_bytes);
        contract = contract.replace(
            "bytes constant COMMITMENT_KZG = hex\"\";",
            &format!("bytes constant COMMITMENT_KZG = hex\"{}\";", hex_string),
        );
    } else {
        // Remove the SwapProofCommitments inheritance and the checkKzgCommits function call if no commitment is provided
        contract = contract.replace(", SwapProofCommitments", "");
        contract = contract.replace(
            "require(checkKzgCommits(encoded), \"Invalid KZG commitments\");",
            "",
        );
    }

    // if both input and output data is none then we will only deploy the DataAttest contract, adding in the verifyWithDataAttestationMulti function
    if input_data.is_none()
        && output_data.is_none()
        && commitment_bytes.as_ref().is_some()
        && !commitment_bytes.as_ref().unwrap().is_empty()
    {
        contract = contract.replace(
            "contract SwapProofCommitments {",
            "contract DataAttestationMulti {",
        );

        // Remove everything past the end of the checkKzgCommits function
        if let Some(pos) = contract.find("    } /// end checkKzgCommits") {
            contract.truncate(pos);
            contract.push('}');
        }

        // Add the Solidity function below checkKzgCommits
        contract.push_str(
            r#"
    function verifyWithDataAttestation(
        address verifier,
        bytes calldata encoded
    ) public view returns (bool) {
        require(verifier.code.length > 0, "Address: call to non-contract");
        require(checkKzgCommits(encoded), "Invalid KZG commitments");
        // static call the verifier contract to verify the proof
        (bool success, bytes memory returndata) = verifier.staticcall(encoded);

        if (success) {
            return abi.decode(returndata, (bool));
        } else {
            revert("low-level call to verifier failed");
        }
    }
}"#,
        );
    }

    Ok(contract)
}

/// Sets the constants stored in the da verifier
pub fn fix_da_single_sol(
    input_len: Option<usize>,
    output_len: Option<usize>,
) -> Result<String, EthError> {
    let mut contract = ATTESTDATA_SOL.to_string();

    // fill in the quantization params and total calls
    // as constants to the contract to save on gas
    if let Some(input_len) = &input_len {
        contract = contract.replace(
            "uint256 constant INPUT_LEN = 0;",
            &format!("uint256 constant INPUT_LEN = {};", input_len),
        );
    }
    if let Some(output_len) = &output_len {
        contract = contract.replace(
            "uint256 constant OUTPUT_LEN = 0;",
            &format!("uint256 constant OUTPUT_LEN = {};", output_len),
        );
    }
    Ok(contract)
}
