use crate::graph::DataSource;
use crate::graph::GraphSettings;
use crate::graph::input::{CallToAccount, CallsToAccount, FileSourceInner, GraphData};
use crate::graph::modules::POSEIDON_INSTANCES;
use crate::pfsys::Snark;
use crate::pfsys::evm::EvmVerificationError;
use alloy::contract::CallBuilder;
use alloy::core::primitives::Address as H160;
use alloy::core::primitives::Bytes;
use alloy::core::primitives::U256;
use alloy::dyn_abi::abi::TokenSeq;
use alloy::dyn_abi::abi::token::{DynSeqToken, PackedSeqToken, WordToken};
// use alloy::providers::Middleware;
use alloy::json_abi::JsonAbi;
use alloy::primitives::ruint::ParseError;
use alloy::primitives::{B256, I256, ParseSignedError};
use alloy::providers::ProviderBuilder;
use alloy::providers::fillers::{
    ChainIdFiller, FillProvider, GasFiller, JoinFill, NonceFiller, SignerFiller,
};
use alloy::providers::network::{Ethereum, EthereumSigner};
use alloy::providers::{Identity, Provider, RootProvider};
use alloy::rpc::types::eth::TransactionInput;
use alloy::rpc::types::eth::TransactionRequest;
use alloy::signers::k256::ecdsa;
use alloy::signers::wallet::{LocalWallet, WalletError};
use alloy::sol as abigen;
use alloy::transports::http::Http;
use alloy::transports::{RpcError, TransportErrorKind};
use foundry_compilers::Solc;
use foundry_compilers::artifacts::Settings as SolcSettings;
use foundry_compilers::error::{SolcError, SolcIoError};
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
    #[sol(rpc, bytecode="608060405234801561000f575f80fd5b506040516105c13803806105c183398181016040528101906100319190610229565b5f5b815181101561008e575f8282815181106100505761004f610270565b5b6020026020010151908060018154018082558091505060019003905f5260205f20015f90919091909150558080610086906102d3565b915050610033565b505061031a565b5f604051905090565b5f80fd5b5f80fd5b5f80fd5b5f601f19601f8301169050919050565b7f4e487b71000000000000000000000000000000000000000000000000000000005f52604160045260245ffd5b6100f0826100aa565b810181811067ffffffffffffffff8211171561010f5761010e6100ba565b5b80604052505050565b5f610121610095565b905061012d82826100e7565b919050565b5f67ffffffffffffffff82111561014c5761014b6100ba565b5b602082029050602081019050919050565b5f80fd5b5f819050919050565b61017381610161565b811461017d575f80fd5b50565b5f8151905061018e8161016a565b92915050565b5f6101a66101a184610132565b610118565b905080838252602082019050602084028301858111156101c9576101c861015d565b5b835b818110156101f257806101de8882610180565b8452602084019350506020810190506101cb565b5050509392505050565b5f82601f8301126102105761020f6100a6565b5b8151610220848260208601610194565b91505092915050565b5f6020828403121561023e5761023d61009e565b5b5f82015167ffffffffffffffff81111561025b5761025a6100a2565b5b610267848285016101fc565b91505092915050565b7f4e487b71000000000000000000000000000000000000000000000000000000005f52603260045260245ffd5b7f4e487b71000000000000000000000000000000000000000000000000000000005f52601160045260245ffd5b5f819050919050565b5f6102dd826102ca565b91507fffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffff820361030f5761030e61029d565b5b600182019050919050565b61029a806103275f395ff3fe608060405234801561000f575f80fd5b5060043610610034575f3560e01c806341f654f71461003857806371e5ee5f14610056575b5f80fd5b610040610086565b60405161004d91906101ba565b60405180910390f35b610070600480360381019061006b9190610211565b6100db565b60405161007d919061024b565b60405180910390f35b60605f8054806020026020016040519081016040528092919081815260200182805480156100d157602002820191905f5260205f20905b8154815260200190600101908083116100bd575b5050505050905090565b5f81815481106100e9575f80fd5b905f5260205f20015f915090505481565b5f81519050919050565b5f82825260208201905092915050565b5f819050602082019050919050565b5f819050919050565b61013581610123565b82525050565b5f610146838361012c565b60208301905092915050565b5f602082019050919050565b5f610168826100fa565b6101728185610104565b935061017d83610114565b805f5b838110156101ad578151610194888261013b565b975061019f83610152565b925050600181019050610180565b5085935050505092915050565b5f6020820190508181035f8301526101d2818461015e565b905092915050565b5f80fd5b5f819050919050565b6101f0816101de565b81146101fa575f80fd5b50565b5f8135905061020b816101e7565b92915050565b5f60208284031215610226576102256101da565b5b5f610233848285016101fd565b91505092915050565b61024581610123565b82525050565b5f60208201905061025e5f83018461023c565b9291505056fea26469706673582212200268dbb6c70ece65633c65d53b537a65ed825c0b61dc5704172b1551958b95ab64736f6c63430008140033")]
    contract TestReads {
        int[] public arr;

        constructor(int256[] memory _numbers) {
            for (uint256 i = 0; i < _numbers.length; i++) {
                arr.push(_numbers[i]);
            }
        }
        function readAll() public view returns (int[] memory) {
            return arr;
        }
    }
);
abigen!(
    #[allow(missing_docs)]
    #[sol(rpc)]
    DataAttestation,
    "./abis/DataAttestation.json"
);
abigen!(
    #[allow(missing_docs)]
    #[sol(rpc, bytecode="608060405234801561000f575f80fd5b506113758061001d5f395ff3fe608060405234801561000f575f80fd5b5060043610610034575f3560e01c80631abe6c1314610038578063e7f0aadc14610068575b5f80fd5b610052600480360381019061004d9190610869565b610098565b60405161005f9190610995565b60405180910390f35b610082600480360381019061007d9190610a65565b610255565b60405161008f9190610bc0565b60405180910390f35b6060835167ffffffffffffffff8111156100b5576100b46106fa565b5b6040519080825280602002602001820160405280156100e35781602001602082028036833780820191505090505b5090505f83600a6100f49190610d3c565b90505f836001901b90505f5b835181101561024b575f806f7fffffffffffffffffffffffffffffff6fffffffffffffffffffffffffffffffff1689848151811061014157610140610d86565b5b6020026020010151111561019c5788838151811061016257610161610d86565b5b60200260200101517f30644e72e131a029b85045b68181585d2833e84879b9709143e1f593f00000016101959190610db3565b91506101be565b8883815181106101af576101ae610d86565b5b60200260200101519150600190505b5f6101ca8387876105cb565b905084600286806101de576101dd610de6565b5b8886096101eb9190610e13565b10610200576001816101fd9190610e54565b90505b8161020b5780610216565b8061021590610e87565b5b87858151811061022957610228610d86565b5b602002602001018181525050505050808061024390610ecd565b915050610100565b5050509392505050565b60605f8480602001905181019061026c9190610ffe565b9050825167ffffffffffffffff811115610289576102886106fa565b5b6040519080825280602002602001820160405280156102b75781602001602082028036833780820191505090505b5091508251815110156102ff576040517f08c379a00000000000000000000000000000000000000000000000000000000081526004016102f6906110c5565b60405180910390fd5b5f5b82518110156105c2575f82828151811061031e5761031d610d86565b5b602002602001015190505f80821290508015610341578161033e90610e87565b91505b5f87848151811061035557610354610d86565b5b6020026020010151600a6103699190610d3c565b90505f87858151811061037f5761037e610d86565b5b60200260200101516001901b90505f6103998583856105cb565b905082600284806103ad576103ac610de6565b5b8488096103ba9190610e13565b106103cf576001816103cc9190610e54565b90505b60017f30644e72e131a029b85045b68181585d2833e84879b9709143e1f593f0000001901c811115610436576040517f08c379a000000000000000000000000000000000000000000000000000000000815260040161042d9061112d565b60405180910390fd5b8315610525577fffffffffffffffffffffffffffffffff80000000000000000000000000000000600f0b8161046a90610e87565b136104aa576040517f08c379a00000000000000000000000000000000000000000000000000000000081526004016104a1906111bb565b60405180910390fd5b7f30644e72e131a029b85045b68181585d2833e84879b9709143e1f593f0000001817f30644e72e131a029b85045b68181585d2833e84879b9709143e1f593f00000016104f791906111d9565b6105019190611219565b88878151811061051457610513610d86565b5b6020026020010181815250506105aa565b6f7fffffffffffffffffffffffffffffff6fffffffffffffffffffffffffffffffff168110610589576040517f08c379a0000000000000000000000000000000000000000000000000000000008152600401610580906112b9565b60405180910390fd5b8088878151811061059d5761059c610d86565b5b6020026020010181815250505b505050505080806105ba90610ecd565b915050610301565b50509392505050565b5f805f80198587098587029250828110838203039150505f8103610603578382816105f9576105f8610de6565b5b04925050506106ce565b808411610645576040517f08c379a000000000000000000000000000000000000000000000000000000000815260040161063c90611321565b60405180910390fd5b5f8486880990508281118203915080830392505f60018619018616905080860495508084049350600181825f0304019050808302841793505f600287600302189050808702600203810290508087026002038102905080870260020381029050808702600203810290508087026002038102905080870260020381029050808502955050505050505b9392505050565b5f604051905090565b5f80fd5b5f80fd5b5f80fd5b5f601f19601f8301169050919050565b7f4e487b71000000000000000000000000000000000000000000000000000000005f52604160045260245ffd5b610730826106ea565b810181811067ffffffffffffffff8211171561074f5761074e6106fa565b5b80604052505050565b5f6107616106d5565b905061076d8282610727565b919050565b5f67ffffffffffffffff82111561078c5761078b6106fa565b5b602082029050602081019050919050565b5f80fd5b5f819050919050565b6107b3816107a1565b81146107bd575f80fd5b50565b5f813590506107ce816107aa565b92915050565b5f6107e66107e184610772565b610758565b905080838252602082019050602084028301858111156108095761080861079d565b5b835b81811015610832578061081e88826107c0565b84526020840193505060208101905061080b565b5050509392505050565b5f82601f8301126108505761084f6106e6565b5b81356108608482602086016107d4565b91505092915050565b5f805f606084860312156108805761087f6106de565b5b5f84013567ffffffffffffffff81111561089d5761089c6106e2565b5b6108a98682870161083c565b93505060206108ba868287016107c0565b92505060406108cb868287016107c0565b9150509250925092565b5f81519050919050565b5f82825260208201905092915050565b5f819050602082019050919050565b5f819050919050565b610910816108fe565b82525050565b5f6109218383610907565b60208301905092915050565b5f602082019050919050565b5f610943826108d5565b61094d81856108df565b9350610958836108ef565b805f5b8381101561098857815161096f8882610916565b975061097a8361092d565b92505060018101905061095b565b5085935050505092915050565b5f6020820190508181035f8301526109ad8184610939565b905092915050565b5f80fd5b5f67ffffffffffffffff8211156109d3576109d26106fa565b5b6109dc826106ea565b9050602081019050919050565b828183375f83830152505050565b5f610a09610a04846109b9565b610758565b905082815260208101848484011115610a2557610a246109b5565b5b610a308482856109e9565b509392505050565b5f82601f830112610a4c57610a4b6106e6565b5b8135610a5c8482602086016109f7565b91505092915050565b5f805f60608486031215610a7c57610a7b6106de565b5b5f84013567ffffffffffffffff811115610a9957610a986106e2565b5b610aa586828701610a38565b935050602084013567ffffffffffffffff811115610ac657610ac56106e2565b5b610ad28682870161083c565b925050604084013567ffffffffffffffff811115610af357610af26106e2565b5b610aff8682870161083c565b9150509250925092565b5f81519050919050565b5f82825260208201905092915050565b5f819050602082019050919050565b610b3b816107a1565b82525050565b5f610b4c8383610b32565b60208301905092915050565b5f602082019050919050565b5f610b6e82610b09565b610b788185610b13565b9350610b8383610b23565b805f5b83811015610bb3578151610b9a8882610b41565b9750610ba583610b58565b925050600181019050610b86565b5085935050505092915050565b5f6020820190508181035f830152610bd88184610b64565b905092915050565b7f4e487b71000000000000000000000000000000000000000000000000000000005f52601160045260245ffd5b5f8160011c9050919050565b5f808291508390505b6001851115610c6257808604811115610c3e57610c3d610be0565b5b6001851615610c4d5780820291505b8081029050610c5b85610c0d565b9450610c22565b94509492505050565b5f82610c7a5760019050610d35565b81610c87575f9050610d35565b8160018114610c9d5760028114610ca757610cd6565b6001915050610d35565b60ff841115610cb957610cb8610be0565b5b8360020a915084821115610cd057610ccf610be0565b5b50610d35565b5060208310610133831016604e8410600b8410161715610d0b5782820a905083811115610d0657610d05610be0565b5b610d35565b610d188484846001610c19565b92509050818404811115610d2f57610d2e610be0565b5b81810290505b9392505050565b5f610d46826107a1565b9150610d51836107a1565b9250610d7e7fffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffff8484610c6b565b905092915050565b7f4e487b71000000000000000000000000000000000000000000000000000000005f52603260045260245ffd5b5f610dbd826107a1565b9150610dc8836107a1565b9250828203905081811115610de057610ddf610be0565b5b92915050565b7f4e487b71000000000000000000000000000000000000000000000000000000005f52601260045260245ffd5b5f610e1d826107a1565b9150610e28836107a1565b9250828202610e36816107a1565b91508282048414831517610e4d57610e4c610be0565b5b5092915050565b5f610e5e826107a1565b9150610e69836107a1565b9250828201905080821115610e8157610e80610be0565b5b92915050565b5f610e91826108fe565b91507f80000000000000000000000000000000000000000000000000000000000000008203610ec357610ec2610be0565b5b815f039050919050565b5f610ed7826107a1565b91507fffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffff8203610f0957610f08610be0565b5b600182019050919050565b5f67ffffffffffffffff821115610f2e57610f2d6106fa565b5b602082029050602081019050919050565b610f48816108fe565b8114610f52575f80fd5b50565b5f81519050610f6381610f3f565b92915050565b5f610f7b610f7684610f14565b610758565b90508083825260208201905060208402830185811115610f9e57610f9d61079d565b5b835b81811015610fc75780610fb38882610f55565b845260208401935050602081019050610fa0565b5050509392505050565b5f82601f830112610fe557610fe46106e6565b5b8151610ff5848260208601610f69565b91505092915050565b5f60208284031215611013576110126106de565b5b5f82015167ffffffffffffffff8111156110305761102f6106e2565b5b61103c84828501610fd1565b91505092915050565b5f82825260208201905092915050565b7f64617461206c656e677468206d7573742062652067726561746572207468616e5f8201527f206f7220657175616c20746f207363616c6573206c656e677468000000000000602082015250565b5f6110af603a83611045565b91506110ba82611055565b604082019050919050565b5f6020820190508181035f8301526110dc816110a3565b9050919050565b7f4f766572666c6f77206669656c64206d6f64756c7573000000000000000000005f82015250565b5f611117601683611045565b9150611122826110e3565b602082019050919050565b5f6020820190508181035f8301526111448161110b565b9050919050565b7f5175616e74697a65642076616c7565206973206c657373207468616e20696e745f8201527f313238206d696e00000000000000000000000000000000000000000000000000602082015250565b5f6111a5602783611045565b91506111b08261114b565b604082019050919050565b5f6020820190508181035f8301526111d281611199565b9050919050565b5f6111e3826108fe565b91506111ee836108fe565b925082820390508181125f8412168282135f85121516171561121357611212610be0565b5b92915050565b5f611223826107a1565b915061122e836107a1565b92508261123e5761123d610de6565b5b828206905092915050565b7f5175616e74697a65642076616c75652069732067726561746572207468616e205f8201527f696e74313238206d617800000000000000000000000000000000000000000000602082015250565b5f6112a3602a83611045565b91506112ae82611249565b604082019050919050565b5f6020820190508181035f8301526112d081611297565b9050919050565b7f4d6174683a206d756c446976206f766572666c6f7700000000000000000000005f82015250565b5f61130b601583611045565b9150611316826112d7565b602082019050919050565b5f6020820190508181035f830152611338816112ff565b905091905056fea2646970667358221220b3a1aa1bd289008a87ce833d9f043f76d5bac7187beef597a9debfee6f64fc6a64736f6c63430008140033")]
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

        uint256 constant HALF_ORDER = ORDER >> 1;

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

        function quantize_data(
            bytes memory data,
            uint256[] memory decimals,
            uint256[] memory scales
        ) external pure returns (uint256[] memory quantized_data) {
            int[] memory _data = abi.decode(data, (int256[]));
            quantized_data = new uint256[](scales.length);
            /// There are cases when both the inputs and the outputs of the model are attested to.
            /// In that case we sometimes only need to return attested to model inputs, not outputs.
            /// Therefore _data.length might be greater than scales.length
            require(_data.length >= scales.length, "data length must be greater than or equal to scales length");
            for (uint i; i < quantized_data.length; i++) {
                int x = _data[i];
                bool neg = x < 0;
                if (neg) x = -x;
                uint denom = 10 ** decimals[i];
                uint scale = 1 << scales[i];
                uint output = mulDiv(uint256(x), scale, denom);
                if (mulmod(uint256(x), scale, denom) * 2 >= denom) {
                    output += 1;
                }

                if (output > HALF_ORDER) {
                    revert("Overflow field modulus");
                }

                if (neg) {
                    // No possibility of overflow here since output is less than or equal to HALF_ORDER
                    // and therefore falls within the max range of int256 without overflow
                    if(-int256(output) <= type(int128).min) {
                        revert("Quantized value is less than int128 min");
                    }
                    quantized_data[i] = uint256(int(ORDER) - int256(output)) % ORDER;
                } else {
                    if(output >= uint128(type(int128).max)) {
                        revert("Quantized value is greater than int128 max");
                    }
                    quantized_data[i] = output;
                }
            }
        }

        function dequantize(
            uint256[] memory instances,
            uint256 decimals,
            uint256 scales
        ) external pure returns (int256[] memory rescaled_instances) {
            rescaled_instances = new int256[](instances.length);
            uint numerator = 10 ** decimals;
            uint denominator = 1 << scales;
            for (uint i; i < rescaled_instances.length; i++) {
                int x;
                bool neg;
                if (instances[i] > uint128(type(int128).max)) {
                    x = int256(ORDER - instances[i]);
                } else {
                    x = int256(instances[i]);
                    neg = true;
                }
                uint output = mulDiv(uint256(x), numerator, denominator);
                if (mulmod(uint256(x), numerator, denominator) * 2 >= denominator) {
                    output += 1;
                }

                rescaled_instances[i] = neg ? -int256(output) : int256(output);
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
    rpc_url: &str,
    private_key: Option<&str>,
) -> Result<(EthersClient, alloy::primitives::Address), EthError> {
    // Launch anvil

    let endpoint = rpc_url.to_string();

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
    rpc_url: &str,
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
    rpc_url: &str,
    runs: usize,
    private_key: Option<&str>,
) -> Result<H160, EthError> {
    let (client, client_address) = setup_eth_backend(rpc_url, private_key).await?;

    let input = GraphData::from_str(&input).map_err(|_| EthError::GraphData)?;

    let settings = GraphSettings::load(&settings_path).map_err(|_| EthError::GraphSettings)?;

    let mut scales: Vec<u32> = vec![];
    // The data that will be stored in the test contracts that will eventually be read from.
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
        // match statement for enum type of source.call
        call_to_account = Some(source.call);
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
        call_to_account = Some(source.call);
        // match statement for enum type of source.calls
    }

    deploy_da_contract(
        client,
        contract_instance_offset,
        client_address,
        scales,
        call_to_account,
        sol_code_path,
        runs,
        &settings,
    )
    .await
}
async fn deploy_da_contract(
    client: EthersClient,
    contract_instance_offset: usize,
    client_address: alloy::primitives::Address,
    scales: Vec<u32>,
    call_to_accounts: Option<CallToAccount>,
    sol_code_path: PathBuf,
    runs: usize,
    settings: &GraphSettings,
) -> Result<H160, EthError> {
    let (abi, bytecode, runtime_bytecode) =
        get_contract_artifacts(sol_code_path, "DataAttestation", runs).await?;
    let (contract_address, call_data, decimals) = if let Some(call_to_accounts) = call_to_accounts {
        parse_call_to_account(call_to_accounts)?
    } else {
        // if calls to accounts is empty then we know need to check that atleast there kzg visibility in the settings file
        let kzg_visibility = settings.run_args.input_visibility.is_polycommit()
            || settings.run_args.output_visibility.is_polycommit()
            || settings.run_args.param_visibility.is_polycommit();
        if !kzg_visibility {
            return Err(EthError::OnChainDataSource);
        }
        let factory =
            get_sol_contract_factory(abi, bytecode, runtime_bytecode, client, None::<()>)?;
        let contract = factory.deploy().await?;
        return Ok(contract);
    };

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
            // uint256 [] _decimals,
            DynSeqToken(
                decimals
                    .iter()
                    .map(|i| WordToken(B256::from(*i)))
                    .collect_vec(),
            ),
            // uint[] memory _bits,
            DynSeqToken(
                scales
                    .clone()
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

    debug!("scales: {:#?}", scales);
    debug!("call_data: {:#?}", call_data);
    debug!("contract_addresses: {:#?}", contract_address);
    debug!("decimals: {:#?}", decimals);

    let contract = factory.deploy().await?;

    Ok(contract)
}
type ParsedCallToAccount = (H160, Bytes, Vec<U256>);

fn parse_call_to_account(call_to_account: CallToAccount) -> Result<ParsedCallToAccount, EthError> {
    let contract_address_bytes = hex::decode(&call_to_account.address)?;
    let contract_address = H160::from_slice(&contract_address_bytes);
    let call_data_bytes = hex::decode(&call_to_account.call_data)?;
    let call_data = Bytes::from(call_data_bytes);
    // Parse the decimals array as uint256 array for the contract.
    // iterate through the decimals array and convert each element to a uint256
    let mut decimals: Vec<U256> = vec![];
    for decimal in &call_to_account.decimals {
        decimals.push(I256::from_dec_str(&decimal.to_string())?.unsigned_abs());
    }
    // let decimal = I256::from_dec_str(&call_to_account.decimals.to_string())?.unsigned_abs();
    Ok((contract_address, call_data, decimals))
}

/// Verify a proof using a Solidity verifier contract
pub async fn verify_proof_via_solidity(
    proof: Snark<Fr, G1Affine>,
    addr: H160,
    addr_vk: Option<H160>,
    rpc_url: &str,
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

fn count_decimal_places(num: f64) -> usize {
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
    for input in data.iter().flatten() {
        if input.is_float() {
            let input = input.to_float();
            let decimal_places = count_decimal_places(input) as u8;
            let scaled_by_decimals = input * f64::powf(10., decimal_places.into());
            scaled_by_decimals_data.push(I256::from_dec_str(
                &(scaled_by_decimals as i128).to_string(),
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
    debug!("scaled_by_decimals_data: {:#?}", scaled_by_decimals_data);
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
    rpc_url: &str,
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
        Token::Address(addr_verifier.0.0.into()),
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
) -> Result<CallToAccount, EthError> {
    let (contract, decimals) = setup_test_contract(client, data).await?;

    // Get the encoded calldata for the input
    let builder = contract.readAll();
    let call = builder.calldata();
    let call_to_account = CallToAccount {
        call_data: hex::encode(call),
        decimals,
        address: hex::encode(contract.address().0.0),
    };
    info!("call_to_account: {:#?}", call_to_account);
    Ok(call_to_account)
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
pub async fn read_on_chain_inputs<M: 'static + Provider<Http<Client>, Ethereum>>(
    client: Arc<M>,
    address: H160,
    data: &CallToAccount,
) -> Result<Bytes, EthError> {
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
    Ok(result)
}

pub async fn evm_quantize<M: 'static + Provider<Http<Client>, Ethereum>>(
    client: Arc<M>,
    scales: Vec<crate::Scale>,
    data: &Bytes,
    decimals: &Vec<u8>,
) -> Result<Vec<Fr>, EthError> {
    let contract = QuantizeData::deploy(&client).await?;

    let fetched_inputs = data;

    let fetched_inputs =
        Result::<_, std::convert::Infallible>::Ok(Bytes::from(fetched_inputs.to_vec()))?;

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
        .quantize_data(fetched_inputs, decimals, scales)
        .call()
        .await?
        .quantized_data;

    debug!("evm quantization results: {:#?}", results);

    let mut felts: Vec<Fr> = vec![];

    for x in results {
        felts.push(PrimeField::from_str_vartime(&x.to_string()).unwrap());
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
        SHANGHAI_SOLC, SolcInput,
        artifacts::{Optimizer, output_selection::OutputSelection},
        compilers::CompilerInput,
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
pub fn fix_da_sol(commitment_bytes: Option<Vec<u8>>, only_kzg: bool) -> Result<String, EthError> {
    let mut contract = ATTESTDATA_SOL.to_string();

    // The case where a combination of on-chain data source + kzg commit is provided.
    if commitment_bytes.is_some() && !commitment_bytes.as_ref().unwrap().is_empty() {
        let commitment_bytes = commitment_bytes.as_ref().unwrap();
        let hex_string = hex::encode(commitment_bytes);
        contract = contract.replace(
            "bytes constant COMMITMENT_KZG = hex\"1234\";",
            &format!("bytes constant COMMITMENT_KZG = hex\"{}\";", hex_string),
        );
        if only_kzg {
            contract = contract.replace(
                "contract SwapProofCommitments {",
                "contract DataAttestation {",
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
    } else {
        // Remove the SwapProofCommitments inheritance and the checkKzgCommits function call if no commitment is provided
        contract = contract.replace(", SwapProofCommitments", "");
        contract = contract.replace(
            "require(checkKzgCommits(encoded), \"Invalid KZG commitments\");",
            "",
        );
    }

    Ok(contract)
}
