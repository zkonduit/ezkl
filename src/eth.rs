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
    #[sol(rpc, bytecode="608060405234801561000f575f80fd5b506040516105c13803806105c183398181016040528101906100319190610229565b5f5b815181101561008e575f8282815181106100505761004f610270565b5b6020026020010151908060018154018082558091505060019003905f5260205f20015f90919091909150558080610086906102d3565b915050610033565b505061031a565b5f604051905090565b5f80fd5b5f80fd5b5f80fd5b5f601f19601f8301169050919050565b7f4e487b71000000000000000000000000000000000000000000000000000000005f52604160045260245ffd5b6100f0826100aa565b810181811067ffffffffffffffff8211171561010f5761010e6100ba565b5b80604052505050565b5f610121610095565b905061012d82826100e7565b919050565b5f67ffffffffffffffff82111561014c5761014b6100ba565b5b602082029050602081019050919050565b5f80fd5b5f819050919050565b61017381610161565b811461017d575f80fd5b50565b5f8151905061018e8161016a565b92915050565b5f6101a66101a184610132565b610118565b905080838252602082019050602084028301858111156101c9576101c861015d565b5b835b818110156101f257806101de8882610180565b8452602084019350506020810190506101cb565b5050509392505050565b5f82601f8301126102105761020f6100a6565b5b8151610220848260208601610194565b91505092915050565b5f6020828403121561023e5761023d61009e565b5b5f82015167ffffffffffffffff81111561025b5761025a6100a2565b5b610267848285016101fc565b91505092915050565b7f4e487b71000000000000000000000000000000000000000000000000000000005f52603260045260245ffd5b7f4e487b71000000000000000000000000000000000000000000000000000000005f52601160045260245ffd5b5f819050919050565b5f6102dd826102ca565b91507fffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffff820361030f5761030e61029d565b5b600182019050919050565b61029a806103275f395ff3fe608060405234801561000f575f80fd5b5060043610610034575f3560e01c806341f654f71461003857806371e5ee5f14610056575b5f80fd5b610040610086565b60405161004d91906101ba565b60405180910390f35b610070600480360381019061006b9190610211565b6100db565b60405161007d919061024b565b60405180910390f35b60605f8054806020026020016040519081016040528092919081815260200182805480156100d157602002820191905f5260205f20905b8154815260200190600101908083116100bd575b5050505050905090565b5f81815481106100e9575f80fd5b905f5260205f20015f915090505481565b5f81519050919050565b5f82825260208201905092915050565b5f819050602082019050919050565b5f819050919050565b61013581610123565b82525050565b5f610146838361012c565b60208301905092915050565b5f602082019050919050565b5f610168826100fa565b6101728185610104565b935061017d83610114565b805f5b838110156101ad578151610194888261013b565b975061019f83610152565b925050600181019050610180565b5085935050505092915050565b5f6020820190508181035f8301526101d2818461015e565b905092915050565b5f80fd5b5f819050919050565b6101f0816101de565b81146101fa575f80fd5b50565b5f8135905061020b816101e7565b92915050565b5f60208284031215610226576102256101da565b5b5f610233848285016101fd565b91505092915050565b61024581610123565b82525050565b5f60208201905061025e5f83018461023c565b9291505056fea26469706673582212204750739470e91a44d3644a347a636f68dc57278161e9b45547ed2b4eab8eccda64736f6c63430008140033")]
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
    #[sol(rpc, bytecode="608060405234801561000f575f80fd5b5061158f8061001d5f395ff3fe608060405234801561000f575f80fd5b506004361061004a575f3560e01c806345ab50981461004e5780639e564bbc1461007e578063b404abab146100ae578063d3dc6d1f146100de575b5f80fd5b610068600480360381019061006391906109a1565b61010e565b6040516100759190610b05565b60405180910390f35b61009860048036038101906100939190610c03565b610299565b6040516100a59190610b05565b60405180910390f35b6100c860048036038101906100c39190610d91565b61041f565b6040516100d59190610e8f565b60405180910390f35b6100f860048036038101906100f39190610fa5565b61056e565b6040516101059190610e8f565b60405180910390f35b60605f848060200190518101906101259190611095565b9050805167ffffffffffffffff81111561014257610141610786565b5b6040519080825280602002602001820160405280156101705781602001602082028036833780820191505090505b5091505f5b8151811015610290575f828281518110610192576101916110dc565b5b602002602001015190505f808212905080156101b557816101b290611136565b91505b5f8784815181106101c9576101c86110dc565b5b6020026020010151600a6101dd91906112ab565b90505f8785815181106101f3576101f26110dc565b5b60200260200101516001901b90505f61020d858385610653565b90508260028480610221576102206112f5565b5b84880961022e9190611322565b10610243576001816102409190611363565b90505b8361024e5780610259565b8061025890611136565b5b88878151811061026c5761026b6110dc565b5b6020026020010181815250505050505050808061028890611396565b915050610175565b50509392505050565b6060835167ffffffffffffffff8111156102b6576102b5610786565b5b6040519080825280602002602001820160405280156102e45781602001602082028036833780820191505090505b5090505f5b8451811015610417575f858281518110610306576103056110dc565b5b602002602001015180602001905181019061032191906113dd565b90505f8082129050801561033c578161033990611136565b91505b5f8684815181106103505761034f6110dc565b5b6020026020010151600a61036491906112ab565b90505f86858151811061037a576103796110dc565b5b60200260200101516001901b90505f610394858385610653565b905082600284806103a8576103a76112f5565b5b8488096103b59190611322565b106103ca576001816103c79190611363565b90505b836103d557806103e0565b806103df90611136565b5b8787815181106103f3576103f26110dc565b5b6020026020010181815250505050505050808061040f90611396565b9150506102e9565b509392505050565b6060815167ffffffffffffffff81111561043c5761043b610786565b5b60405190808252806020026020018201604052801561046a5781602001602082028036833780820191505090505b5090505f5b8251811015610568575f83828151811061048c5761048b6110dc565b5b6020026020010151121580156104dc57507f30644e72e131a029b85045b68181585d2833e84879b9709143e1f593f00000018382815181106104d1576104d06110dc565b5b602002602001015111155b61051b576040517f08c379a000000000000000000000000000000000000000000000000000000000815260040161051290611462565b60405180910390fd5b82818151811061052e5761052d6110dc565b5b6020026020010151828281518110610549576105486110dc565b5b602002602001018181525050808061056090611396565b91505061046f565b50919050565b6060815167ffffffffffffffff81111561058b5761058a610786565b5b6040519080825280602002602001820160405280156105b95781602001602082028036833780820191505090505b5090505f5b825181101561064d577f30644e72e131a029b85045b68181585d2833e84879b9709143e1f593f0000001808483815181106105fc576105fb6110dc565b5b602002602001015160070b6106119190611480565b61061b91906114c1565b82828151811061062e5761062d6110dc565b5b602002602001018181525050808061064590611396565b9150506105be565b50919050565b5f805f80198587098587029250828110838203039150505f810361068b57838281610681576106806112f5565b5b0492505050610756565b8084116106cd576040517f08c379a00000000000000000000000000000000000000000000000000000000081526004016106c49061153b565b60405180910390fd5b5f8486880990508281118203915080830392505f60018619018616905080860495508084049350600181825f0304019050808302841793505f600287600302189050808702600203810290508087026002038102905080870260020381029050808702600203810290508087026002038102905080870260020381029050808502955050505050505b9392505050565b5f604051905090565b5f80fd5b5f80fd5b5f80fd5b5f80fd5b5f601f19601f8301169050919050565b7f4e487b71000000000000000000000000000000000000000000000000000000005f52604160045260245ffd5b6107bc82610776565b810181811067ffffffffffffffff821117156107db576107da610786565b5b80604052505050565b5f6107ed61075d565b90506107f982826107b3565b919050565b5f67ffffffffffffffff82111561081857610817610786565b5b61082182610776565b9050602081019050919050565b828183375f83830152505050565b5f61084e610849846107fe565b6107e4565b90508281526020810184848401111561086a57610869610772565b5b61087584828561082e565b509392505050565b5f82601f8301126108915761089061076e565b5b81356108a184826020860161083c565b91505092915050565b5f67ffffffffffffffff8211156108c4576108c3610786565b5b602082029050602081019050919050565b5f80fd5b5f819050919050565b6108eb816108d9565b81146108f5575f80fd5b50565b5f81359050610906816108e2565b92915050565b5f61091e610919846108aa565b6107e4565b90508083825260208201905060208402830185811115610941576109406108d5565b5b835b8181101561096a578061095688826108f8565b845260208401935050602081019050610943565b5050509392505050565b5f82601f8301126109885761098761076e565b5b813561099884826020860161090c565b91505092915050565b5f805f606084860312156109b8576109b7610766565b5b5f84013567ffffffffffffffff8111156109d5576109d461076a565b5b6109e18682870161087d565b935050602084013567ffffffffffffffff811115610a0257610a0161076a565b5b610a0e86828701610974565b925050604084013567ffffffffffffffff811115610a2f57610a2e61076a565b5b610a3b86828701610974565b9150509250925092565b5f81519050919050565b5f82825260208201905092915050565b5f819050602082019050919050565b5f819050919050565b610a8081610a6e565b82525050565b5f610a918383610a77565b60208301905092915050565b5f602082019050919050565b5f610ab382610a45565b610abd8185610a4f565b9350610ac883610a5f565b805f5b83811015610af8578151610adf8882610a86565b9750610aea83610a9d565b925050600181019050610acb565b5085935050505092915050565b5f6020820190508181035f830152610b1d8184610aa9565b905092915050565b5f67ffffffffffffffff821115610b3f57610b3e610786565b5b602082029050602081019050919050565b5f610b62610b5d84610b25565b6107e4565b90508083825260208201905060208402830185811115610b8557610b846108d5565b5b835b81811015610bcc57803567ffffffffffffffff811115610baa57610ba961076e565b5b808601610bb7898261087d565b85526020850194505050602081019050610b87565b5050509392505050565b5f82601f830112610bea57610be961076e565b5b8135610bfa848260208601610b50565b91505092915050565b5f805f60608486031215610c1a57610c19610766565b5b5f84013567ffffffffffffffff811115610c3757610c3661076a565b5b610c4386828701610bd6565b935050602084013567ffffffffffffffff811115610c6457610c6361076a565b5b610c7086828701610974565b925050604084013567ffffffffffffffff811115610c9157610c9061076a565b5b610c9d86828701610974565b9150509250925092565b5f67ffffffffffffffff821115610cc157610cc0610786565b5b602082029050602081019050919050565b610cdb81610a6e565b8114610ce5575f80fd5b50565b5f81359050610cf681610cd2565b92915050565b5f610d0e610d0984610ca7565b6107e4565b90508083825260208201905060208402830185811115610d3157610d306108d5565b5b835b81811015610d5a5780610d468882610ce8565b845260208401935050602081019050610d33565b5050509392505050565b5f82601f830112610d7857610d7761076e565b5b8135610d88848260208601610cfc565b91505092915050565b5f60208284031215610da657610da5610766565b5b5f82013567ffffffffffffffff811115610dc357610dc261076a565b5b610dcf84828501610d64565b91505092915050565b5f81519050919050565b5f82825260208201905092915050565b5f819050602082019050919050565b610e0a816108d9565b82525050565b5f610e1b8383610e01565b60208301905092915050565b5f602082019050919050565b5f610e3d82610dd8565b610e478185610de2565b9350610e5283610df2565b805f5b83811015610e82578151610e698882610e10565b9750610e7483610e27565b925050600181019050610e55565b5085935050505092915050565b5f6020820190508181035f830152610ea78184610e33565b905092915050565b5f67ffffffffffffffff821115610ec957610ec8610786565b5b602082029050602081019050919050565b5f8160070b9050919050565b610eef81610eda565b8114610ef9575f80fd5b50565b5f81359050610f0a81610ee6565b92915050565b5f610f22610f1d84610eaf565b6107e4565b90508083825260208201905060208402830185811115610f4557610f446108d5565b5b835b81811015610f6e5780610f5a8882610efc565b845260208401935050602081019050610f47565b5050509392505050565b5f82601f830112610f8c57610f8b61076e565b5b8135610f9c848260208601610f10565b91505092915050565b5f60208284031215610fba57610fb9610766565b5b5f82013567ffffffffffffffff811115610fd757610fd661076a565b5b610fe384828501610f78565b91505092915050565b5f81519050610ffa81610cd2565b92915050565b5f61101261100d84610ca7565b6107e4565b90508083825260208201905060208402830185811115611035576110346108d5565b5b835b8181101561105e578061104a8882610fec565b845260208401935050602081019050611037565b5050509392505050565b5f82601f83011261107c5761107b61076e565b5b815161108c848260208601611000565b91505092915050565b5f602082840312156110aa576110a9610766565b5b5f82015167ffffffffffffffff8111156110c7576110c661076a565b5b6110d384828501611068565b91505092915050565b7f4e487b71000000000000000000000000000000000000000000000000000000005f52603260045260245ffd5b7f4e487b71000000000000000000000000000000000000000000000000000000005f52601160045260245ffd5b5f61114082610a6e565b91507f8000000000000000000000000000000000000000000000000000000000000000820361117257611171611109565b5b815f039050919050565b5f8160011c9050919050565b5f808291508390505b60018511156111d1578086048111156111ad576111ac611109565b5b60018516156111bc5780820291505b80810290506111ca8561117c565b9450611191565b94509492505050565b5f826111e957600190506112a4565b816111f6575f90506112a4565b816001811461120c576002811461121657611245565b60019150506112a4565b60ff84111561122857611227611109565b5b8360020a91508482111561123f5761123e611109565b5b506112a4565b5060208310610133831016604e8410600b841016171561127a5782820a90508381111561127557611274611109565b5b6112a4565b6112878484846001611188565b9250905081840481111561129e5761129d611109565b5b81810290505b9392505050565b5f6112b5826108d9565b91506112c0836108d9565b92506112ed7fffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffff84846111da565b905092915050565b7f4e487b71000000000000000000000000000000000000000000000000000000005f52601260045260245ffd5b5f61132c826108d9565b9150611337836108d9565b9250828202611345816108d9565b9150828204841483151761135c5761135b611109565b5b5092915050565b5f61136d826108d9565b9150611378836108d9565b92508282019050808211156113905761138f611109565b5b92915050565b5f6113a0826108d9565b91507fffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffff82036113d2576113d1611109565b5b600182019050919050565b5f602082840312156113f2576113f1610766565b5b5f6113ff84828501610fec565b91505092915050565b5f82825260208201905092915050565b7f496e76616c6964206669656c6420656c656d656e7400000000000000000000005f82015250565b5f61144c601583611408565b915061145782611418565b602082019050919050565b5f6020820190508181035f83015261147981611440565b9050919050565b5f61148a82610a6e565b915061149583610a6e565b92508282019050828112155f8312168382125f8412151617156114bb576114ba611109565b5b92915050565b5f6114cb826108d9565b91506114d6836108d9565b9250826114e6576114e56112f5565b5b828206905092915050565b7f4d6174683a206d756c446976206f766572666c6f7700000000000000000000005f82015250565b5f611525601583611408565b9150611530826114f1565b602082019050919050565b5f6020820190508181035f83015261155281611519565b905091905056fea264697066735822122042fb029acf1fa25b69b3dee11c2d12801f675cb77aed3817f5fa3c27df065f1364736f6c63430008140033")]
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
            uint256[] memory decimals,
            uint256[] memory scales
        ) external pure returns (int256[] memory quantized_data) {
            int[] memory _data = abi.decode(data, (int256[]));
            quantized_data = new int256[](_data.length);
            for (uint i; i < _data.length; i++) {
                int x = _data[i];
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
            // uint256 [] _decimals,
            DynSeqToken(
                decimals
                    .iter()
                    .map(|i| WordToken(B256::from(*i)))
                    .collect_vec(),
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
pub async fn test_on_chain_data_multi<M: 'static + Provider<Http<Client>, Ethereum>>(
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

/// Tests on-chain data storage by deploying a contract that stores the network input and or output
/// data in its storage. It does this by converting the floating point values to integers and storing the
/// the number of decimals of the floating point value on chain.
pub async fn test_on_chain_data_single<M: 'static + Provider<Http<Client>, Ethereum>>(
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
        address: hex::encode(contract.address().0 .0),
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
pub async fn read_on_chain_inputs_single<M: 'static + Provider<Http<Client>, Ethereum>>(
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
        .quantize_data_single(fetched_inputs, decimals, scales)
        .call()
        .await?
        .quantized_data;

    debug!("evm quantization results: {:#?}", results);

    let mut felts: Vec<Fr> = vec![];

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
    commitment_bytes: Option<Vec<u8>>,
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
    Ok(contract)
}
