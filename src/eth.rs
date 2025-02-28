use crate::graph::input::{CallToAccount, CallsToAccount, FileSourceInner, GraphData};
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
    DataAttestationSingle,
    "./abis/DataAttestationSingle.json"
);
abigen!(
    #[allow(missing_docs)]
    #[sol(rpc, bytecode="608060405234801561000f575f80fd5b50611a3c8061001d5f395ff3fe608060405234801561000f575f80fd5b506004361061004a575f3560e01c80631abe6c131461004e57806345ab50981461007e5780639e564bbc146100ae578063b404abab146100de575b5f80fd5b61006860048036038101906100639190610bcd565b61010e565b6040516100759190610cf9565b60405180910390f35b61009860048036038101906100939190610dc9565b6102cb565b6040516100a59190610f24565b60405180910390f35b6100c860048036038101906100c39190611022565b61065a565b6040516100d59190610cf9565b60405180910390f35b6100f860048036038101906100f391906111b0565b6107e0565b6040516101059190610f24565b60405180910390f35b6060835167ffffffffffffffff81111561012b5761012a610a5e565b5b6040519080825280602002602001820160405280156101595781602001602082028036833780820191505090505b5090505f83600a61016a9190611353565b90505f836001901b90505f5b83518110156102c1575f806f7fffffffffffffffffffffffffffffff6fffffffffffffffffffffffffffffffff168984815181106101b7576101b661139d565b5b60200260200101511115610212578883815181106101d8576101d761139d565b5b60200260200101517f30644e72e131a029b85045b68181585d2833e84879b9709143e1f593f000000161020b91906113ca565b9150610234565b8883815181106102255761022461139d565b5b60200260200101519150600190505b5f61024083878761092f565b90508460028680610254576102536113fd565b5b888609610261919061142a565b1061027657600181610273919061146b565b90505b81610281578061028c565b8061028b9061149e565b5b87858151811061029f5761029e61139d565b5b60200260200101818152505050505080806102b9906114e4565b915050610176565b5050509392505050565b60605f848060200190518101906102e291906115d4565b9050825167ffffffffffffffff8111156102ff576102fe610a5e565b5b60405190808252806020026020018201604052801561032d5781602001602082028036833780820191505090505b5091508351835114610374576040517f08c379a000000000000000000000000000000000000000000000000000000000815260040161036b9061169b565b60405180910390fd5b8251815110156103b9576040517f08c379a00000000000000000000000000000000000000000000000000000000081526004016103b090611729565b60405180910390fd5b5f5b8251811015610651575f8282815181106103d8576103d761139d565b5b602002602001015190505f808212905080156103fb57816103f89061149e565b91505b5f87848151811061040f5761040e61139d565b5b6020026020010151600a6104239190611353565b90505f8785815181106104395761043861139d565b5b60200260200101516001901b90505f61045385838561092f565b90508260028480610467576104666113fd565b5b848809610474919061142a565b1061048957600181610486919061146b565b90505b60017f30644e72e131a029b85045b68181585d2833e84879b9709143e1f593f0000001901c8111156104f0576040517f08c379a00000000000000000000000000000000000000000000000000000000081526004016104e790611791565b60405180910390fd5b83156105b4577fffffffffffffffffffffffffffffffff80000000000000000000000000000000600f0b816105249061149e565b13610564576040517f08c379a000000000000000000000000000000000000000000000000000000000815260040161055b9061181f565b60405180910390fd5b807f30644e72e131a029b85045b68181585d2833e84879b9709143e1f593f0000001610590919061183d565b8887815181106105a3576105a261139d565b5b602002602001018181525050610639565b6f7fffffffffffffffffffffffffffffff6fffffffffffffffffffffffffffffffff168110610618576040517f08c379a000000000000000000000000000000000000000000000000000000000815260040161060f906118ed565b60405180910390fd5b8088878151811061062c5761062b61139d565b5b6020026020010181815250505b50505050508080610649906114e4565b9150506103bb565b50509392505050565b6060835167ffffffffffffffff81111561067757610676610a5e565b5b6040519080825280602002602001820160405280156106a55781602001602082028036833780820191505090505b5090505f5b84518110156107d8575f8582815181106106c7576106c661139d565b5b60200260200101518060200190518101906106e2919061190b565b90505f808212905080156106fd57816106fa9061149e565b91505b5f8684815181106107115761071061139d565b5b6020026020010151600a6107259190611353565b90505f86858151811061073b5761073a61139d565b5b60200260200101516001901b90505f61075585838561092f565b90508260028480610769576107686113fd565b5b848809610776919061142a565b1061078b57600181610788919061146b565b90505b8361079657806107a1565b806107a09061149e565b5b8787815181106107b4576107b361139d565b5b602002602001018181525050505050505080806107d0906114e4565b9150506106aa565b509392505050565b6060815167ffffffffffffffff8111156107fd576107fc610a5e565b5b60405190808252806020026020018201604052801561082b5781602001602082028036833780820191505090505b5090505f5b8251811015610929575f83828151811061084d5761084c61139d565b5b60200260200101511215801561089d57507f30644e72e131a029b85045b68181585d2833e84879b9709143e1f593f00000018382815181106108925761089161139d565b5b602002602001015111155b6108dc576040517f08c379a00000000000000000000000000000000000000000000000000000000081526004016108d390611980565b60405180910390fd5b8281815181106108ef576108ee61139d565b5b602002602001015182828151811061090a5761090961139d565b5b6020026020010181815250508080610921906114e4565b915050610830565b50919050565b5f805f80198587098587029250828110838203039150505f81036109675783828161095d5761095c6113fd565b5b0492505050610a32565b8084116109a9576040517f08c379a00000000000000000000000000000000000000000000000000000000081526004016109a0906119e8565b60405180910390fd5b5f8486880990508281118203915080830392505f60018619018616905080860495508084049350600181825f0304019050808302841793505f600287600302189050808702600203810290508087026002038102905080870260020381029050808702600203810290508087026002038102905080870260020381029050808502955050505050505b9392505050565b5f604051905090565b5f80fd5b5f80fd5b5f80fd5b5f601f19601f8301169050919050565b7f4e487b71000000000000000000000000000000000000000000000000000000005f52604160045260245ffd5b610a9482610a4e565b810181811067ffffffffffffffff82111715610ab357610ab2610a5e565b5b80604052505050565b5f610ac5610a39565b9050610ad18282610a8b565b919050565b5f67ffffffffffffffff821115610af057610aef610a5e565b5b602082029050602081019050919050565b5f80fd5b5f819050919050565b610b1781610b05565b8114610b21575f80fd5b50565b5f81359050610b3281610b0e565b92915050565b5f610b4a610b4584610ad6565b610abc565b90508083825260208201905060208402830185811115610b6d57610b6c610b01565b5b835b81811015610b965780610b828882610b24565b845260208401935050602081019050610b6f565b5050509392505050565b5f82601f830112610bb457610bb3610a4a565b5b8135610bc4848260208601610b38565b91505092915050565b5f805f60608486031215610be457610be3610a42565b5b5f84013567ffffffffffffffff811115610c0157610c00610a46565b5b610c0d86828701610ba0565b9350506020610c1e86828701610b24565b9250506040610c2f86828701610b24565b9150509250925092565b5f81519050919050565b5f82825260208201905092915050565b5f819050602082019050919050565b5f819050919050565b610c7481610c62565b82525050565b5f610c858383610c6b565b60208301905092915050565b5f602082019050919050565b5f610ca782610c39565b610cb18185610c43565b9350610cbc83610c53565b805f5b83811015610cec578151610cd38882610c7a565b9750610cde83610c91565b925050600181019050610cbf565b5085935050505092915050565b5f6020820190508181035f830152610d118184610c9d565b905092915050565b5f80fd5b5f67ffffffffffffffff821115610d3757610d36610a5e565b5b610d4082610a4e565b9050602081019050919050565b828183375f83830152505050565b5f610d6d610d6884610d1d565b610abc565b905082815260208101848484011115610d8957610d88610d19565b5b610d94848285610d4d565b509392505050565b5f82601f830112610db057610daf610a4a565b5b8135610dc0848260208601610d5b565b91505092915050565b5f805f60608486031215610de057610ddf610a42565b5b5f84013567ffffffffffffffff811115610dfd57610dfc610a46565b5b610e0986828701610d9c565b935050602084013567ffffffffffffffff811115610e2a57610e29610a46565b5b610e3686828701610ba0565b925050604084013567ffffffffffffffff811115610e5757610e56610a46565b5b610e6386828701610ba0565b9150509250925092565b5f81519050919050565b5f82825260208201905092915050565b5f819050602082019050919050565b610e9f81610b05565b82525050565b5f610eb08383610e96565b60208301905092915050565b5f602082019050919050565b5f610ed282610e6d565b610edc8185610e77565b9350610ee783610e87565b805f5b83811015610f17578151610efe8882610ea5565b9750610f0983610ebc565b925050600181019050610eea565b5085935050505092915050565b5f6020820190508181035f830152610f3c8184610ec8565b905092915050565b5f67ffffffffffffffff821115610f5e57610f5d610a5e565b5b602082029050602081019050919050565b5f610f81610f7c84610f44565b610abc565b90508083825260208201905060208402830185811115610fa457610fa3610b01565b5b835b81811015610feb57803567ffffffffffffffff811115610fc957610fc8610a4a565b5b808601610fd68982610d9c565b85526020850194505050602081019050610fa6565b5050509392505050565b5f82601f83011261100957611008610a4a565b5b8135611019848260208601610f6f565b91505092915050565b5f805f6060848603121561103957611038610a42565b5b5f84013567ffffffffffffffff81111561105657611055610a46565b5b61106286828701610ff5565b935050602084013567ffffffffffffffff81111561108357611082610a46565b5b61108f86828701610ba0565b925050604084013567ffffffffffffffff8111156110b0576110af610a46565b5b6110bc86828701610ba0565b9150509250925092565b5f67ffffffffffffffff8211156110e0576110df610a5e565b5b602082029050602081019050919050565b6110fa81610c62565b8114611104575f80fd5b50565b5f81359050611115816110f1565b92915050565b5f61112d611128846110c6565b610abc565b905080838252602082019050602084028301858111156111505761114f610b01565b5b835b8181101561117957806111658882611107565b845260208401935050602081019050611152565b5050509392505050565b5f82601f83011261119757611196610a4a565b5b81356111a784826020860161111b565b91505092915050565b5f602082840312156111c5576111c4610a42565b5b5f82013567ffffffffffffffff8111156111e2576111e1610a46565b5b6111ee84828501611183565b91505092915050565b7f4e487b71000000000000000000000000000000000000000000000000000000005f52601160045260245ffd5b5f8160011c9050919050565b5f808291508390505b600185111561127957808604811115611255576112546111f7565b5b60018516156112645780820291505b808102905061127285611224565b9450611239565b94509492505050565b5f82611291576001905061134c565b8161129e575f905061134c565b81600181146112b457600281146112be576112ed565b600191505061134c565b60ff8411156112d0576112cf6111f7565b5b8360020a9150848211156112e7576112e66111f7565b5b5061134c565b5060208310610133831016604e8410600b84101617156113225782820a90508381111561131d5761131c6111f7565b5b61134c565b61132f8484846001611230565b92509050818404811115611346576113456111f7565b5b81810290505b9392505050565b5f61135d82610b05565b915061136883610b05565b92506113957fffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffff8484611282565b905092915050565b7f4e487b71000000000000000000000000000000000000000000000000000000005f52603260045260245ffd5b5f6113d482610b05565b91506113df83610b05565b92508282039050818111156113f7576113f66111f7565b5b92915050565b7f4e487b71000000000000000000000000000000000000000000000000000000005f52601260045260245ffd5b5f61143482610b05565b915061143f83610b05565b925082820261144d81610b05565b91508282048414831517611464576114636111f7565b5b5092915050565b5f61147582610b05565b915061148083610b05565b9250828201905080821115611498576114976111f7565b5b92915050565b5f6114a882610c62565b91507f800000000000000000000000000000000000000000000000000000000000000082036114da576114d96111f7565b5b815f039050919050565b5f6114ee82610b05565b91507fffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffff82036115205761151f6111f7565b5b600182019050919050565b5f81519050611539816110f1565b92915050565b5f61155161154c846110c6565b610abc565b9050808382526020820190506020840283018581111561157457611573610b01565b5b835b8181101561159d5780611589888261152b565b845260208401935050602081019050611576565b5050509392505050565b5f82601f8301126115bb576115ba610a4a565b5b81516115cb84826020860161153f565b91505092915050565b5f602082840312156115e9576115e8610a42565b5b5f82015167ffffffffffffffff81111561160657611605610a46565b5b611612848285016115a7565b91505092915050565b5f82825260208201905092915050565b7f7363616c657320616e6420646563696d616c73206d757374206265206f6620745f8201527f68652073616d65206c656e677468000000000000000000000000000000000000602082015250565b5f611685602e8361161b565b91506116908261162b565b604082019050919050565b5f6020820190508181035f8301526116b281611679565b9050919050565b7f64617461206c656e677468206d7573742062652067726561746572207468616e5f8201527f206f7220657175616c20746f207363616c6573206c656e677468000000000000602082015250565b5f611713603a8361161b565b915061171e826116b9565b604082019050919050565b5f6020820190508181035f83015261174081611707565b9050919050565b7f4f766572666c6f77206669656c64206d6f64756c7573000000000000000000005f82015250565b5f61177b60168361161b565b915061178682611747565b602082019050919050565b5f6020820190508181035f8301526117a88161176f565b9050919050565b7f5175616e74697a65642076616c7565206973206c657373207468616e20696e745f8201527f313238206d696e00000000000000000000000000000000000000000000000000602082015250565b5f61180960278361161b565b9150611814826117af565b604082019050919050565b5f6020820190508181035f830152611836816117fd565b9050919050565b5f61184782610c62565b915061185283610c62565b925082820390508181125f8412168282135f851215161715611877576118766111f7565b5b92915050565b7f5175616e74697a65642076616c75652069732067726561746572207468616e205f8201527f696e74313238206d617800000000000000000000000000000000000000000000602082015250565b5f6118d7602a8361161b565b91506118e28261187d565b604082019050919050565b5f6020820190508181035f830152611904816118cb565b9050919050565b5f602082840312156119205761191f610a42565b5b5f61192d8482850161152b565b91505092915050565b7f496e76616c6964206669656c6420656c656d656e7400000000000000000000005f82015250565b5f61196a60158361161b565b915061197582611936565b602082019050919050565b5f6020820190508181035f8301526119978161195e565b9050919050565b7f4d6174683a206d756c446976206f766572666c6f7700000000000000000000005f82015250565b5f6119d260158361161b565b91506119dd8261199e565b602082019050919050565b5f6020820190508181035f8301526119ff816119c6565b905091905056fea26469706673582212208259099aee98f1ec0166e5ed183b5d46b040528ab848225379f031417c7c50e364736f6c63430008140033")]
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
            // assert that scales and decimals are of the same length
            require(scales.length == decimals.length, "scales and decimals must be of the same length");
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
                    quantized_data[i] = uint256(int(ORDER) - int256(output));
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
        get_contract_artifacts(sol_code_path, "DataAttestationSingle", runs).await?;
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
    for input in data.iter().flatten() {
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
                "contract DataAttestationSingle {",
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
