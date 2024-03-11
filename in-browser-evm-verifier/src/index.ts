import { defaultAbiCoder as AbiCoder } from '@ethersproject/abi'
import { Address, hexToBytes } from '@ethereumjs/util'
import { Chain, Common, Hardfork } from '@ethereumjs/common'
import { LegacyTransaction, LegacyTxData } from '@ethereumjs/tx'
// import { DefaultStateManager } from '@ethereumjs/statemanager'
// import { Blockchain } from '@ethereumjs/blockchain'
import { VM } from '@ethereumjs/vm'
import { EVM } from '@ethereumjs/evm'
import { buildTransaction, encodeDeployment } from './utils/tx-builder'
import { getAccountNonce, insertAccount } from './utils/account-utils'
import { encodeVerifierCalldata } from '../nodejs/ezkl';

async function deployContract(
  vm: VM,
  common: Common,
  senderPrivateKey: Uint8Array,
  deploymentBytecode: string,
): Promise<Address> {
  // Contracts are deployed by sending their deployment bytecode to the address 0
  // The contract params should be abi-encoded and appended to the deployment bytecode.
  // const data =
  const data = encodeDeployment(deploymentBytecode)
  const txData = {
    data,
    nonce: await getAccountNonce(vm, senderPrivateKey),
  }

  const tx = LegacyTransaction.fromTxData(
    buildTransaction(txData) as LegacyTxData,
    { common, allowUnlimitedInitCodeSize: true },
  ).sign(senderPrivateKey)

  const deploymentResult = await vm.runTx({
    tx,
    skipBlockGasLimitValidation: true,
  })

  if (deploymentResult.execResult.exceptionError) {
    throw deploymentResult.execResult.exceptionError
  }

  return deploymentResult.createdAddress!
}

async function verify(
  vm: VM,
  contractAddress: Address,
  caller: Address,
  proof: Uint8Array | Uint8ClampedArray,
): Promise<boolean> {
  if (proof instanceof Uint8Array) {
    proof = new Uint8ClampedArray(proof.buffer)
  }
  const data = encodeVerifierCalldata(proof)

  const verifyResult = await vm.evm.runCall({
    to: contractAddress,
    caller: caller,
    origin: caller, // The tx.origin is also the caller here
    data: data,
  })

  if (verifyResult.execResult.exceptionError) {
    throw verifyResult.execResult.exceptionError
  }

  const results = AbiCoder.decode(['bool'], verifyResult.execResult.returnValue)

  return results[0]
}

/**
 * Spins up an ephemeral EVM instance for executing the bytecode of a solidity verifier
 * @param proof Json serialized proof file
 * @param bytecode The bytecode of a compiled solidity verifier.
 * @param evmVersion The evm version to use for the verification. (Default: London)
 * @returns The result of the evm verification.
 * @throws If the verify transaction reverts
 */
export default async function localEVMVerify(
  proof: Uint8Array | Uint8ClampedArray,
  bytecode: string,
  evmVersion?: Hardfork,
): Promise<boolean> {
  try {
    const hardfork = evmVersion ? evmVersion : Hardfork['London']
    const common = new Common({ chain: Chain.Mainnet, hardfork })
    const accountPk = hexToBytes(
      '0xe331b6d69882b4cb4ea581d88e0b604039a3de5967688d3dcffdd2270c0fd109',
    )

    const evm = new EVM({
      allowUnlimitedContractSize: true,
      allowUnlimitedInitCodeSize: true,
    })

    const vm = await VM.create({ common, evm })
    const accountAddress = Address.fromPrivateKey(accountPk)

    await insertAccount(vm, accountAddress)

    const contractAddress = await deployContract(
      vm,
      common,
      accountPk,
      bytecode,
    )

    const result = await verify(vm, contractAddress, accountAddress, proof)

    return result
  } catch (error) {
    // log or re-throw the error, depending on your needs
    console.error('An error occurred:', error)
    throw error
  }
}

export { Hardfork }
