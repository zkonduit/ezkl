import { Interface, defaultAbiCoder as AbiCoder } from '@ethersproject/abi'
import {
  AccessListEIP2930TxData,
  FeeMarketEIP1559TxData,
  TxData,
} from '@ethereumjs/tx'

type TransactionsData =
  | TxData
  | AccessListEIP2930TxData
  | FeeMarketEIP1559TxData

export const encodeFunction = (
  method: string,
  params?: {
    // eslint-disable-next-line @typescript-eslint/no-explicit-any
    types: any[]
    values: unknown[]
  },
): string => {
  const parameters = params?.types ?? []
  const methodWithParameters = `function ${method}(${parameters.join(',')})`
  const signatureHash = new Interface([methodWithParameters]).getSighash(method)
  const encodedArgs = AbiCoder.encode(parameters, params?.values ?? [])

  return signatureHash + encodedArgs.slice(2)
}

export const encodeDeployment = (
  bytecode: string,
  params?: {
    // eslint-disable-next-line @typescript-eslint/no-explicit-any
    types: any[]
    values: unknown[]
  },
) => {
  const deploymentData = '0x' + bytecode
  if (params) {
    const argumentsEncoded = AbiCoder.encode(params.types, params.values)
    return deploymentData + argumentsEncoded.slice(2)
  }
  return deploymentData
}

export const buildTransaction = (
  data: Partial<TransactionsData>,
): TransactionsData => {
  const defaultData: Partial<TransactionsData> = {
    gasLimit: 3_000_000_000_000_000,
    gasPrice: 7,
    value: 0,
    data: '0x',
  }

  return {
    ...defaultData,
    ...data,
  }
}
