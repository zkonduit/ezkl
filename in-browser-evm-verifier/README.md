# inbrowser-evm-verify

We would like the Solidity verifier to be canonical and usually all you ever need. For this, we need to be able to run that verifier in browser.

## How to use (Node js)

```ts 
import localEVMVerify from '@ezkljs/verify';

// Load in the proof file as a buffer
const proofFileBuffer = fs.readFileSync(`${path}/${example}/proof.pf`)

// Stringified EZKL evm verifier bytecode (this is just an example don't use in production)
const bytecode = '0x608060405234801561001057600080fd5b5060d38061001f6000396000f3fe608060405234801561001057600080fd5b50600436106100415760003560e01c8063cfae321714610046575b600080fd5b6100496100f1565b60405161005691906100f1565b60405180910390f35b'

const result = await localEVMVerify(proofFileBuffer, bytecode)

console.log('result', result)
```

**Note**: Run `ezkl create-evm-verifier` to get the Solidity verifier, with which you can retrieve the bytecode once compiled. We recommend compiling to the London hardfork target, else you will have to pass an additional parameter specifying the EVM version to the `localEVMVerify` function like so (for Paris hardfork):

```ts
import localEVMVerify, { hardfork } from '@ezkljs/verify';

const result = await localEVMVerify(proofFileBuffer, bytecode, hardfork['Paris'])
```

## How to use (Browser)

```ts
import localEVMVerify from '@ezkljs/verify';

// Load in the proof file as a buffer using the web apis (fetch, FileReader, etc)
// We use fetch in this example to load the proof file as a buffer
const proofFileBuffer = await fetch(`${path}/${example}/proof.pf`).then(res => res.arrayBuffer())

// Stringified EZKL evm verifier bytecode (this is just an example don't use in production)
const bytecode = '0x608060405234801561001057600080fd5b5060d38061001f6000396000f3fe608060405234801561001057600080fd5b50600436106100415760003560e01c8063cfae321714610046575b600080fd5b6100496100f1565b60405161005691906100f1565b60405180910390f35b'

const result = await browserEVMVerify(proofFileBuffer, bytecode)

console.log('result', result)
```

Output:

```ts  
result: true
```


