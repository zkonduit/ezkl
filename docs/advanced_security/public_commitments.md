## EZKL Security Note: Public Commitments and Low-Entropy Data

> **Disclaimer:** this a more technical post that requires some prior knowledge of how ZK proving systems like Halo2 operate, and in particular in how these APIs are constructed. For background reading we highly recommend the [Halo2 book](https://zcash.github.io/halo2/) and [Halo2 Club](https://halo2.club/).

## Overview of commitments in EZKL

A common design pattern in a zero knowledge (zk) application is thus:
- A prover has some data which is used within a circuit.
- This data, as it may be high-dimensional or somewhat private, is pre-committed to using some hash function.
- The zk-circuit which forms the core of the application then proves (para-phrasing) a statement of the form:
>"I know some data D which when hashed corresponds to the pre-committed to value H + whatever else the circuit is proving over D". 

From our own experience, we've implemented such patterns using snark-friendly hash functions like [Poseidon](https://www.poseidon-hash.info/), for which there is a relatively well vetted [implementation](https://docs.rs/halo2_gadgets/latest/halo2_gadgets/poseidon/index.html) in Halo2. Even then these hash functions can introduce lots of overhead and can be very expensive to generate proofs for if the dimensionality of the data D is large. 

You can also implement such a pattern using Halo2's `Fixed` columns _if the privacy preservation of the pre-image is not necessary_. These are Halo2 columns (i.e in reality just polynomials) that are left unblinded (unlike the blinded `Advice` columns), and whose commitments are shared with the verifier by way of the verifying key for the application's zk-circuit. These commitments are much lower cost to generate than implementing a hashing function, such as Poseidon, within a circuit.

> **Note:** Blinding is the process whereby a certain set of the final elements (i.e rows) of a Halo2 column are set to random field elements. This is the mechanism by which Halo2 achieves its zero knowledge properties for `Advice` columns. By contrast `Fixed` columns aren't zero-knowledge in that they are vulnerable to dictionary attacks in the same manner a hash function is. Given some set of known or popular data D an attacker can attempt to recover the pre-image of a hash by running D through the hash function to see if the outputs match a public commitment. These attacks aren't "possible" on blinded `Advice` columns.

> **Further Note:** Note that without blinding, with access to `M` proofs, each of which contains an evaluation of the polynomial at a different point, an attacker can more easily recover a non blinded column's pre-image. This is because each proof generates a new query and evaluation of the polynomial represented by the column and as such with repetition a clearer picture can emerge of the column's pre-image. Thus unblinded columns should only be used for privacy preservation, in the manner of a hash, if the number of proofs generated against a fixed set of values is limited. More formally if M independent and _unique_ queries are generated; if M is equal to the degree + 1 of the polynomial represented by the column (i.e the unique lagrange interpolation of the values in the columns), then the column's pre-image can be recovered. As such as the logrows K increases, the more queries are required to recover the pre-image (as 2^K unique queries are required). This assumes that the entries in the column are not structured, as if they are then the number of queries required to recover the pre-image is reduced (eg. if all rows above a certain point are known to be nil).

The annoyance in using `Fixed` columns comes from the fact that they require generating a new verifying key every time a new set of commitments is generated.

> **Example:** Say for instance an application leverages a zero-knowledge circuit to prove the correct execution of a neural network. Every week the neural network is finetuned or retrained on new data. If the architecture remains the same then commiting to the new network parameters, along with a new proof of performance on a test set, would be an ideal setup. If we leverage `Fixed` columns to commit to the model parameters, each new commitment will require re-generating a  verifying key and sharing the new key with the verifier(s). This is not-ideal UX and can become expensive if the verifier is deployed on-chain. 

An ideal commitment would thus have the low cost of a `Fixed`  column but wouldn't require regenerating a new verifying key for each new commitment.

### Unblinded Advice Columns

A first step in designing such a commitment is to allow for optionally unblinded `Advice` columns within the Halo2 API. These won't be included in the verifying key, AND are blinded with a constant factor `1` -- such that if someone knows the pre-image to the commitment, they can recover it by running it through the corresponding polynomial commitment scheme (in ezkl's case [KZG commitments](https://dankradfeist.de/ethereum/2020/06/16/kate-polynomial-commitments.html)). 

This is implemented using the `polycommit` visibility parameter in the ezkl API.

## The Vulnerability of Public Commitments


Public commitments in EZKL (both Poseidon-hashed inputs and KZG commitments) can be vulnerable to brute-force attacks when input data has low entropy. A malicious actor could reveal committed data by searching through possible input values, compromising privacy in applications like anonymous credentials. This is particularly relevant when input data comes from known finite sets (e.g., names, dates).

Example Risk: In an anonymous credential system using EZKL for ID verification, an attacker could match hashed outputs against a database of common identifying information to deanonymize users.



