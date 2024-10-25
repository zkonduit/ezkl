// SPDX-License-Identifier: MIT
pragma solidity 0.8.20;

// lib/openzeppelin-contracts/contracts/utils/Context.sol

// OpenZeppelin Contracts (last updated v5.0.1) (utils/Context.sol)

/**
 * @dev Provides information about the current execution context, including the
 * sender of the transaction and its data. While these are generally available
 * via msg.sender and msg.data, they should not be accessed in such a direct
 * manner, since when dealing with meta-transactions the account sending and
 * paying for execution may not be the actual sender (as far as an application
 * is concerned).
 *
 * This contract is only required for intermediate, library-like contracts.
 */
abstract contract Context {
    function _msgSender() internal view virtual returns (address) {
        return msg.sender;
    }

    function _msgData() internal view virtual returns (bytes calldata) {
        return msg.data;
    }

    function _contextSuffixLength() internal view virtual returns (uint256) {
        return 0;
    }
}

// lib/openzeppelin-contracts/contracts/access/Ownable.sol

// OpenZeppelin Contracts (last updated v5.0.0) (access/Ownable.sol)

/**
 * @dev Contract module which provides a basic access control mechanism, where
 * there is an account (an owner) that can be granted exclusive access to
 * specific functions.
 *
 * The initial owner is set to the address provided by the deployer. This can
 * later be changed with {transferOwnership}.
 *
 * This module is used through inheritance. It will make available the modifier
 * `onlyOwner`, which can be applied to your functions to restrict their use to
 * the owner.
 */
abstract contract Ownable is Context {
    /// set the owener initialy to be the anvil test account
    address private _owner = 0xf39Fd6e51aad88F6F4ce6aB8827279cffFb92266;

    /**
     * @dev The caller account is not authorized to perform an operation.
     */
    error OwnableUnauthorizedAccount(address account);

    /**
     * @dev The owner is not a valid owner account. (eg. `address(0)`)
     */
    error OwnableInvalidOwner(address owner);

    event OwnershipTransferred(
        address indexed previousOwner,
        address indexed newOwner
    );

    /**
     * @dev Initializes the contract setting the address provided by the deployer as the initial owner.
     */
    constructor() {
        _transferOwnership(msg.sender);
    }

    /**
     * @dev Throws if called by any account other than the owner.
     */
    modifier onlyOwner() {
        _checkOwner();
        _;
    }

    /**
     * @dev Returns the address of the current owner.
     */
    function owner() public view virtual returns (address) {
        return _owner;
    }

    /**
     * @dev Throws if the sender is not the owner.
     */
    function _checkOwner() internal view virtual {
        if (owner() != _msgSender()) {
            revert OwnableUnauthorizedAccount(_msgSender());
        }
    }

    /**
     * @dev Leaves the contract without owner. It will not be possible to call
     * `onlyOwner` functions. Can only be called by the current owner.
     *
     * NOTE: Renouncing ownership will leave the contract without an owner,
     * thereby disabling any functionality that is only available to the owner.
     */
    function renounceOwnership() public virtual onlyOwner {
        _transferOwnership(address(0));
    }

    /**
     * @dev Transfers ownership of the contract to a new account (`newOwner`).
     * Can only be called by the current owner.
     */
    function transferOwnership(address newOwner) public virtual onlyOwner {
        if (newOwner == address(0)) {
            revert OwnableInvalidOwner(address(0));
        }
        _transferOwnership(newOwner);
    }

    /**
     * @dev Transfers ownership of the contract to a new account (`newOwner`).
     * Internal function without access restriction.
     */
    function _transferOwnership(address newOwner) internal virtual {
        address oldOwner = _owner;
        _owner = newOwner;
        emit OwnershipTransferred(oldOwner, newOwner);
    }
}

// interface for the reusable verifier.
interface Halo2VerifierReusable {
    function verifyProof(
        address vkArtifact,
        bytes calldata proof,
        uint256[] calldata instances
    ) external returns (bool);
}

// Manages the deployment of all EZKL reusbale verifiers (ezkl version specific), verifiying key artifacts (circuit specific) and
// routing proof verifications to the correct VKA and associate reusable verifier.
// Helps to prevent the deployment of duplicate verifiers.
contract EZKLVerifierManager is Ownable {
    /// @dev Mapping that checks if a given reusable verifier has been deployed
    mapping(address => bool) public verifierAddresses;

    event DeployedVerifier(address addr);

    // 1. Compute the address of the verifier to be deployed
    function precomputeAddress(
        bytes memory bytecode
    ) public view returns (address) {
        bytes32 hash = keccak256(
            abi.encodePacked(
                bytes1(0xff),
                address(this),
                uint(0),
                keccak256(bytecode)
            )
        );

        return address(uint160(uint(hash)));
    }

    // 2. Deploy the reusable verifier using create2
    /// @param bytecode The bytecode of the reusable verifier to deploy
    function deployVerifier(
        bytes memory bytecode
    ) public returns (address addr) {
        assembly {
            addr := create2(
                0x0, // value, hardcode to 0
                add(bytecode, 0x20),
                mload(bytecode),
                0x0 // salt, hardcode to 0
            )
            if iszero(extcodesize(addr)) {
                revert(0, 0)
            }
        }
        verifierAddresses[addr] = true;
        emit DeployedVerifier(addr);
    }
}
