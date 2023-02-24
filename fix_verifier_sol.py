#!/usr/bin/env python3

import sys
import re

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: fix_verifier_sol.py <input>")
        sys.exit(1)

    input_file = sys.argv[1]
    lines = open(input_file).readlines()

    transcript_addrs = list()
    modified_lines = list()

    num_pubinputs = 0

    # convert calldataload 0x0 to 0x40 to read from pubInputs, and the rest
    # from proof
    calldata_pattern = r"^.*(calldataload\((0x[a-f0-9]+)\)).*$"
    mstore_pattern = r"^\s*(mstore\(0x([0-9a-fA-F]+)+),.+\)"
    mstore8_pattern = r"^\s*(mstore8\((\d+)+),.+\)"
    mstoren_pattern = r"^\s*(mstore\((\d+)+),.+\)"
    mload_pattern = r"(mload\((0x[0-9a-fA-F]+))\)"
    keccak_pattern = r"(keccak256\((0x[0-9a-fA-F]+))"
    modexp_pattern = r"(staticcall\(gas\(\), 0x5, (0x[0-9a-fA-F]+), 0xc0, (0x[0-9a-fA-F]+), 0x20)"
    ecmul_pattern = r"(staticcall\(gas\(\), 0x7, (0x[0-9a-fA-F]+), 0x60, (0x[0-9a-fA-F]+), 0x40)"
    ecadd_pattern = r"(staticcall\(gas\(\), 0x6, (0x[0-9a-fA-F]+), 0x80, (0x[0-9a-fA-F]+), 0x40)"
    ecpairing_pattern = r"(staticcall\(gas\(\), 0x8, (0x[0-9a-fA-F]+), 0x180, (0x[0-9a-fA-F]+), 0x20)"
    bool_pattern = r":bool"

    # Count the number of pub inputs
    start = None
    end = None
    i = 0
    for line in lines:
        if line.strip().startswith("mstore(0x20"):
            start = i

        if line.strip().startswith("mstore(0x0"):
            end = i
            break
        i += 1

    if start is None:
        num_pubinputs = 0
    else:
        num_pubinputs = end - start

    max_pubinputs_addr = 0
    if num_pubinputs > 0:
        max_pubinputs_addr = num_pubinputs * 32 - 32

    for line in lines:
        m = re.search(bool_pattern, line)
        if m:
            line = line.replace(":bool", "")

        m = re.search(calldata_pattern, line)
        if m:
            calldata_and_addr = m.group(1)
            addr = m.group(2)
            addr_as_num = int(addr, 16)

            if addr_as_num <= max_pubinputs_addr:
                proof_addr = hex(addr_as_num + 32)
                line = line.replace(calldata_and_addr, "mload(add(pubInputs, " + proof_addr + "))")
            else:
                proof_addr = hex(addr_as_num - max_pubinputs_addr)
                line = line.replace(calldata_and_addr, "mload(add(proof, " + proof_addr + "))")

        m = re.search(mstore8_pattern, line)
        if m:
            mstore = m.group(1)
            addr = m.group(2)
            addr_as_num = int(addr)
            transcript_addr = hex(addr_as_num)
            transcript_addrs.append(addr_as_num)
            line = line.replace(mstore, "mstore8(add(transcript, " + transcript_addr + ")")

        m = re.search(mstoren_pattern, line)
        if m:
            mstore = m.group(1)
            addr = m.group(2)
            addr_as_num = int(addr)
            transcript_addr = hex(addr_as_num)
            transcript_addrs.append(addr_as_num)
            line = line.replace(mstore, "mstore(add(transcript, " + transcript_addr + ")")

        m = re.search(modexp_pattern, line)
        if m:
            modexp = m.group(1)
            start_addr = m.group(2)
            result_addr = m.group(3)
            start_addr_as_num = int(start_addr, 16)
            result_addr_as_num = int(result_addr, 16)

            transcript_addr = hex(start_addr_as_num)
            transcript_addrs.append(addr_as_num)
            result_addr = hex(result_addr_as_num)
            line = line.replace(modexp, "staticcall(gas(), 0x5, add(transcript, " + transcript_addr + "), 0xc0, add(transcript, " + result_addr + "), 0x20")

        m = re.search(ecmul_pattern, line)
        if m:
            ecmul = m.group(1)
            start_addr = m.group(2)
            result_addr = m.group(3)
            start_addr_as_num = int(start_addr, 16)
            result_addr_as_num = int(result_addr, 16)

            transcript_addr = hex(start_addr_as_num)
            result_addr = hex(result_addr_as_num)
            transcript_addrs.append(start_addr_as_num)
            transcript_addrs.append(result_addr_as_num)
            line = line.replace(ecmul, "staticcall(gas(), 0x7, add(transcript, " + transcript_addr + "), 0x60, add(transcript, " + result_addr + "), 0x40")

        m = re.search(ecadd_pattern, line)
        if m:
            ecadd = m.group(1)
            start_addr = m.group(2)
            result_addr = m.group(3)
            start_addr_as_num = int(start_addr, 16)
            result_addr_as_num = int(result_addr, 16)

            transcript_addr = hex(start_addr_as_num)
            result_addr = hex(result_addr_as_num)
            transcript_addrs.append(start_addr_as_num)
            transcript_addrs.append(result_addr_as_num)
            line = line.replace(ecadd, "staticcall(gas(), 0x6, add(transcript, " + transcript_addr + "), 0x80, add(transcript, " + result_addr + "), 0x40")

        m = re.search(ecpairing_pattern, line)
        if m:
            ecpairing = m.group(1)
            start_addr = m.group(2)
            result_addr = m.group(3)
            start_addr_as_num = int(start_addr, 16)
            result_addr_as_num = int(result_addr, 16)

            transcript_addr = hex(start_addr_as_num)
            result_addr = hex(result_addr_as_num)
            transcript_addrs.append(start_addr_as_num)
            transcript_addrs.append(result_addr_as_num)
            line = line.replace(ecpairing, "staticcall(gas(), 0x8, add(transcript, " + transcript_addr + "), 0x180, add(transcript, " + result_addr + "), 0x20")

        m = re.search(mstore_pattern, line)
        if m:
            mstore = m.group(1)
            addr = m.group(2)
            addr_as_num = int(addr, 16)
            transcript_addr = hex(addr_as_num)
            transcript_addrs.append(addr_as_num)
            line = line.replace(mstore, "mstore(add(transcript, " + transcript_addr + ")")

        m = re.search(keccak_pattern, line)
        if m:
            keccak = m.group(1)
            addr = m.group(2)
            addr_as_num = int(addr, 16)
            transcript_addr = hex(addr_as_num)
            transcript_addrs.append(addr_as_num)
            line = line.replace(keccak, "keccak256(add(transcript, " + transcript_addr + ")")

        # mload can show up multiple times per line
        while True:
            m = re.search(mload_pattern, line)
            if not m:
                break
            mload = m.group(1)
            addr = m.group(2)
            addr_as_num = int(addr, 16)
            transcript_addr = hex(addr_as_num)
            transcript_addrs.append(addr_as_num)
            line = line.replace(mload, "mload(add(transcript, " + transcript_addr + ")")

        # print(line, end="")
        modified_lines.append(line)

    # get the max transcript addr
    max_transcript_addr = int(max(transcript_addrs) / 32)
    print("""// SPDX-License-Identifier: MIT
pragma solidity ^0.8.17;

contract Verifier {{
    function verify(
        uint256[] memory pubInputs,
        bytes memory proof
    ) public view returns (bool) {{
        bool success = true;
        bytes32[{}] memory transcript;
        assembly {{
    """.strip().format(max_transcript_addr))
    for line in modified_lines[16:-7]:
        print(line, end="")
    print("""}
        return success;
    }
}""")
