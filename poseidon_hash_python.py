import ezkl
import os
import pytest
import json
import subprocess
import time

def test_poseidon_hash():
    """
    Test for poseidon_hash
    """
    message = [1.0, 2.0, 3.0, 4.0]
    message = [ezkl.float_to_string(x, 7) for x in message]
    res = ezkl.poseidon_hash(message)
    print(res)
    assert ezkl.string_to_felt(
        res[0]) == "0x0da7e5e5c8877242fa699f586baf770d731defd54f952d4adeb85047a0e32f45"
    
    
if __name__ == "__main__":
    test_poseidon_hash()
    print("poseidon_hash test passed")