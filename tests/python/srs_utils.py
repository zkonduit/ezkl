"""
This is meant to be used locally for development.
Generating the SRS is costly so we run this instead of creating a new SRS each
time we run tests.
"""

import ezkl_lib
import os

params_path = os.path.abspath(
    os.path.join(
        os.path.dirname(__file__),
        '.',
        'kzg_test.params',
    )
)

def gen_test_srs(logrows=17):
    """Generates a test srs with 17 log rows"""
    ezkl_lib.gen_srs(params_path, logrows)


def delete_test_srs():
    """Deletes test srs after tests are done"""
    os.remove(params_path)


if __name__ == "__main__":
    # gen_test_srs()
    path = os.path.abspath(
        os.path.join(
            os.path.dirname(__file__),
            '..',
            '..',
            'examples',
            'onnx',
            '1l_average',
            'network.onnx'
        )
    )
    print(ezkl_lib.table(path))