import ezkl_lib
import os
import pytest

folder_path = os.path.abspath(
    os.path.join(
        os.path.dirname(__file__),
        '.',
    )
)

params_path = os.path.join(folder_path, 'kzg_test.params')

def test_table_1l_average():
    """
    Test for table() with 1l_average.onnx
    """
    path = os.path.abspath(
        os.path.join(
            folder_path,
            '..',
            '..',
            'examples',
            'onnx',
            '1l_average',
            'network.onnx'
        )
    )

    expected_table = \
        """+-------+-----------------------------------------------------------------+------------+----------+-----------+-------------+-----------------+--------+-------------+-----------+-----+--------+
| usize | opkind                                                          | output_max | in_scale | out_scale | const_value | raw_const_value | inputs | in_dims     | out_dims  | idx | bucket |
+-------+-----------------------------------------------------------------+------------+----------+-----------+-------------+-----------------+--------+-------------+-----------+-----+--------+
| 0     | input                                                           | 256        | 7        | 7         |             |                 |        | [[1, 5, 5]] | [1, 5, 5] | 0   |        |
+-------+-----------------------------------------------------------------+------------+----------+-----------+-------------+-----------------+--------+-------------+-----------+-----+--------+
| 1     | padding: (0, 0)                                                 | 256        | 7        | 7         |             |                 | [0]    | [[1, 5, 5]] | [1, 5, 5] | 1   |        |
+-------+-----------------------------------------------------------------+------------+----------+-----------+-------------+-----------------+--------+-------------+-----------+-----+--------+
| 2     | avg pl w/ padding: (0, 0), stride: (1, 1), kernel shape: (3, 3) | 32768      | 7        | 7         |             |                 | [1]    | [[1, 5, 5]] | [1, 3, 3] | 2   |        |
+-------+-----------------------------------------------------------------+------------+----------+-----------+-------------+-----------------+--------+-------------+-----------+-----+--------+"""

    assert ezkl_lib.table(path) == expected_table


def test_gen_srs():
    """
    Test for gen_srs() with 17 logrows.
    You may want to comment this test as it takes a long time to run
    """
    ezkl_lib.gen_srs(params_path, 17)
    assert os.path.isfile(params_path)
