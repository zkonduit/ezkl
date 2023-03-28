import ezkl_lib
import os

# Expect to fail
# print(ezkl_lib.table("test"))
path = os.path.abspath(
    os.path.join(
        os.path.dirname( __file__ ),
        '..',
        'examples',
        'onnx',
        '1l_average',
        'network.onnx'
    )
)
print(
    ezkl_lib.table(path)
)