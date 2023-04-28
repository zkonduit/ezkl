import ezkl_lib
import os

folder_path = os.path.abspath(
    os.path.join(
        os.path.dirname(__file__),
        '.',
    )
)

examples_path = os.path.abspath(
        os.path.join(
            folder_path,
            '..',
            '..',
            'examples',
        )
    )

def test_x():
    """
    Test for mock
    """

    data_path = os.path.join(
        examples_path,
        'onnx',
        '1l_average',
        'input.json'
    )

    model_path = os.path.join(
        examples_path,
        'onnx',
        '1l_average',
        'network.onnx'
    )

    print(ezkl_lib.mock(data_path, model_path))


if __name__ == "__main__":
    test_x()
