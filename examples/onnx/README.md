# Examples

These are examples used for testing the various ONNX operations that should be supported

To generate a new set of `input.json` and `network.onnx` files where `gen.py` exists run:

```shell
# set up a new virtual env
python -m venv .env

# install additional dependencies
pip install -r requirements

# run the script that calls gen.py where available
./gen.sh
```