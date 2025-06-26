from torch import nn
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from export import export

circuit = nn.Linear(1, 1)

export(circuit, input_shape=[1], include_output=False)
