import json
import os
import time
import pandas as pd
import numpy as np
from thomas_tools import Inferencer, write_pickframe

project_name = '2012-London-final-ld-lcw'
# project_name = '2021-England-final-lzj-asl'
# project_name = '2020-Tokyo-final-cyf-dzy'
project_name = '2019-China-final-tt-jt'

# project_name = 'tennis-1'
# project_name = 'tennis-2'
# project_name = 'asl-vs-why'

shift_param = (100, 100)

tmp = time.time()
project_inferencer = Inferencer(project_name, shift_param)



# predictions = project_inferencer.infer(model_type='new-model')

# print(f'track time: {time.time() - tmp:.2f}s')

project_inferencer.write_testpickframe(window=60)

print(f'pickframe time: {time.time() - tmp:.2f}s')

# project_inferencer.write_label()

# print(f'label time: {time.time() - tmp:.2f}s')