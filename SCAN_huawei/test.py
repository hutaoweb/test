from vocab import Vocabulary
import evaluation
from time import *
import os
from mindspore import context

begin_time = time()
os.environ["CUDA_VISIBLE_DEVICES"] = "0"
context.set_context(mode=context.PYNATIVE_MODE, device_target="Ascend")
evaluation.evalrank("temp/", data_path="data/", split="test")
end_time = time()
run_time = end_time-begin_time
print ('该循环程序运行时间：',run_time) 