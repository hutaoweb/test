from vocab import Vocabulary
import evaluation
from time import *
import os
from mindspore import context

begin_time = time()
os.environ["CUDA_VISIBLE_DEVICES"] = "0"
context.set_context(mode=context.PYNATIVE_MODE, device_target="Ascend")
#model_path  
#data_path   
#split
range_ = range(0,20)  #
evaluation.val(model_path = "weight_log/", data_path="./data/", split="dev",range_=range_)
end_time = time()
run_time = end_time-begin_time
print ('time: ',run_time) 