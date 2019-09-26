import requests
import json
from tensorflow.examples.tutorials.mnist import input_data
import numpy as np



mnist = input_data.read_data_sets("./data/")


class NumpyEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        return json.JSONEncoder.default(self, obj)


x_test, y_test = mnist.test.next_batch(1)
data = {'input_x': x_test[0]}
param = {'instances': [data]}
param = json.dumps(param, cls=NumpyEncoder)
# res = requests.post("http://localhost:8501/v1/models/mnist:predict", data=param)
res = requests.post("http://localhost:8500/v1/models/mnist:predict", data=param)

res.encoding = 'gb18030'
res = res.json()

print("得到的结果是",res)
#print(res.text)
