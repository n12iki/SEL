from scipy.io import arff
import numpy as np
from io import StringIO
import sys

fname=sys.argv[1]
file = open(fname, "r")
content= file.read()
f = StringIO(content)
data, meta = arff.loadarff(f)
np.random.shuffle(data)

train=data[:int(len(data)*.85)]
test=data[int(len(data)*.85):]
np.savetxt("./train/"+fname[:-5]+".csv",train,fmt = '%s', delimiter=",")
np.savetxt("./test/"+fname[:-5]+".csv",test,fmt = '%s', delimiter=",")
print("Finished conversion")
#print(type(data))