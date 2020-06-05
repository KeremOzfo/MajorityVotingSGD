import numpy as np
import matplotlib.pyplot as plt

result = np.load('sparse1.0_type-majority_Voting-ErrorCor-True-new.npy')[0]
benchmark = np.load('SingleMachine-mnist-startingLR-0.1-B-Size-64--0.npy')

x_axis = []
for i in range(len(result)):
    x_axis.append(i*10)

plt.plot(x_axis,result,linewidth=2, label='10-Worker-majority-Voting')
plt.plot(x_axis,benchmark,linewidth=2, label='SingleMachine-Benchmark')
plt.xlabel('iteration')
plt.ylabel('Accuracy')
plt.legend()
plt.show()