# -*- coding:utf-8 -*-
import matplotlib.pyplot as plt 
import numpy as np
plt.subplot(3,2,1)
x1 = np.array([1,2,3,1,5,6,5,5,6,7,8,9,7,9])
x2 = np.array([1,3,2,2,8,6,7,6,7,1,2,1,1,3])
x = np.array(zip(x1,x2)).reshape(len(x1), 2)
plt.xlim([0, 10])
plt.ylim([0, 10])
plt.title('Instance01')
plt.scatter(x1, x2)
plt.show()
