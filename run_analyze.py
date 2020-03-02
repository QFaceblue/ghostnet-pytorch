import numpy as np
import matplotlib.pyplot as plt


# log = np.load('configs/cnn_l4b2_cifar10.npy')
log = np.load('log.npy')
acc = []
for i in range(len(log)):
    acc.append(log[i][0])

plt.plot(acc, color='r', label='acc')
plt.legend(bbox_to_anchor=(0., 1.02, 1., .102), loc=3,
               ncol=3, mode='expand', borderaxespad=0.)

acc = np.around(acc, decimals=4)

print('acc:')
print(acc)
print('acc_max:', float(np.max(acc)))

plt.show()
