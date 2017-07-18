import numpy as np

biases = np.random.rand(10,5)
nabla_b = [np.zeros(b.shape) for b in biases]
delta_nabla_b = np.random.rand(10,5)
eta = 0.01
mini_batch = np.zeros((100,0))

nabla_b_new = [nb+dnb for nb, dnb in zip(nabla_b, delta_nabla_b)]

biases_new = [b-(eta/len(mini_batch))*nb for b, nb in zip(biases, nabla_b_new)]
