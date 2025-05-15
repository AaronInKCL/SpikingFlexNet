import numpy as np
eigen = np.load("./hessian_iter_0/density_eigen.npy")
weight = np.load("./hessian_iter_0/density_weight.npy")

print("Eigen:", eigen[:10])
print("Weight:", weight[:10])
print("Weight sum:", np.sum(weight))
