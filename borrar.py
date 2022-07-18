import numpy as np
import scipy.optimize as op
from joblib import Parallel, delayed
from time import perf_counter
from sklearn.linear_model import LinearRegression


n_jobs_ = -1
x = np.random.random(4000000).reshape(10000,400)
y = -1.2*x[:,0] - 2.7*x[:,1] + 3 + 1*np.random.random(10000)-0.5

X = np.column_stack((np.ones(x.shape[0]), x))

s = perf_counter()
if y.ndim < 2:
    temp = op.nnls(X, y)[0]
    intercept_, coef_ = temp[0], temp[1:]
else:
    # scipy.optimize.nnls cannot handle y with shape (M, K)
    temps = Parallel(n_jobs=n_jobs_)(
        delayed(op.nnls)(X, y[:, j]) for j in range(y.shape[1])
    )
    t = np.vstack([temp[0] for temp in temps])
    intercept_, coef_ = t[:,0], t[:,1:]
print(f'{perf_counter()-s}')
# print(coef_)

s = perf_counter()
if y.ndim < 2:
    temp = op.nnls(X,y)[0]
    intercept_, coef_ = temp[0], temp[1:]
else:
    # slicing a column y[:,j] gives 1D array
    temp = np.array([op.nnls(X, y[:,j])[0] for j in range(y.shape[1])])
    intercept_, coef_ = temp[:,0], temp[:,1:]
print(f'{perf_counter()-s}')

s = perf_counter()
L = LinearRegression(n_jobs=n_jobs_, positive=True).fit(x,y)
print(f'{perf_counter()-s}')

# print(coef_)