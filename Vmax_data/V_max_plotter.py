import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.linalg import sqrtm

# Load the data
V_mock = np.load("Vmax_data/V_mock_alpha_-1_scatter_0.2.npy")
V_mock_err = np.load("Vmax_data/V_mock_err_alpha_-1_scatter_0.2.npy")
V_obs = np.load("Vmax_data/V_obs_alpha_-1_scatter_0.2.npy")
V_obs_err = np.load("Vmax_data/V_obs_err_alpha_-1_scatter_0.2.npy")

V_mock = V_mock.reshape(-1, 1)
V_mock_err = V_mock_err.reshape(-1, 1)
V_obs = V_obs.reshape(-1, 1)
V_obs_err = V_obs_err.reshape(-1, 1)

Var_X = np.cov(np.hstack((V_mock_err, V_mock_err)))
#Var_Y = np.cov(V_obs, V_obs)
#Var_XY = np.cov(V_mock, V_obs)

print(Var_X) #, Var_Y, Var_XY)

exit()

Sigma = np.array([[Var_X, Var_XY], [Var_XY, Var_Y]])

plt.figure(figsize=(10, 8))
sns.heatmap(np.log10(Sigma), annot=False, cmap='coolwarm')
plt.title('Covariance Matrix of Combined Errors')
plt.show()



