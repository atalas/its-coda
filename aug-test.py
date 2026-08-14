import numpy as np
from datetime import datetime

noise = .1

# +1 -2 +3 -4 +5 -6
arr = np.array([[3, 4, 2, 5, 1, 6, 0 ],
		[5, 6, 4, 7, 3, 8, 2],
		[12, 13, 11, 4, 10, 15, 9],
		[9, 10, 8, 11, 7, 12, 6],
		[8, 9, 7, 10, 6, 11, 5]])


def aitchisonPerturbation(X, noise_scale=0.01, noise_type='uniform'):
	"""
	Apply Aitchison perturbations to CLR-transformed data.

	Args:
		data: numpy array (n_samples, n_features)
		noise_scale: max absolute perturbation magnitude
		noise_type: 'uniform' or 'laplace'
	Returns:
		perturbed data (same shape)
	"""
	if noise_type == 'uniform':
		noise = np.random.uniform(-noise_scale, noise_scale, X.shape)
	elif noise_type == 'laplace':
		noise = np.random.laplace(0, noise_scale, X.shape)
	else:
		raise ValueError("noise_type must be 'uniform' or 'laplace'")

	return X + noise


def augment1(X, y=None, noise_std=0.1):
	X_augmented = X.copy()
	for col in range(X.shape[1]):
		X_augmented[:, col] += np.random.normal(
			0,
			noise_std * X[:, col].std(),  # Scale noise by feature std
			size=X.shape[0]
		)
	return X_augmented


def augment0(X, y=None, noise_std=0.1):
	X_augmented = X.copy()

	numeric_mask = np.isfinite(X).all(axis=0)
	if numeric_mask.any():
		X_augmented[:, numeric_mask] += np.random.normal(
			0,
			noise_std * np.std(X[:, numeric_mask], axis=0),
			size=X.shape
		)

	return X_augmented

def outputArray(arr, fmtstr):
	now = datetime.now().strftime("%Y-%m-%d-%H%M%S")
	np.savetxt(now + ".tsv", arr, delimiter='\t', fmt=fmtstr)


# handle zeros
pseudo_count = 1.0 
features_safe = arr + pseudo_count

# Calculate the geometric mean across features for each sample
# (row-wise, axis=1)
geom_mean = np.exp(np.mean(np.log(features_safe), axis=1, keepdims=True))

# Apply the Centered Log-Ratio (CLR)
features = np.log(features_safe / geom_mean)
	
outputArray(features, '%.4e')

arrOutput = aitchisonPerturbation(arr)
outputArray(arrOutput, '%.4e')


