#calculate posterior methylation rate

import numpy as np

def calculate_posterior_mc_rate(mc_da, cov_da, normalize_per_cell=True, clip_norm_value=10):
	# TODO calculate cell_a, cell_b separately
	# so we can do post_rate only in a very small set of gene to prevent memory issue
	raw_rate = mc_da / cov_da
	cell_rate_mean = np.nanmean(raw_rate, axis=1)[:, None]  # this skip na
	cell_rate_var = np.nanvar(raw_rate, axis=1)[:, None]  # this skip na
	# based on beta distribution mean, var
	# a / (a + b) = cell_rate_mean
	# a * b / ((a + b) ^ 2 * (a + b + 1)) = cell_rate_var
	# calculate alpha beta value for each cell
	cell_a = (1 - cell_rate_mean) * (cell_rate_mean ** 2) / cell_rate_var - cell_rate_mean
	cell_b = cell_a * (1 / cell_rate_mean - 1)
	# cell specific posterior rate
	post_rate = (mc_da + cell_a) / (cov_da + cell_a + cell_b)
	if normalize_per_cell:
		# there are two ways of normalizing per cell, by posterior or prior mean:
		# prior_mean = cell_a / (cell_a + cell_b)
		# posterior_mean = post_rate.mean(dim=var_dim)
		# Here I choose to use prior_mean to normalize cell,
		# therefore all cov == 0 features will have normalized rate == 1 in all cells.
		# i.e. 0 cov feature will provide no info
		prior_mean = cell_a / (cell_a + cell_b)
		post_rate = post_rate / prior_mean
		if clip_norm_value is not None:
			post_rate[post_rate > clip_norm_value] = clip_norm_value
	return post_rate

