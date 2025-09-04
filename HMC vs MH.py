import numpy as np
import matplotlib.pyplot as plt

def HMC(
    neg_log_prob, grad_neg_log_prob, 
    initial_position,
    M, 
    path_len=1.0, step_size=0.1, 
    n_samples=1000,

):
    
    position = initial_position.copy()
    dim = len(position)
    samples = [position.copy()]
    steps = int(path_len / step_size)
    momentum_dist = lambda: np.random.multivariate_normal(mean=np.zeros(dim), cov=M)
    M_inv = np.linalg.inv(M)

    accepted = 0

    for _ in range(n_samples):
        q0 = position.copy()
        p0 = momentum_dist()
        q = q0.copy()
        p = p0.copy()

        # Initial half step for momentum
        p -= 0.5 * step_size * grad_neg_log_prob(q)

        # Leapfrog steps
        for _ in range(steps):
            q += step_size * M_inv @ p
            if _ != steps - 1:
                p -= step_size * grad_neg_log_prob(q)

        # Final half step for momentum
        p -= 0.5 * step_size * grad_neg_log_prob(q)

        # Flip momentum to make proposal symmetric
        p = -p

        current_H = neg_log_prob(q0) + 0.5 * p0 @ M_inv @ p0
        proposed_H = neg_log_prob(q) + 0.5 * p @ M_inv @ p 


        acceptance_prob = min(1, np.exp(current_H - proposed_H))

        if np.random.rand() < acceptance_prob:
            position = q.copy()
            accepted += 1
        samples.append(position.copy())

    print("HMC acceptance rate:", accepted/n_samples)
    return np.array(samples)


#-------------------------------------------------------------

def MH(
    neg_log_prob,
    initial_position,
    proposal_cov,
    scaling_factor,
    n_samples=1000
):
    position = initial_position.copy()
    samples = [position.copy()]
    
    accepted = 0

    for _ in range(n_samples):
        proposal = position + np.random.multivariate_normal(np.array([0, 0]), scaling_factor * proposal_cov)
        acceptance_prob = min(1, np.exp(-neg_log_prob(proposal) + neg_log_prob(position)))
        
        if np.random.rand() < acceptance_prob:
            position = proposal
            accepted += 1
        samples.append(position.copy())
        
    print("MH acceptance rate:", accepted/n_samples)
    return np.array(samples)

#-------------------------------------------------------------

def make_multivariate_normal(mu, cov):
    cov_inv = np.linalg.inv(cov)

    def neg_log_prob(x):
        diff = x - mu
        return 0.5 * diff @ cov_inv @ diff.T

    def grad_neg_log_prob(x):
        return cov_inv @ (x - mu)

    return neg_log_prob, grad_neg_log_prob


#------------------------------------------------------------

# 2D Gaussian with correlation
mu = np.array([0.0, 0.0])
cov = np.array([[1.0, 0.8],
                [0.8, 1.0]])
neg_log_prob, grad_neg_log_prob = make_multivariate_normal(mu, cov)

#------------------------------------------------------------

HMC_samples = HMC(
    neg_log_prob=neg_log_prob,
    grad_neg_log_prob=grad_neg_log_prob,
    initial_position=np.array([0.0, 0.0]),
    step_size=0.5,
    path_len=5,
    n_samples=20000,
    M = np.linalg.inv(cov)
)

HMC_samples = HMC_samples[5000:, :]

#---------------------------------------------------------------

MH_samples = MH(
    neg_log_prob=neg_log_prob,
    initial_position=np.array([0.0, 0.0]),
    proposal_cov = cov,
    scaling_factor = 5.7,
    n_samples=20000
)

MH_samples = MH_samples[5000:, :]

#--------------------------------------------------------------

print("HMC estimates the mean to be:", np.mean(HMC_samples, axis=0))
print("HMC estimates the variance to be:", np.var(HMC_samples, axis=0))

print("MH estimates the mean to be:", np.mean(MH_samples, axis=0))
print("MH estimates the variance to be:", np.var(MH_samples, axis=0))






#--------------------------------------------------------------
#'''
# Create a figure with two subplots side by side
fig, axes = plt.subplots(1, 2, figsize=(12, 6))


# HMC plot
axes[0].scatter(HMC_samples[500:, 0], HMC_samples[500:, 1], alpha=0.3, s=5)
axes[0].set_title("Samples from Bivariate Normal via HMC")
axes[0].set_xlabel("x")
axes[0].set_ylabel("y")
axes[0].set_xlim(-4,4)
axes[0].set_ylim(-4,4)
#axes[0].axis('equal')
axes[0].grid(True)

# MH plot
axes[1].scatter(MH_samples[500:, 0], MH_samples[500:, 1], alpha=0.3, s=5)
axes[1].set_title("Samples from Bivariate Normal via MH")
axes[1].set_xlabel("x")
axes[1].set_ylabel("y")
axes[1].set_xlim(-4,4)
axes[1].set_ylim(-4,4)
#axes[1].axis('equal')
axes[1].grid(True)

# Show the plot
plt.tight_layout()
#plt.show()

#'''

#-------------------------------------------------------------------------------
import arviz as az

HMC_array = np.expand_dims(np.array(HMC_samples), axis=0)
MH_array = np.expand_dims(np.array(MH_samples), axis=0)

# Wrap into InferenceData
HMC_idata = az.from_dict(posterior={"x": HMC_array})
MH_idata = az.from_dict(posterior={"x": MH_array})

# Compute ESS
HMC_ess = az.ess(HMC_idata)
MH_ess = az.ess(MH_idata)

print("HMC ESS:", HMC_ess)
print("MH ESS:", MH_ess)

#-----------------------------------------------------------------------------

from statsmodels.graphics.tsaplots import plot_acf

from statsmodels.graphics.tsaplots import plot_acf

fig, axs = plt.subplots(2, 2, figsize=(12, 8))

# HMC - Dimension 1
plot_acf(HMC_samples[:, 0], lags=50, ax=axs[0, 0])
axs[0, 0].set_title("HMC ACF - Dimension 1")

# HMC - Dimension 2
plot_acf(HMC_samples[:, 1], lags=50, ax=axs[0, 1])
axs[0, 1].set_title("HMC ACF - Dimension 2")

# MH - Dimension 1
plot_acf(MH_samples[:, 0], lags=50, ax=axs[1, 0])
axs[1, 0].set_title("MH ACF - Dimension 1")

# MH - Dimension 2
plot_acf(MH_samples[:, 1], lags=50, ax=axs[1, 1])
axs[1, 1].set_title("MH ACF - Dimension 2")

plt.tight_layout()
plt.suptitle("Autocorrelation Comparison: HMC vs MH", fontsize=16, y=1.03)
plt.show()