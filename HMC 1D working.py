import numpy as np
import random
import scipy.stats as st
import matplotlib.pyplot as plt

def normal(x, mu, sigma):
    numerator = np.exp(-((x - mu) ** 2) / (2 * sigma ** 2))
    denominator = sigma * np.sqrt(2 * np.pi)
    return numerator / denominator

def neg_log_prob(x, mu, sigma):
    return -np.log(normal(x=x, mu=mu, sigma=sigma))

def HMC(mu=0.0, sigma=1.0, path_len=1.0, step_size=0.125, initial_position=0.0, epochs=1000):
    # Setup
    steps = int(path_len / step_size)
    samples = [initial_position]
    momentum_dist = st.norm(0, 1)

    accepted = 0

    # Sampling loop
    for _ in range(epochs):
        q0 = samples[-1]
        p0 = momentum_dist.rvs()
        
        q1 = np.copy(q0)
        p1 = np.copy(p0)

        # Initial half-step for momentum
        dVdQ = - (q1 - mu) / (sigma**2)
        p1 += 0.5 * step_size * dVdQ

        # Leapfrog steps
        for s in range(steps):
            # Full step for position
            q1 += step_size * p1

            # Full step for momentum (except at final step)
            if s < steps - 1:
                dVdQ = - (q1 - mu) / (sigma**2)
                p1 += step_size * dVdQ

        # Final half-step for momentum
        dVdQ = - (q1 - mu) / (sigma**2)
        p1 += 0.5 * step_size * dVdQ

        # Momentum flip for reversibility
        p1 = -p1

        # Metropolis-Hastings acceptance step
        current_H = neg_log_prob(q0, mu, sigma) + neg_log_prob(p0, 0, 1)
        proposed_H = neg_log_prob(q1, mu, sigma) + neg_log_prob(p1, 0, 1)
        acceptance_log_prob = current_H - proposed_H

        if np.log(random.uniform(0, 1)) < acceptance_log_prob:
            accepted += 1
            samples.append(q1)
        else:
            samples.append(q0)
    print('Acceptance:', accepted/len(samples))
    return samples

# Test the sampler
mu = 0
sigma = 1
samples = HMC(mu=mu, sigma=sigma, path_len=10, step_size=0.1, initial_position=0.0, epochs=20000)

# Plot the results
x_vals = np.linspace(-6, 6, 1000)
true_density = normal(x_vals, mu, sigma)

print("Program complete.")


plt.figure(figsize=(10, 5))
plt.plot(x_vals, true_density, label='True Normal PDF', linewidth=2)
plt.hist(samples[5000:], density=True, bins=200, alpha=0.6, label='HMC Samples')
plt.title("HMC Sampling of 1D Normal Distribution")
plt.xlabel("x")
plt.ylabel("Density")
plt.legend()
plt.grid(True)
plt.show()
