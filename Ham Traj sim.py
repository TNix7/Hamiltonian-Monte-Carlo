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
    all_trajectories = []

    for _ in range(n_samples):
        q0 = position.copy()
        p0 = momentum_dist().reshape((1, 1))
        q = q0.copy()
        p = p0.copy()

        # Store the trajectory for this sample
        trajectory = [(q.item(), p.item())]

        # Initial half step
        p -= 0.5 * step_size * grad_neg_log_prob(q)

        for _ in range(steps):
            q += step_size * M_inv @ p
            if _ != steps - 1:
                p -= step_size * grad_neg_log_prob(q)
            trajectory.append((q.item(), p.item()))

        # Final half step
        p -= 0.5 * step_size * grad_neg_log_prob(q)
        trajectory.append((q.item(), p.item()))

        all_trajectories.append(trajectory)

        # Flip momentum to make proposal symmetric
        p = -p

        current_H = neg_log_prob(q0) + 0.5 * p0 @ M_inv @ p0
        proposed_H = neg_log_prob(q) + 0.5 * p @ M_inv @ p

        acceptance_prob = min(1, np.exp(current_H - proposed_H))
        if np.random.rand() < acceptance_prob:
            position = q.copy()
            accepted += 1
        samples.append(position.copy())

    print("HMC acceptance rate:", accepted / n_samples)
    return np.array(samples), all_trajectories

def make_multivariate_normal(mu, cov):
    cov_inv = np.linalg.inv(cov)

    def neg_log_prob(x):
        diff = x - mu
        return 0.5 * diff @ cov_inv @ diff.T

    def grad_neg_log_prob(x):
        return cov_inv @ (x - mu)

    return neg_log_prob, grad_neg_log_prob

mu = np.array([[0.0]])
cov = np.array([[1.0]])

neg_log_prob, grad_neg_log_prob = make_multivariate_normal(mu, cov)

HMC_samples, all_trajectories = HMC(
    neg_log_prob=neg_log_prob,
    grad_neg_log_prob=grad_neg_log_prob,
    initial_position=np.array([[0.0]]),
    step_size=0.01,
    path_len=3,
    n_samples=5,
    M = np.linalg.inv(cov)
)

# Plot phase space trajectories
plt.figure(figsize=(10, 6))

trajectory_idx = 0 # For traj colours

for trajectory in all_trajectories:
    q_vals, p_vals = zip(*trajectory)
    q_vals = np.array(q_vals)
    p_vals = np.array(p_vals)

    # Step through the trajectory and change color after momentum flip
    step = 10  #adjust arrow density (higher = less dense)

    q_vals_sampled = q_vals[::step]
    p_vals_sampled = p_vals[::step]

    # Compute differences between consecutive sampled points
    dq = q_vals_sampled[1:] - q_vals_sampled[:-1]
    dp = p_vals_sampled[1:] - p_vals_sampled[:-1]

    # Start points for arrows
    q_start = q_vals_sampled[:-1]
    p_start = p_vals_sampled[:-1]

    # Define color for this trajectory (change color for each new trajectory)
    color = plt.cm.coolwarm(trajectory_idx / len(all_trajectories))

    # Plot arrows for this trajectory
    plt.quiver(
        q_start, p_start,
        dq, dp,            # arrow directions
        angles='xy', scale_units='xy', scale=1, 
        width=0.003, headwidth=3, alpha=0.6,
        color=color
    )

    trajectory_idx += 1  # Increment trajectory index for coloring

plt.title("HMC Phase Space Trajectories")
plt.xlabel("Position (q)")
plt.ylabel("Momentum (p)")
plt.grid(True)
plt.tight_layout()
plt.show()