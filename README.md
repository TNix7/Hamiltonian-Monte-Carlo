# Hamiltonian-Monte-Carlo
A demonstration of the performance of a Hamiltonian Markov Chain against that of a traditional Metropolis-Hastings Chain.

The files in this repository represent my attempts at producing a functioning Hamiltonian Markov Chain that can be used to produce samples from a geometrically complex target distribution, or one with considerable correlation between parameters.

"HMC vs MH.py"
This is the main code body for this project, and test my Hamiltonian MCMC method against the Metropolis Hastings MCMC.
The chosen target distribution was simple: a 2-dimensional normal distribution with a high covariance. While a more complex target could have been chosen - something I will do in the future - this example is perfectly functional.
The MH chain suffers from severe autocorrelation on account of the high covariance between the two parameters. As a result, the performance falls far short of that from the HMC algorithm, which utilises the underlying differential geometry of the target to reduce autocorrelation.

"HMC 1D working"
This is a simple program that can run a HMC markov chain on a simple 1-dimensional target distribution.

"Ham Traj sim"
This runs the Hamiltonian Markov Chain, again, on a simple normal distribution, but produces an instructive plot of the trajectory, which is helpful with the intuition of the process.

"Hamiltonian_Monte_Carlo Report D4" is my full project report and background research. It also contains my program in the appendix.

"Hamiltonian Monte Carlo Slides" is my breakdown of the report for presentation purposes. It mainly highlights the results of the project, with some background mathematics and intuition.
