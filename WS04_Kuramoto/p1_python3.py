"""
Worksheet 4, Problem 1  --  Python 3 corrected version of p1.py

Kuramoto model of N coupled oscillators

    d(theta_i)/dt = omega_i + (K/N) sum_j sin(theta_j - theta_i)

with theta_i(0) uniform on [0, 2 pi] and intrinsic frequencies drawn from the
Lorentzian density  P(omega) = 1 / (pi (1 + omega^2)).

Plot the magnitude of the order parameter  r e^{i psi} = (1/N) sum_j e^{i theta_j}
in the steady state, against K in [0, 6], for N = 100, 200, 400.

Changes made relative to the original p1.py
-------------------------------------------
1. The mean phase was computed as
        psi = np.angle(np.sum(np.exp(1j * theta))) / n
   Dividing an ANGLE by N is meaningless -- the /N belongs to the magnitude
   only (which the line above already had).  The coupling term therefore
   used a psi that was wrong by a factor of N, which distorts the dynamics
   and moves the transition away from the true K_c.  Corrected.
2. The steady-state r was read off a single snapshot (index len(t)-30).
   A single instant of a fluctuating quantity is noisy, especially near the
   transition; r is now averaged over the last third of the trajectory.
3. plt.show() sat inside the loop over N, so the user had to close three
   windows one at a time to let the program finish.  The three curves now go
   into one figure with three panels.
4. t[i+1] = t[i] + h mutated the time array inside the innermost loop to no
   effect.  Removed.
5. f1 closed over the loop variable k implicitly; K is now an argument.
6. The measured transition is compared against the analytic critical
   coupling for a Lorentzian frequency spread,
        K_c = 2 / (pi P(0)) = 2 gamma = 2   for gamma = 1.

Algorithms are hand written: RK4 is implemented here, and the O(N) mean-field
form of the coupling is used rather than the O(N^2) double sum -- the two are
algebraically identical:
    (K/N) sum_j sin(theta_j - theta_i) = K r sin(psi - theta_i)
"""

import numpy as np
import matplotlib.pyplot as plt


def order_parameter(theta, n):
    """
    r e^{i psi} = (1/N) sum_j e^{i theta_j}.

    Returns (r, psi).  r is divided by N; psi is an angle and is NOT.
    """
    z = np.sum(np.exp(1j * theta))
    return np.abs(z) / n, np.angle(z)


def derivative(theta, omega, K, n):
    """d(theta)/dt for the whole oscillator array at once."""
    r, psi = order_parameter(theta, n)
    return omega + K * r * np.sin(psi - theta)


def simulate(K, omega, theta0, n, t_end=40.0, n_steps=2000,
             average_fraction=0.33):
    """
    Integrate the N oscillators with RK4 and return the steady-state r,
    averaged over the last `average_fraction` of the trajectory.
    """
    h = t_end / n_steps
    theta = theta0.copy()
    n_avg = int(n_steps * average_fraction)
    start_avg = n_steps - n_avg

    r_sum = 0.0
    for i in range(n_steps):
        k1 = h * derivative(theta, omega, K, n)
        k2 = h * derivative(theta + k1 / 2.0, omega, K, n)
        k3 = h * derivative(theta + k2 / 2.0, omega, K, n)
        k4 = h * derivative(theta + k3, omega, K, n)
        theta = theta + (k1 + 2.0 * k2 + 2.0 * k3 + k4) / 6.0

        if i >= start_avg:
            r_sum += order_parameter(theta, n)[0]

    return r_sum / n_avg


def main():
    sizes = [100, 200, 400]
    K_grid = np.linspace(0.0, 6.0, 40)
    K_c_exact = 2.0        # 2 gamma, with gamma = 1 for P(w)=1/(pi(1+w^2))

    fig, axes = plt.subplots(1, len(sizes), figsize=(14, 4), sharey=True)

    # Two analytic checks, both exact in the N -> infinity limit:
    #   below threshold the incoherent state has r ~ 1/sqrt(N) (finite size)
    #   above threshold  r = sqrt(1 - K_c/K)
    K_top = K_grid[-1]
    r_top_exact = np.sqrt(1.0 - K_c_exact / K_top)

    print("Analytic critical coupling for a unit Lorentzian: K_c = %.1f"
          % K_c_exact)
    print("Analytic r at K = %.1f : sqrt(1 - K_c/K) = %.4f\n"
          % (K_top, r_top_exact))
    print("%-8s %-16s %-16s %-16s"
          % ("N", "r at K=0", "1/sqrt(N)", "r at K=%.0f" % K_top))

    for ax, n in zip(axes, sizes):
        # Cauchy == Lorentzian with gamma = 1, exactly the P(omega) asked for
        omega = np.random.standard_cauchy(n)
        theta0 = np.random.uniform(0.0, 2.0 * np.pi, n)

        R = np.zeros(len(K_grid))
        for j, K in enumerate(K_grid):
            R[j] = simulate(K, omega, theta0, n)

        print("%-8d %-16.4f %-16.4f %-16.4f"
              % (n, R[0], 1.0 / np.sqrt(n), R[-1]))

        ax.plot(K_grid, R, "o-", ms=4)
        ax.axvline(K_c_exact, ls="--", color="red",
                   label=r"$K_c$ = %.0f (theory)" % K_c_exact)
        ax.set_xlabel("K")
        ax.set_title("N = %d" % n)
        ax.set_ylim(0.0, 1.05)
        ax.legend(fontsize=8)

    axes[0].set_ylabel(r"$|r|$  (steady state)")
    fig.suptitle("Kuramoto order parameter vs coupling strength")
    fig.tight_layout()
    plt.show()


if __name__ == "__main__":
    main()
