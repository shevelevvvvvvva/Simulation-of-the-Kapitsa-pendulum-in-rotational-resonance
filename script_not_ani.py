import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import solve_ivp


def equation_of_rotation(t, y, omega0, omega, gamma, a, L):
    theta, phi = y
    dydt = [
        phi,
        - 2 * gamma * (phi - omega) + (omega0 ** 2 - (a / L) * omega ** 2 * np.cos(omega * t)) * np.sin(theta)
    ]
    return dydt


def sustainability_matrix(params, T, sol_main):
    omega = params['omega']
    omega0 = params['omega0']
    gamma = params['gamma']
    a = params['a']
    L = params['L']

    def variational_eq(t, y):
        xi, dxi = y  # xi = theta - theta0, dxi = phi - phi0
        theta0 = sol_main.sol(t)[0]
        coeff = (omega0 ** 2 - (a / L) * omega ** 2 * np.cos(omega * t)) * np.cos(theta0)

        return [
            dxi,
            -2 * gamma * dxi + coeff * xi
        ]

    sol1 = solve_ivp(variational_eq, [0, T], [1, 0], method='RK45', rtol=1e-8)
    sol2 = solve_ivp(variational_eq, [0, T], [0, 1], method='RK45', rtol=1e-8)

    M = np.array([
        [sol1.y[0, -1], sol2.y[0, -1]],
        [sol1.y[1, -1], sol2.y[1, -1]]
    ])
    return M


def simulate_and_analyze(params, y0, t_span=(0, 50)):
    sol_main = solve_ivp(equation_of_rotation, t_span, y0,
                         args=(params['omega0'], params['omega'], params['gamma'], params['a'], params['L']),
                         method='RK45', dense_output=True, rtol=1e-6)
    t = np.linspace(t_span[0], t_span[1], 5000)
    theta, phi = sol_main.sol(t)

    plt.figure(figsize=(16, 12))

    plt.subplot(2, 2, 1)
    plt.plot(t, theta, label=f'θ(t), ω={params["omega"]:.1f}, ω₀={params["omega0"]:.1f}')
    plt.xlabel('Время t')
    plt.ylabel('Угол θ')
    plt.grid(True)
    plt.legend()

    plt.subplot(2, 2, 2)
    plt.plot(t, phi, 'r', label=f'φ(t)')
    plt.xlabel('Время t', fontsize=10)
    plt.ylabel('Угловая скорость φ', fontsize=10)
    plt.grid(True)
    plt.legend()

    plt.subplot(2, 2, 3)
    plt.plot(theta, phi, '.', markersize=1)
    plt.xlabel('θ (рад)', fontsize=10)
    plt.ylabel('φ (рад/с)', fontsize=10)
    plt.title('Фазовый портрет (θ, φ)', fontsize=10, loc='left')
    plt.grid(True)

    theta_minus_omegat = theta - params['omega'] * t
    plt.subplot(2, 2, 4)
    plt.plot(t, theta_minus_omegat, 'g', label='ξ(t) = θ(t) - ωt')
    plt.xlabel('Время t')
    plt.ylabel('ξ(t) (рад)')
    plt.grid(True)
    plt.legend()

    plt.tight_layout()
    plt.show()

    T = 2 * np.pi / params['omega']
    M = sustainability_matrix(params, T, sol_main)
    eigvals = np.linalg.eigvals(M)
    print(f"Собственные значения: {eigvals[0]:.4f}, {eigvals[1]:.4f}")
    print(f"Модули: {np.abs(eigvals[0]):.4f}, {np.abs(eigvals[1]):.4f}")
    if all(np.abs(eigvals) < 1):
        print("Система устойчива")
    else:
        print("Система неустойчива")


if __name__ == "__main__":
    params = {
        'omega0': np.sqrt(9.8),
        'omega': 4 * np.sqrt(9.8),
        'gamma': 0.1,
        'a': 0.01,
        'L': 1.0
    }
    y0 = [np.pi, params['omega']]
    simulate_and_analyze(params, y0, t_span=(0, 30))