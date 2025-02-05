import numpy as np
import matplotlib.pyplot as plt


def wiener(time, seed=42):
    """
    Generates the values for three different Wiener processes.
    :param time: The final time.
    :param seed: Random generation seed.
    :return: Array of values for the three different Wiener processes.
    """
    np.random.seed(seed)  # Seed the generator for reproducibility.
    n = int(time/ref_dt)  # Compute the number of approximations.
    result = np.zeros((3, n))  # Set the initial values to zero.
    for i in range(3):  # Iterate over the three Wiener processes.
        for j in range(1, n):  # Iterate over the time stamps.
            result[i][j] = (result[i][j-1]
                            + np.random.normal(0, np.sqrt(ref_dt)))  # Wiener process random increment
    return result


def euler_maruyama(dt, time, seed=42, be=1, de=1.4, beta1=1, beta2=1, alpha1=0.5, alpha2=0.5):
    """
    Euler-Maruyama approximation.
    :param dt: Time step.
    :param time: Final time.
    :param seed: Random generation seed.
    :param be: Formula constant.
    :param de: Formula constant.
    :param beta1: Formula constant.
    :param beta2: Formula constant.
    :param alpha1: Formula constant.
    :param alpha2: Formula constant.
    :return: The Euler-Maruyama approximations of the population size.
    """
    # Compute the number of time stamps.
    n = int(time / dt)

    # Initialize the computation arrays.
    y = np.empty(n)
    b = np.empty(n)
    d = np.empty(n)

    # Set the initial values.
    y[0] = 30
    b[0] = be
    d[0] = de

    # Compute the step difference compared to the reference time step.
    step = int(dt/ref_dt)

    # Take the corresponding Wiener process values.
    w = wiener(time, seed)
    w1 = w[0][:n * step:step]
    w2 = w[1][:n * step:step]
    w3 = w[2][:n * step:step]

    # Compute the Euler-Maruyama approximation.
    for n in range(n-1):

        y[n + 1] = (y[n] + (b[n] - d[n]) * y[n] * dt
                    + np.sqrt(b[n] + d[n]) * np.sqrt(y[n]) * (w1[n+1] - w1[n]))
        b[n + 1] = b[n] + beta1 * (be - b[n]) * dt + alpha1 * (w2[n+1] - w2[n])
        d[n + 1] = d[n] + beta2 * (de - d[n]) * dt + alpha2 * (w3[n+1] - w3[n])

    return y


def milstein_scheme(dt, time, seed=42, be=1, de=1.4, beta1=1, beta2=1, alpha1=0.5, alpha2=0.5):
    """
    Milstein scheme approximation.
    :param dt: Time step.
    :param time: Final time.
    :param seed: Random generation seed.
    :param be: Formula constant.
    :param de: Formula constant.
    :param beta1: Formula constant.
    :param beta2: Formula constant.
    :param alpha1: Formula constant.
    :param alpha2: Formula constant.
    :return: The Milstein scheme approximations of the population size.
    """
    # Compute the number of time stamps.
    n = int(time / dt)

    # Initialize the computation arrays.
    y = np.empty(n)
    b = np.empty(n)
    d = np.empty(n)

    # Set the initial values.
    y[0] = 30
    b[0] = be
    d[0] = de

    # Compute the step difference compared to the reference time step.
    step = int(dt / ref_dt)

    # Take the corresponding Wiener process values.
    w = wiener(time, seed)
    w1 = w[0][:n * step:step]
    w2 = w[1][:n * step:step]
    w3 = w[2][:n * step:step]

    # Compute the Milstein scheme approximation.
    for n in range(n - 1):

        y[n + 1] = (y[n] + (b[n] - d[n]) * y[n] * dt
                    + np.sqrt(b[n] + d[n]) * np.sqrt(y[n]) * (w1[n+1] - w1[n])
                    + (b[n] + d[n]) / 4 * ((w1[n+1] - w1[n]) ** 2 - dt))  # Milstein term
        b[n + 1] = b[n] + beta1 * (be - b[n]) * dt + alpha1 * (w2[n+1] - w2[n])
        d[n + 1] = d[n] + beta2 * (de - d[n]) * dt + alpha2 * (w3[n+1] - w3[n])

    return y


def test_strong(func):
    """
    Test strong convergence of approximation scheme.
    :param func: Implemented method of the approximation scheme.
    :return: Plots the strong error for the approximation scheme.
    """
    n_ref = int(t/ref_dt)  # Compute the number of reference approximations.
    ys_ref = np.empty((num_samples, n_ref))  # Initialize an array for the reference values.
    for i in range(num_samples):  # Iterate over the number of samples.
        ys_ref[i] = func(ref_dt, t, i, alpha1=0, alpha2=0)  # Compute the reference values.

    strong_errors = np.empty(len(dt_values))  # Initialize an array for storing errors.
    for k, dt in enumerate(dt_values):  # Iterate over the solutions.
        n = int(t/dt)  # Compute the number of approximations in the solution.
        step = int(dt / ref_dt)  # Compute the step difference compared to the reference time step.
        errors = np.empty(num_samples)  # Initialize an array for storing the global errors.
        for i in range(num_samples):  # Iterate over the number of samples.
            val = func(dt, t, i, alpha1=0, alpha2=0)  # Compute the solution values.
            ref = (ys_ref[i][:n * step:step])  # Take the corresponding reference solution values.
            errors[i] = np.mean(np.abs(val-ref))  # Compute the global error.
        strong_errors[k] = np.max(errors)  # Take the max error value.

    return strong_errors


def test_weak(func):
    """
    Test weak convergence of approximation scheme.
    :param func: Implemented method of the approximation scheme.
    :return: Plots the weak error for the approximation scheme.
    """
    n_ref = int(t/ref_dt)  # Compute the number of reference approximations.
    ys_ref = np.zeros(n_ref)  # Initialize an array for the reference values.
    for i in range(num_samples):  # Iterate for the number of samples.
        ys_ref += func(ref_dt, t, i, alpha1=0, alpha2=0)  # Compute the reference values.
    ys_ref /= num_samples  # Take the average of the reference solution values.

    weak_errors = np.empty(len(dt_values))  # Initialize an array for storing errors.
    for k, dt in enumerate(dt_values):  # Iterate over the solutions.
        n = int(t / dt)  # Compute the number of approximations in the solution.
        step = int(dt / ref_dt)  # Compute the step difference compared to the reference time step.
        ys = np.zeros(n)  # Initialize an array for solution values.
        for i in range(num_samples):  # Iterate over the number of samples.
            ys += func(dt, t, i, alpha1=0, alpha2=0)  # Compute the solution values.
        ys /= num_samples  # Take the average of the solution values.
        ref = ys_ref[:n*step:step]  # Take the corresponding reference solution values.
        weak_errors[k] = np.max(np.abs(ys-ref))  # Take the max error value.

    return weak_errors


def plot_distribution(dt, times, samples):
    """
    Create plot of the population distribution at different time stamps using Euler-Maruyama.
    :param dt: Time step for Euler-Maruyama.
    :param times: Times at which the population distribution is analysed.
    :param samples: Number of samples used to generate the distribution.
    :return: Plots the population distributions of Euler-Maruyama approximations at the specified time stamps.
    """
    time = max(times)  # Take the max time to use as final time.
    n = int(time/dt)  # Compute the number of time stamps.

    # Initialize arrays for the population values.
    pop1 = np.zeros((samples, n))
    pop2 = np.zeros((samples, n))

    # Compute the population values for the specified number of samples.
    for i in range(samples):
        pop1[i, :] = euler_maruyama(dt, time, i, alpha1=0.5, alpha2=0.5, beta1=1, beta2=1)
        pop2[i, :] = euler_maruyama(dt, time, i, alpha1=0, alpha2=0, beta1=0, beta2=0)

    index = [int(time / dt) - 1 for time in times]  # Index the required time frames.

    plt.figure(figsize=(12, 6))  # Generate the plot figure.

    # Plot the distributions of the two populations for each desired time frame.
    for i, time in enumerate(times):
        t1 = pop1[:, index[i]]
        t2 = pop2[:, index[i]]

        data_min = min(np.min(t1), np.min(t2))
        data_max = max(np.max(t1), np.max(t2))
        bin_width = (data_max - data_min) / 20
        bins = np.arange(data_min, data_max + bin_width, bin_width)

        plt.subplot(1, 3, i + 1)
        plt.hist(t1, bins=bins, alpha=0.5, label='With Noise')
        plt.hist(t2, bins=bins, alpha=0.5, label='Without Noise')
        plt.title(f'Population Distribution at T = {time}')
        plt.xlabel('Population')
        plt.ylabel('Frequency')
        plt.legend()

    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    # Convergence Test Variables
    t = 1
    ref_dt = 0.0001
    dt_values = [0.001, 0.005, 0.01, 0.02]
    num_samples = 100

    # 1) Test the Euler-Maruyama convergence
    strong_euler = test_strong(euler_maruyama)
    weak_euler = test_weak(euler_maruyama)

    # Plot the Euler-Maruyama errors
    plt.plot(dt_values, strong_euler, 'o-', label='Strong')
    plt.plot(dt_values, weak_euler, 'o-', label='Weak')
    plt.xlabel('Time Step')
    plt.ylabel('Absolute Error')
    plt.title(f'Absolute Error versus Time Step for Euler-Maruyama')
    plt.grid(True)
    plt.legend()
    plt.show()

    # 2) Plot the Euler-Maruyama distribution
    plot_distribution(0.001, [0.5, 1, 2], 100)

    # 3) Test the Milstein Scheme convergence
    strong_milstein = test_strong(milstein_scheme)
    weak_milstein = test_weak(milstein_scheme)

    # Plot the Milstein Scheme errors
    plt.plot(dt_values, strong_milstein, 'o-', label='Strong')
    plt.plot(dt_values, weak_milstein, 'o-', label='Weak')
    plt.xlabel('Time Step')
    plt.ylabel('Absolute Error')
    plt.title(f'Absolute Error versus Time Step for Milstein Scheme')
    plt.grid(True)
    plt.legend()
    plt.show()

    # Plot the strong and weak errors for Euler-Maruayama and Milstein Scheme together
    plt.plot(dt_values, weak_euler, 'o-', label='Weak Error Euler-Maruyama')
    plt.plot(dt_values, weak_milstein, 'o-', label='Weak Error Milstein Scheme')
    plt.plot(dt_values, strong_euler, 'o-', label='Strong Error Euler-Maruyama')
    plt.plot(dt_values, strong_milstein, 'o-', label='Strong Error Milstein Scheme')
    plt.title(f'Absolute Error versus Time Step')
    plt.xlabel('Time Step')
    plt.ylabel('Absolute Error')
    plt.grid(True)
    plt.legend()
    plt.show()
