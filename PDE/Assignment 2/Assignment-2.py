# Solution of 1D Poisson’s equation with FDM
# Kaloyan Yanchev (c) 2024
import numpy as np
import matplotlib.pyplot as plt

if __name__ == "__main__":

    def assignment2(n: int = 5) -> (float, float):

        # Create a grid that splits the [0, 3] range in n equal length intervals
        # n intervals => n+1 points
        xgrid = np.linspace(0.0, 3.0, n+1)

        # Length of single interval over [0, 3]
        h = 3 / n

        def func1(x):
            return 3*x - 2

        def func2(x):
            return x**2 + 3*x - 2

        # Compute values of f1 and f2 over the grid
        f1 = func1(xgrid)
        f2 = func2(xgrid)

        # Create a plot of values of f1 and f2 over the grid
        # plt.figure()
        # plt.plot(xgrid, f1, marker='o', label='f1', color='blue')
        # plt.plot(xgrid, f2, marker='o', label='f2', color='red')
        # plt.legend()
        # plt.title("Graphs of f1 and f2 for n=5")
        # plt.xlabel("x")
        # plt.ylabel("f(x)")
        # plt.grid(True)
        # plt.show()

        def u1(x):
            return x**2 - x**3/2 + 3*x/2 + 1

        def u2(x):
            return x**2 - x**3/2 - x**4/12 + 15*x/4 + 1

        # Compute values of u1 and u2 over the grid
        u1ex = u1(xgrid)
        u2ex = u2(xgrid)

        # Create a plot of values of u1 and u2 over the grid
        # plt.figure()
        # plt.plot(xgrid, u1ex, marker='o', label='u1', color='blue')
        # plt.plot(xgrid, u2ex, marker='o', label='u2', color='red')
        # plt.legend()
        # plt.title("Graphs of u1 and u2 for n=5")
        # plt.xlabel("x")
        # plt.ylabel("u(x)")
        # plt.grid(True)
        # plt.show()

        # Construct coefficient matrix A
        a = np.zeros((n-1, n-1))
        a = a + np.diag(np.ones((n-2)), 1) + np.diag(np.ones((n-2)), -1) + np.diag(np.ones((n-1)), 0) * (-2)
        a = a * (-1/h**2)

        # Create a plot of the structure of A
        # plt.spy(a, marker='o', color='green')
        # plt.title("Structure of Matrix A")
        # plt.grid(True)
        # plt.show()

        # Compute and print eigenvalues of A
        # vals = np.linalg.eigvals(a)
        # vals.sort()
        # print(vals)

        # Compute the vector values f1 and f2
        f1rhs = f1[1:-1]
        f1rhs[0] = f1rhs[0] + 1 / h ** 2
        f1rhs[-1] = f1rhs[-1] + 1 / h ** 2
        f2rhs = f2[1:-1]
        f2rhs[0] = f2rhs[0] + 1 / h ** 2
        f2rhs[-1] = f2rhs[-1] + 1 / h ** 2

        # Solve equation Au=f
        u1 = np.linalg.solve(a, f1rhs)
        u2 = np.linalg.solve(a, f2rhs)

        # Insert boundary points into solution
        u1 = np.insert(u1, 0, 1)
        u1 = np.append(u1, 1)
        u2 = np.insert(u2, 0, 1)
        u2 = np.append(u2, 1)

        # Create a plot of u1, u2 and their approximations over the grid
        # plt.figure()
        # plt.plot(xgrid, u1, marker='o', label='u1', color='blue', linestyle='--')
        # plt.plot(xgrid, u2, marker='o', label='u2', color='red', linestyle='--')
        # plt.plot(xgrid, u1ex, marker='o', label='u1ex', color='blue')
        # plt.plot(xgrid, u2ex, marker='o', label='u2ex', color='red')
        # plt.legend()
        # plt.title("Graphs of u1, u2 and their approximations for n=5")
        # plt.xlabel("x")
        # plt.ylabel("u(x)")
        # plt.grid(True)
        # plt.show()

        # Compute the piece-wise difference of u and its approximation
        diff1 = u1ex - u1
        diff2 = u2ex - u2

        # Compute the RMSE global error
        err1 = np.sqrt(np.sum(diff1 ** 2) / (n - 1))
        err2 = np.sqrt(np.sum(diff2 ** 2) / (n - 1))

        # Print global error values
        # print(err1)
        # print(err2)

        return err1, err2

    # Execute FD approximation with n = 5
    # assignment2()

    # Variables for storing global error convergence
    # index = []
    # gl1 = []
    # gl2 = []
    #
    # # Iterate n from 5 to 1000 with step 5
    # for i in range(5, 1001, 5):
    #     er1, er2 = assignment2(i)
    #     index.append(i)
    #     gl1.append(er1)
    #     gl2.append(er2)
    #
    # # Plot global error convergence
    # plt.figure()
    # plt.loglog(index, gl1, label='global_error(u1)', color='blue')
    # plt.loglog(index, gl2, label='global_error(u2)', color='red')
    # plt.legend()
    # plt.title("Global Errors of u1 and u2")
    # plt.xlabel("n")
    # plt.ylabel("global_error")
    # plt.show()
