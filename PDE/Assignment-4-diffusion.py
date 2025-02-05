import time

import numpy as np
import matplotlib.pyplot as plt
import scipy.sparse as sp
import scipy.sparse.linalg as la

if __name__ == "__main__":

    Lx = 10.0
    Ly = 5.0

    x1 = 0.25 * Lx
    y1 = 0.25 * Ly
    x2 = 0.25 * Lx
    y2 = 0.75 * Ly
    x3 = 0.75 * Lx
    y3 = 0.75 * Ly
    x4 = 0.75 * Lx
    y4 = 0.25 * Ly

    alpha = 40

    def assignment4(nx: int, ny: int):

        hx = Lx / nx  # grid step in x-direction
        hy = Ly / ny  # grid step in y-direction

        def create2DLFVM(coeffK):

            kx = hx / 2
            ky = hy / 2

            x, y = np.mgrid[1:2*nx, 1:2*ny]
            x = x.astype('float64')
            x *= kx
            y = y.astype('float64')
            y *= ky

            k = coeffK(x, y)
            main_d = []
            x_d = []
            y_d = []

            for j in range(1, ny):
                for i in range(1, nx):
                    main_d.append(k[2*i-2, 2*j-1]/hx**2 + k[2*i-1, 2*j-2]/hy**2 + k[2*i, 2*j-1]/hx**2 + k[2*i-1, 2*j]/hy**2)
                    if i < nx-1:
                        x_d.append(-k[2*i, 2*j-1]/hx**2)
                    else:
                        x_d.append(0)
                    y_d.append(-k[2*i-1, 2*j]/hy**2)

            A = sp.diags([y_d, x_d, main_d, x_d, y_d], [-2*ny+1, -1, 0, 1, 2*ny-1], ((nx-1)*(ny-1), (nx-1)*(ny-1)), format='csc')

            return A

        def sourcefunc(x, y):
            return (np.exp(-alpha * (x - x1) ** 2 - alpha * (y - y1) ** 2) + np.exp(-alpha * (x - x2) ** 2 - alpha * (y - y2) ** 2) +
                    np.exp(-alpha * (x - x3) ** 2 - alpha * (y - y3) ** 2) + np.exp(-alpha * (x - x4) ** 2 - alpha * (y - y4) ** 2))

        def k_point(x, y):
            if x < Lx/2 and y < Ly/2:
                return 0.1
            elif x < Lx/2 and y >= Ly/2:
                return 0.4
            elif x >= Lx/2 and y >= Ly/2:
                return 0.7
            elif x >= Lx/2 and y < Ly/2:
                return 1.0

        def k_grid(x,y):
            k = np.empty_like(x)
            for i in range(x.shape[0]):
                for j in range(x.shape[1]):
                    k[i, j] = k_point(x[i, j], y[i, j])
            return k

        def grid() -> (np.array, np.array):
            x, y = np.mgrid[1:nx, 1:ny]
            x = x.astype('float64')
            x *= hx
            y = y.astype('float64')
            y *= hy

            return np.array(x), np.array(y)

        x, y = grid()
        f = sourcefunc(x, y).transpose()

        def vis_arr(func: np.ndarray, x_val: np.ndarray, y_val: np.ndarray, title: str):
            plt.ion()
            plt.figure(1)
            plt.clf()
            plt.imshow(func)
            plt.title(title)
            plt.colorbar()

            # Invert y-axis
            ax = plt.gca()
            ax.set_ylim(ax.get_ylim()[::-1])

            # Scale x,y ranges
            plt.xticks(range(19, nx-1, 20), np.round(x_val[19::20, 0], 2))
            plt.yticks(range(19, ny-1, 20), np.round(y_val[0, 19::20], 2))
            plt.show()

        vis_arr(f, x, y, "Visualization of f")
        fLX = np.reshape(f, ((nx - 1) * (ny - 1), 1))

        k = k_grid(x, y).transpose()
        vis_arr(k, x, y, "Visualization of k")

        def solveFE(uStart, tStart, tEnd, Nt):
            h = (tEnd - tStart)/Nt
            A = create2DLFVM(k_grid)
            I = sp.eye((nx - 1) * (ny - 1), (nx - 1) * (ny - 1), format='csc')
            k = I - h * A
            for i in range(0, Nt):
                uStart = k.dot(uStart) + h * fLX
            return uStart

        def solveTR(uStart, tStart, tEnd, Nt):
            h = (tEnd - tStart) / Nt
            A = create2DLFVM(k_grid)
            I = sp.eye((nx - 1) * (ny - 1), (nx - 1) * (ny - 1), format='csc')
            k = I - h / 2 * A
            k1 = I + h / 2 * A
            for i in range(0, Nt):
                uStart = la.spsolve(k1, (k.dot(uStart) + h * fLX)).reshape((nx-1)*(ny-1), 1)
            return uStart

        A = create2DLFVM(k_grid)

        nt = int(np.ceil(2 * (1 / hx**2 + 1 / hy**2)))
        print(f"Forward-Euler Number of Time Steps: {nt}")

        u0 = np.zeros((A.shape[0], 1), dtype=float)

        t0 = time.time()
        u_FE1 = solveFE(u0, 0, 1, nt)
        print(f"Forward-Euler {nt} Time Steps Execution Time: {time.time()-t0} s")
        vis_arr(np.reshape(u_FE1, (ny - 1, nx - 1)), x, y, f"Forward-Euler, T=1, {nt} time steps")

        t0 = time.time()
        u_FE2 = solveFE(u0, 0, 1, 2*nt)
        print(f"Forward-Euler {2*nt} Time Steps Execution Time: {time.time() - t0} s")
        vis_arr(np.reshape(u_FE2, (ny - 1, nx - 1)), x, y, f"Forward-Euler, T=1, {2*nt} time steps")

        def RMS(u_1, u_2):
            return np.sqrt(np.sum((u_1-u_2)**2)/(u_1.shape[0]*u_1.shape[1]))

        u_TR = np.empty((8, A.shape[0], 1))
        RMS_1 = np.empty((8, 1))
        RMS_2 = np.empty((8, 1))

        for i in range(0, 8):
            u_TR[i] = solveTR(u0, 0, 1, 2**i)
            RMS_1[i] = RMS(u_FE1, u_TR[i])
            RMS_2[i] = RMS(u_FE2, u_TR[i])
            print(RMS_1[i]/RMS_2[i])

        vis_arr(np.reshape(u_TR[7], (ny - 1, nx - 1)), x, y, f"Trapezoidal, T=1, {2**7} time steps")

        vis_arr(np.reshape(u_FE1-u_TR[7], (ny - 1, nx - 1)), x, y, f"Difference of Forward-Euler{nt} and Trapezoidal{2**7}")

        plt.semilogy(np.arange(0, 8), RMS_1, base=10, label=f'Forward-Euler {nt} steps')
        plt.semilogy(np.arange(0, 8), RMS_2, base=10, label=f'Forward-Euler {2*nt} steps')
        plt.title("RMS error between Forward-Euler and Trapezoidal Methods")
        plt.xlabel("x")
        plt.ylabel("RMS for Trapezoidal 2^x steps")
        plt.legend()
        plt.show()

    assignment4(200, 100)
