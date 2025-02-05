import numpy as np
import matplotlib.pyplot as plt
import scipy.sparse as sp
import scipy.sparse.linalg as la

if __name__ == "__main__":

    LeftX = 0.0
    RightX = 10.0
    LeftY = 0.0
    RightY = 5.0

    def assignment3(Nx: int, Ny: int):

        dx = (RightX - LeftX) / Nx  # grid step in x-direction
        dy = (RightY - LeftY) / Ny  # grid step in y-direction

        def FDLaplacian2D(nx: int, ny: int, vis: bool = False, out: bool = False):

            hx = (RightX - LeftX) / nx  # grid step in x-direction
            hy = (RightY - LeftY) / ny  # grid step in y-direction

            Dx = sp.diags([-1, 1], (-1, 0), (nx, nx-1)) / hx
            Dy = sp.diags([-1, 1], (-1, 0), (ny, ny-1)) / hy

            DxT = Dx.transpose()
            DyT = Dy.transpose()

            Lxx = DxT.dot(Dx)
            Lyy = DyT.dot(Dy)

            Ix = sp.eye(nx-1)
            Iy = sp.eye(ny-1)

            A = sp.kron(Iy, Lxx) + sp.kron(Lyy, Ix)

            if vis:
                plt.spy(A)
                plt.grid(True)
                plt.title(f"Structure of A [{A.shape[0]}x{A.shape[1]}]")
                plt.show()

            if out:
                print(A)

            return A

        FDLaplacian2D(4, 4, True, True)

        def sourcefunc(x, y):
            f = 0.0
            alpha = 40.0
            x = np.array(x)
            y = np.array(y)
            for i in range(1, 10):
                for j in range(1, 5):
                    f += np.exp(-alpha*(x-i)**2-alpha*(y-j)**2)
            return f

        def grid() -> (np.array, np.array):
            x, y = np.mgrid[1:Nx, 1:Ny]
            x = x.astype('float64')
            x *= dx
            y = y.astype('float64')
            y *= dy

            return x, y

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
            plt.xticks(range(19, Nx-1, 20), np.round(x_val[19::20, 0], 2))
            plt.yticks(range(19, Ny-1, 20), np.round(y_val[0, 19::20], 2))
            plt.show()

        vis_arr(f, x, y, "Visualization of f")

        # lexicographic source vector
        fLX = np.reshape(f, ((Nx-1) * (Ny-1)))

        # 2D FD Laplacian on rectangular domain
        A = FDLaplacian2D(Nx, Ny)

        u = la.spsolve(A, fLX)

        # reshaping the solution vector into 2D array
        uArr = np.reshape(u, (Ny-1, Nx-1))

        vis_arr(uArr, x, y, "Visualization of FDM approximated solution u")

        def coeffK1(x, y):
            x = np.array(x)
            K = np.ones_like(x)
            return K

        def coeffK2(x, y):
            x = np.array(x)
            y = np.array(y)
            K = 1 + 0.1 * (x + y + x*y)
            return K

        fig, (ax1, ax2) = plt.subplots(2, 1)
        ax1.imshow(coeffK1(x, y).transpose())
        ax1.set_ylim(ax1.get_ylim()[::-1])
        ax1.set_title("Coeff1")
        plt.sca(ax1)
        plt.xticks(range(19, Nx - 1, 20), np.round(x[19::20, 0], 2))
        plt.yticks(range(19, Ny - 1, 20), np.round(y[0, 19::20], 2))
        im = ax2.imshow(coeffK2(x, y).transpose())
        ax2.set_ylim(ax2.get_ylim()[::-1])
        ax2.set_title("Coeff2")
        cbar_ax = fig.add_axes([0.85, 0.15, 0.05, 0.7])
        plt.colorbar(im, cax=cbar_ax)
        plt.sca(ax2)
        plt.xticks(range(19, Nx - 1, 20), np.round(x[19::20, 0], 2))
        plt.yticks(range(19, Ny - 1, 20), np.round(y[0, 19::20], 2))
        fig.subplots_adjust(hspace=0.5)
        plt.show()

        def create2DLFVM(nx: int, ny: int, coeffFun, out: bool = False):
            hx = (RightX - LeftX) / nx  # grid step in x-direction
            hy = (RightY - LeftY) / ny  # grid step in y-direction

            kx = (RightX - LeftX) / (2*nx)
            ky = (RightY - LeftY) / (2*ny)

            x, y = np.mgrid[1:2*nx, 1:2*ny]
            x = x.astype('float64')
            x *= kx
            y = y.astype('float64')
            y *= ky

            k = coeffFun(x, y)
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

            if out:
                print(A)

            return A

        create2DLFVM(4, 4, coeffK1, out=True)
        create2DLFVM(4, 4, coeffK2, out=True)

        A_K1 = create2DLFVM(Nx, Ny, coeffK1)
        A_K2 = create2DLFVM(Nx, Ny, coeffK2)

        u_K1 = la.spsolve(A_K1, fLX)
        u_K2 = la.spsolve(A_K2, fLX)

        uArr_K1 = np.reshape(u_K1, (Ny - 1, Nx - 1))
        uArr_K2 = np.reshape(u_K2, (Ny - 1, Nx - 1))

        vis_arr(uArr_K1, x, y, "Visualization of FVM approximated solution u with K1")
        vis_arr(uArr_K2, x, y, "Visualization of FVM approximated solution u with K2")

        vis_arr(uArr_K1 - uArr, x, y, "Difference of FDM and FVM (FVM-FDM)")

    assignment3(200, 100)
