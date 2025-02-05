import time
import numpy as np
import matplotlib.pyplot as plt
import scipy.sparse as sp
import scipy.sparse.linalg as la

if __name__ == "__main__":

    side = 4.0
    T = 20.0
    Du = 0.05
    Dv = 1.0
    k = 5.0
    a = 0.1305
    b = 0.7695
    e = 0.001

    def assignment5(N: int):

        np.random.seed(40)

        def FDLaplacian2D(N: int):

            h = side/N

            Dx = sp.diags([1, -1], (0, 1), (N - 2, N - 1))/h
            Dy = sp.diags([1, -1], (0, 1), (N - 2, N - 1))/h

            DxT = Dx.transpose()
            DyT = Dy.transpose()

            Lxx = DxT.dot(Dx)
            Lyy = DyT.dot(Dy)

            Ix = sp.eye(N - 1)
            Iy = sp.eye(N - 1)

            A = sp.kron(Iy, Lxx) + sp.kron(Lyy, Ix)

            return A

        u0 = np.random.rand((N-1)**2, 1) * 0.01 * (a + b) + a + b
        v0 = np.ones(((N-1)**2, 1)) * (b / (a + b)**2)

        def vis_arr(func: np.ndarray, title: str):
            plt.ion()
            plt.figure(1)
            plt.clf()
            plt.imshow(func)
            plt.title(title)
            plt.colorbar()

            ax = plt.gca()
            ax.set_ylim(ax.get_ylim()[::-1])

            plt.xticks([-2, 19, 39, 59, 79, 100], [0, 0.8, 1.6, 2.4, 3.2, 4])
            plt.yticks([-2, 19, 39, 59, 79, 100], [0, 0.8, 1.6, 2.4, 3.2, 4])

            plt.show()

        vis_arr(u0.reshape((N-1, N-1, 1)), "u, T=0")
        vis_arr(v0.reshape((N-1, N-1, 1)), "v, T=0")

        A = FDLaplacian2D(N)

        def solveFE(uStart, vStart, Nt):
            ht = T/Nt
            for i in range(0, Nt):
                uNew = uStart + ht * (- Du * A.dot(uStart) + k * (a - uStart + uStart * uStart * vStart))
                vNew = vStart + ht * (- Dv * A.dot(vStart) + k * (b - uStart * uStart * vStart))
                uStart = uNew.copy()
                vStart = vNew.copy()
            return uStart, vStart

        lam = 8 * (N / side) ** 2
        k_stability = 500.0
        FEt = int(T * (lam + k_stability) / 2)

        t0 = time.time()
        uF, vF = solveFE(u0, v0, FEt)

        print(f"Forward-Euler CPU Time: {time.time() - t0} s")
        print(f"Forward-Euler # Time Steps: {FEt}")

        vis_arr(uF.reshape((N - 1, N - 1, 1)), f"u, T={T}, Forward-Euler with {FEt} Time Steps")
        vis_arr(vF.reshape((N - 1, N - 1, 1)), f"v, T={T}, Forward-Euler with {FEt} Time Steps")

        def Jacobian(u, v, h):
            diag_uv = sp.diags([(u * v).ravel()], [0], ((N-1)**2, (N-1)**2), format='csc')
            diag_u2 = sp.diags([(u ** 2).ravel()], [0], ((N-1)**2, (N-1)**2), format='csc')

            J11 = -Du * A + k * (2 * diag_uv - sp.eye((N-1)**2))
            J12 = k * diag_u2
            J21 = -2 * k * diag_uv
            J22 = -Dv * A - k * diag_u2

            J = sp.bmat([[J11, J12], [J21, J22]], format='csc')
            return sp.eye(2*(N-1)**2)-h*J

        def Residual(uStart, u0, vStart, v0, h):
            Ru = u0 + h * (-Du * A.dot(uStart) + k * (a - uStart + uStart ** 2 * vStart)) - uStart
            Rv = v0 + h * (-Dv * A.dot(vStart) + k * (b - uStart ** 2 * vStart)) - vStart
            return np.concatenate([Ru, Rv])

        def solveBENR(uStart, vStart, Nt):
            ht = T / Nt
            for i in range(Nt):
                uOld = uStart.copy()
                vOld = vStart.copy()
                R = Residual(uStart, uOld, vStart, vOld, ht)
                error = np.linalg.norm(R)
                while error > e:
                    delta = la.spsolve(Jacobian(uStart, vStart, ht), R).reshape((2*(N-1)**2, 1))
                    uStart += delta[:(N-1)**2]
                    vStart += delta[(N-1)**2:]
                    R = Residual(uStart, uOld, vStart, vOld, ht)
                    error = np.linalg.norm(R)
                    print(f"Backward-Euler Iteration {i}: Residual norm {error}")
            return uStart, vStart

        BEt = 1000
        t0 = time.time()
        uB, vB = solveBENR(u0, v0, BEt)

        print(f"Backward-Euler CPU Time: {time.time() - t0} s")
        print(f"Backward-Euler # Time Steps: {BEt}")

        vis_arr(uB.reshape((N - 1, N - 1, 1)), f"u, T={T}, Backward-Euler Newton-Raphson with {BEt} Time Steps")
        vis_arr(vB.reshape((N - 1, N - 1, 1)), f"v, T={T}, Backward-Euler Newton-Raphson with {BEt} Time Steps")

    assignment5(100)