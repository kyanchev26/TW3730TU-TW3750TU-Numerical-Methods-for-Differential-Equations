import numpy as np
import matplotlib.pyplot as plt
import scipy.sparse as sp
from matplotlib import animation

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

        def sourcefunc(x, y, t):
            return np.sin(4*np.pi*t) * (np.exp(-alpha * (x - x1) ** 2 - alpha * (y - y1) ** 2) + np.exp(-alpha * (x - x2) ** 2 - alpha * (y - y2) ** 2) +
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

        A = create2DLFVM(k_grid)
        x, y = grid()

        def stepWave(u0Start,u1Start,tStart,dt):
            f = sourcefunc(x, y, tStart+dt).transpose()
            fLX = np.reshape(f, ((nx - 1) * (ny - 1), 1))
            return 2*u1Start - u0Start + dt**2*(-A.dot(u1Start)+fLX)

        ht = np.sqrt(1/(1/hx**2+1/hy**2))
        nt = int(np.ceil(4 / ht))
        print(f"According to derived CFL bound, ht = {np.round(ht, 3)}, Nt = {nt} for T = 4.")

        plt.ion()
        fig = plt.figure(3)
        plt.clf()
        t = 0
        prev = np.zeros(((nx - 1) * (ny - 1), 1))
        cur = np.copy(prev)
        next = np.copy(cur)

        # figure initialization
        u3arr = np.reshape(next, (ny - 1, nx - 1, 1))
        img = plt.imshow(u3arr, extent=(hx / 2, Lx - hx / 2, Ly - hy / 2, hy / 2), interpolation='none')
        plt.gca().invert_yaxis()
        plt.colorbar(img, orientation='horizontal')
        tlt = plt.title(r"$u(x,y,t), t =$ " + str(np.round(t, 3)))

        def animate(frame):
            global t, prev, cur, next
            t = frame * ht
            if frame == 0:
                prev = np.zeros(((nx - 1) * (ny - 1), 1))
                cur = np.copy(prev)
            next = np.copy(stepWave(prev, cur, t, ht))
            prev = np.copy(cur)
            cur = np.copy(next)
            img.set_array(np.reshape(next, (ny - 1, nx - 1, 1)))
            tlt.set_text(r"$u(x,y,t), t =$ " + str(np.round(t + 2 * ht, 3)))
            img.set_clim(next.min(), next.max())
            return img

        anim = animation.FuncAnimation(fig, animate, nt-1, repeat=False)
        anim.save("animation.gif")

        u0 = np.zeros(((nx - 1) * (ny - 1), 1))
        u1 = np.copy(u0)
        u2 = np.copy(u1)

        for t in range(0, nt-1):
            u2 = stepWave(u0, u1, t * ht, ht)
            u0 = np.copy(u1)
            u1 = np.copy(u2)

        uEnd = np.reshape(u2, [ny - 1, nx - 1])
        vis_arr(uEnd, x, y, r"$u(x,y,t), t =$ " + str(np.round((t+2)*ht, 3)))

    assignment4(200, 100)
