from typing import Tuple, List, Optional
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.widgets import Slider
from matplotlib.colors import ListedColormap
from argparse import ArgumentParser


def fprop(x, w0, w1):
    b: float = 0.0
    y_hat = 1 / (1 + np.exp(-(w0 * x[:, 0] + w1 * x[:, 1] + b)))
    return y_hat


def loss_fn(x, y, w0, w1):
    n: int = x.shape[0]
    y_hat = fprop(x, w0, w1)
    y_hat = np.maximum(y_hat, 0.0000001)
    y_hat = np.minimum(y_hat, 0.9999999)
    loss = -np.sum(y * np.log(y_hat) + (1-y) * np.log(1-y_hat))/n
    return loss


def get_data(rnd_seed: Optional[int]):
    if rnd_seed is not None:
        np.random.seed(rnd_seed)

    num_pos: int = 200
    num_neg: int = 200

    gauss_std: float = 0.5
    x_pos = np.random.multivariate_normal([-1, 1], [[gauss_std, 0], [0, gauss_std]], num_pos)
    x_neg = np.random.multivariate_normal([1, -1], [[gauss_std, 0], [0, gauss_std]], num_neg)
    # x_neg = np.concatenate((x_neg, np.random.multivariate_normal([0, 5], [[0.01, 0], [0, 0.01]], 20)))
    x = np.concatenate((x_pos, x_neg), axis=0)
    y = np.array([1] * num_pos + [0] * num_neg)

    return x, y


def train_params(x, y, w0_init: float, w1_init: float, lr: float, steps: int) -> Tuple[List[float], List[float]]:
    n: int = y.shape[0]
    w0: float = w0_init
    w1: float = w1_init
    w0_l: List[float] = [w0]
    w1_l: List[float] = [w1]
    for i in range(steps):
        y_hat = fprop(x, w0, w1)
        w0_grad = np.sum((y_hat - y) * x[:, 0]) / n
        w1_grad = np.sum((y_hat - y) * x[:, 1]) / n

        w0 = w0 - lr * w0_grad
        w1 = w1 - lr * w1_grad
        w0_l.append(w0)
        w1_l.append(w1)

    return w0_l, w1_l


def update_decision_boundary(ax, x, y, x0_mesh, x1_mesh, mesh_points, mesh_size, w0, w1, b):
    ax.cla()
    x1_range = np.linspace(np.min(x), np.max(x), x0_mesh.shape[0])

    cm_decision = plt.cm.RdBu
    cm_bright = ListedColormap(['#FF0000', '#0000FF'])

    y_hat_mesh = fprop(mesh_points, w0, w1)
    y_hat_mesh = y_hat_mesh.reshape((mesh_size, mesh_size))

    ax.contourf(x0_mesh, x1_mesh, y_hat_mesh, cmap=cm_decision)

    ax.scatter(x[:, 0], x[:, 1], c=y, cmap=cm_bright)

    ax.plot(x1_range, (-w0 * x1_range - b) / (w1 + 0.000001), color='k', linewidth=5)

    ax.set_xlim([np.min(x[:, 0]), np.max(x[:, 0])])
    ax.set_ylim([np.min(x[:, 1]), np.max(x[:, 1])])
    ax.set_xlabel('x0')
    ax.set_ylabel('x1')


def main():
    parser: ArgumentParser = ArgumentParser()
    parser.add_argument('--lr', type=float, default=0.1, help="Learning rate")
    parser.add_argument('--w0_init', type=float, default=0.0, help="Initial value for w0")
    parser.add_argument('--w1_init', type=float, default=0.0, help="Initial value for w1")
    parser.add_argument('--steps', type=int, default=100, help="Number of training steps")
    args = parser.parse_args()

    # Get data
    x, y = get_data(rnd_seed=42)

    # train parameters
    w0_l, w1_l = train_params(x, y, args.w0_init, args.w1_init, args.lr, args.steps)

    # plot
    w0_vals = np.linspace(-6, 6, 100)
    w1_vals = np.linspace(-6, 6, 100)

    w0_mesh, w1_mesh = np.meshgrid(w0_vals, w1_vals)
    errs_mesh = np.zeros(w0_mesh.shape)

    for idx1 in range(w0_mesh.shape[0]):
        for idx2 in range(w0_mesh.shape[1]):
            w0 = w0_mesh[idx1, idx2]
            w1 = w1_mesh[idx1, idx2]
            err = loss_fn(x, y, w0, w1)
            errs_mesh[idx1, idx2] = err

    fig, axs = plt.subplots(1, 2)
    for ax in axs:
        ax.set(adjustable='box')
        # ax.set_xticks([])
        # ax.set_yticks([])

    axs[1].set_xlabel('w0')
    axs[1].set_ylabel('w1')
    loss_val: float = loss_fn(x, y, w0_l[0], w1_l[1])
    axs[1].set_title(f'Loss Surface ({loss_val:.2E})')

    mesh_size: int = 100
    x0_vals_contour = np.linspace(np.min(x[:, 0]), np.max(x[:, 0]), mesh_size)
    x1_vals_contour = np.linspace(np.min(x[:, 1]), np.max(x[:, 1]), mesh_size)

    x0_mesh, x1_mesh = np.meshgrid(x0_vals_contour, x1_vals_contour)
    mesh_points = np.stack((x0_mesh.reshape(mesh_size*mesh_size), x1_mesh.reshape(mesh_size*mesh_size)), axis=1)

    cp = axs[1].contourf(w0_mesh, w1_mesh, errs_mesh)
    fig.colorbar(cp)

    # slider
    axstep = fig.add_axes([0.25, 0.01, 0.65, 0.03])
    step_slider = Slider(
        ax=axstep,
        label='Training Step',
        valmin=0,
        valmax=args.steps,
        valinit=0,
        valstep=1,
    )

    # initial line
    update_decision_boundary(axs[0], x, y, x0_mesh, x1_mesh, mesh_points, mesh_size, w0_l[0], w1_l[0], 0)
    line1, = axs[1].plot(w0_l[0], w1_l[1], marker='o', color='k', markersize=10)

    def update(step_in: int):
        w0_itr: float = w0_l[int(step_in)]
        w1_itr: float = w1_l[int(step_in)]
        update_decision_boundary(axs[0], x, y, x0_mesh, x1_mesh, mesh_points, mesh_size, w0_itr, w1_itr, 0)

        line1.set_xdata(w0_itr)
        line1.set_ydata(w1_itr)

        loss_val_itr: float = loss_fn(x, y, w0_itr, w1_itr)
        axs[1].set_title(f'Loss Surface ({loss_val_itr:.2E})')

    step_slider.on_changed(update)

    plt.show()


if __name__ == "__main__":
    main()
