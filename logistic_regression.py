from typing import List, Optional
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.widgets import Slider
from argparse import ArgumentParser


def fprop(x, w0, w1):
    b: float = 0.0
    y_hat = 1 / (1 + np.exp(-(w0 * x[:, 0] + w1 * x[:, 1] + b)))
    return y_hat


def loss_fn(x, y, w0, w1):
    n: int = x.shape[0]
    y_hat = fprop(x, w0, w1)
    loss = -np.sum(y * np.log(y_hat) + (1-y) * np.log(1-y_hat))/n
    return loss


def get_data(rnd_seed: Optional[int]):
    if rnd_seed is not None:
        np.random.seed(rnd_seed)

    num_pos: int = 200
    num_neg: int = 200

    gauss_std: float = 2
    x_pos = np.random.multivariate_normal([-1, 1], [[gauss_std, 0], [0, gauss_std]], num_pos)
    x_neg = np.random.multivariate_normal([1, -1], [[gauss_std, 0], [0, gauss_std]], 180)
    x_neg = np.concatenate((x_neg, np.random.multivariate_normal([0, 5], [[0.01, 0], [0, 0.01]], 20)))
    x = np.concatenate((x_pos, x_neg), axis=0)
    y = np.array([1] * num_pos + [0] * num_neg)

    return x, y


def train_params(x, y, w0_init: float, w1_init: float, lr: float, steps: int):
    n: int = x.shape[0]
    w0: float = w0_init
    w1: float = w1_init
    w0_l: List[float] = [w0]
    w1_l: List[float] = [w1]
    for i in range(steps):
        y_hat = fprop(x, w0, w1)
        loss = loss_fn(x, y, w0, w1)

        w0_grad = np.sum((y_hat - y) * x[:, 0]) / num_tot
        w1_grad = np.sum((y_hat - y) * x[:, 1]) / num_tot
        b_grad = np.sum((y_hat - y)) / num_tot

        print(i, loss, w0_grad, w0, w1_grad, w1, b, b_grad)

        w0 = w0 - lr * w0_grad
        w1 = w1 - lr * w1_grad
        b = b - lr * b_grad

        plt.pause(0.5)
        plt.draw()


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
    w_l, b_l = train_params(x, y, args.w_init, args.b_init, args.lr, args.steps, args.no_bias)

    # plot
    fig, axs = plt.subplots(1, 2)
    for ax in axs:
        ax.set(adjustable='box')
        # ax.set_xticks([])
        # ax.set_yticks([])

    axs[0].set_xlabel('x')
    axs[0].set_ylabel('y')
    axs[0].set_title('Data')
    axs[0].scatter(x, y)

    # loss surface
    loss_val: float = loss_fn(x, y, args.w_init, args.b_init)
    axs[1].set_title(f'Loss Surface: ({loss_val:.2E})')
    w_vals = np.linspace(0, 6, 100)
    b_vals = np.linspace(0, 6, 100)
    if args.no_bias:
        axs[1].set_xlabel('w')
        axs[1].set_ylabel('L')

        errs: List[int] = []
        for w in w_vals:
            err = loss_fn(x, y, w, 0.0)
            errs.append(err)
        axs[1].plot(w_vals, errs)
    else:
        axs[1].set_xlabel('w')
        axs[1].set_ylabel('b')
        w_mesh, b_mesh = np.meshgrid(w_vals, b_vals)
        errs_mesh = np.zeros(w_mesh.shape)
        for idx1 in range(w_mesh.shape[0]):
            for idx2 in range(w_mesh.shape[1]):
                w = w_mesh[idx1, idx2]
                b = b_mesh[idx1, idx2]
                err = loss_fn(x, y, w, b)
                errs_mesh[idx1, idx2] = err
        cp = axs[1].contourf(w_mesh, b_mesh, errs_mesh)
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
    line0, = axs[0].plot(x, w_l[0] * x + b_l[0], color='k', linewidth=5)
    if args.no_bias:
        line1, = axs[1].plot(w_l[0], loss_fn(x, y, w_l[0], b_l[0]), marker='o', color='k', markersize=10)
    else:
        line1, = axs[1].plot(w_l[0], b_l[0], marker='o', color='k', markersize=10)

    def update(step_in: int):
        w_itr: float = w_l[int(step_in)]
        b_itr: float = b_l[int(step_in)]
        loss_val_itr: float = loss_fn(x, y, w_itr, b_itr)
        axs[1].set_title(f'Loss Surface ({loss_val_itr:.2E})')

        line0.set_ydata(w_itr * x + b_itr)
        line1.set_xdata(w_itr)
        if args.no_bias:
            line1.set_ydata(loss_val_itr)
        else:
            line1.set_xdata(w_itr)
            line1.set_ydata(b_itr)

    step_slider.on_changed(update)

    plt.show()


if __name__ == "__main__":
    main()
