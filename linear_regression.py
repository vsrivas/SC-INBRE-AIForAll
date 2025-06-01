from typing import List
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.widgets import Slider
from argparse import ArgumentParser
from utils.lin_reg_utils import get_data, train_params


def err_fn(x, y, w: float, b: float):
    n: int = x.shape[0]
    err = (np.sum((y - (w * x + b)) ** 2)) / (2 * n)
    return err


def main():
    parser: ArgumentParser = ArgumentParser()
    parser.add_argument('--lr', type=float, default=0.1, help="Learning rate")
    parser.add_argument('--slope', type=float, default=3.0, help="Line slope")
    parser.add_argument('--bias', type=float, default=0.0, help="Line bias")
    parser.add_argument('--w_init', type=float, default=0.0, help="Initial value for w")
    parser.add_argument('--b_init', type=float, default=0.0, help="Initial value for b")
    parser.add_argument('--no_bias', action='store_true', default=False, help="")
    parser.add_argument('--steps', type=int, default=100, help="Number of training steps")
    args = parser.parse_args()

    # Get data
    x, y = get_data(args.slope, args.bias)
    # y = np.sin(5*x) + np.random.normal(0, 0.5, size=x.shape)
    # w_vals = np.linspace(-6, 6, 100)

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
    axs[1].set_title('Loss Surface')
    w_vals = np.linspace(0, 6, 100)
    b_vals = np.linspace(0, 6, 100)
    if args.no_bias:
        axs[1].set_xlabel('w')
        axs[1].set_ylabel('L')

        errs: List[int] = []
        for w in w_vals:
            err = err_fn(x, y, w, 0.0)
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
                err = err_fn(x, y, w, b)
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
        line1, = axs[1].plot(w_l[0], err_fn(x, y, w_l[0], b_l[0]), marker='o', color='k', markersize=10)
    else:
        line1, = axs[1].plot(w_l[0], b_l[0], marker='o', color='k', markersize=10)

    def update(step_in: int):
        w_itr: float = w_l[int(step_in)]
        b_itr: float = b_l[int(step_in)]
        line0.set_ydata(w_itr * x + b_itr)
        line1.set_xdata(w_itr)
        if args.no_bias:
            line1.set_ydata(err_fn(x, y, w_itr, b_itr))
        else:
            line1.set_xdata(w_itr)
            line1.set_ydata(b_itr)

    step_slider.on_changed(update)

    plt.show()


if __name__ == "__main__":
    main()
