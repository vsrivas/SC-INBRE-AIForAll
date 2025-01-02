import torch
import torch.nn as nn
from torch import Tensor
import numpy as np
import pickle
import time
from argparse import ArgumentParser
from examples.train_mnist import get_nnet

import tkinter as tk
from PIL import Image, ImageGrab

from examples.train_mnist import train_nnet


line_id = None


def main():
    parser: ArgumentParser = ArgumentParser()
    parser.add_argument('--nnet', type=str, default=None, help="")
    args = parser.parse_args()

    # load nnet
    nnet = get_nnet()
    state_dict = torch.load(args.nnet)
    nnet.load_state_dict(state_dict)
    nnet.eval()

    # tk canvas
    root = tk.Tk()

    canvas = tk.Canvas()
    canvas.pack()

    # draw
    line_points = []
    line_options = {}

    def draw_line(event):
        global line_id
        line_points.extend((event.x, event.y))
        if line_id is not None:
            canvas.delete(line_id)
        line_id = canvas.create_line(line_points, **line_options)

    def set_start(event):
        line_points.extend((event.x, event.y))

    def end_line(event=None):
        global line_id
        line_points.clear()
        line_id = None
        x0 = canvas.winfo_rootx()
        y0 = canvas.winfo_rooty()
        x1 = x0 + canvas.winfo_width()
        y1 = y0 + canvas.winfo_height()

        im = ImageGrab.grab((x0, y0, x1, y1))
        im = im.convert("L")
        im = im.resize((28, 28))

    # Tk
    canvas.bind('<Button-1>', set_start)
    canvas.bind('<B1-Motion>', draw_line)
    canvas.bind('<ButtonRelease-1>', end_line)
    root.mainloop()


if __name__ == "__main__":
    main()
