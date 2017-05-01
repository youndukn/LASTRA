import tkinter as tk
from tkinter import ttk

import matplotlib.pyplot as plt
import matplotlib
matplotlib.use("TkAgg")
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg, NavigationToolbar2TkAgg

import numpy as np

import matplotlib.animation as animation

LARGE_FONT = ("Verdana", 12)

class MatrixPage(tk.Frame):
    def __init__(self, parent, controller, index):
        self.index = index
        tk.Frame.__init__(self, parent)
        label = tk.Label(self, text="Start Page", font=LARGE_FONT)
        label.pack(pady=10, padx=10)

        conf_arr = [[33, 2, 0, 0, 0, 0, 0, 0, 0, 1, 3],
                    [3, 31, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                    [0, 4, 41, 0, 0, 0, 0, 0, 0, 0, 1],
                    [0, 1, 0, 30, 0, 6, 0, 0, 0, 0, 1],
                    [0, 0, 0, 0, 38, 10, 0, 0, 0, 0, 0],
                    [0, 0, 0, 3, 1, 39, 0, 0, 0, 0, 4],
                    [0, 2, 2, 0, 4, 1, 31, 0, 0, 0, 2],
                    [0, 1, 0, 0, 0, 0, 0, 36, 0, 2, 0],
                    [0, 0, 0, 0, 0, 0, 1, 5, 37, 5, 1],
                    [3, 0, 0, 0, 0, 0, 0, 0, 0, 39, 0],
                    [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 38]]

        self.fig = plt.figure()
        self.ax = self.fig.add_subplot(1, 1, 1)
        self.ax.set_aspect('equal')

        self.ax.imshow(np.array(conf_arr))

        self.canvas = FigureCanvasTkAgg(self.fig, self)
        self.canvas.show()
        self.canvas.get_tk_widget().pack(side=tk.TOP, fill=tk.BOTH, expand=True)


        button1 = ttk.Button(self, text="Pre",
                            command=self.increase)
        button1.pack()

        button1 = ttk.Button(self, text="Nxt",
                            command=lambda: controller.show_frame(self.index+1))
        button1.pack()


    def increase(self):
        self.index += 1
        print(self.index)
        self.animate(self.index)

    def animate(self, i):
        conf_arr = [[33+i, 2+i, 0+i, 0+i, 0, 0, 0, 0, 0, 1, 3],
                    [3, 31, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                    [0, 4, 41, 0, 0, 0, 0, 0, 0, 0, 1],
                    [0, 1, 0, 30, 0, 6, 0, 0, 0, 0, 1],
                    [0, 0, 0, 0, 38, 10, 0, 0, 0, 0, 0],
                    [0, 0, 0, 3, 1, 39, 0, 0, 0, 0, 4],
                    [0, 2, 2, 0, 4, 1, 31, 0, 0, 0, 2],
                    [0, 1, 0, 0, 0, 0, 0, 36, 0, 2, 0],
                    [0, 0, 0, 0, 0, 0, 1, 5, 37, 5, 1],
                    [3, 0, 0, 0, 0, 0, 0, 0, 0, 39, 0],
                    [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 38]]

        self.ax.clear()
        self.ax.imshow(np.array(conf_arr))