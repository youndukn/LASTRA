import tkinter as tk
from gui.matrix_page import MatrixPage

from tkinter import ttk

class MainWindow(tk.Tk):
    def __init__(self, thread_numb, *args, **kwargs):

        tk.Tk.__init__(self, *args, **kwargs)
        tk.Tk.wm_title(self, "Deep Learn")
        container = tk.Frame(self)

        self.thread_numb = thread_numb

        container.pack(side="top", fill="both", expand = True)

        container.grid_rowconfigure(0, weight=1)
        container.grid_columnconfigure(0, weight=1)

        menubar = tk.Menu(container)
        filemenu = tk.Menu(menubar, tearoff=0)
        filemenu.add_command(label="Save", command = lambda : self.popupmsg("not supported"))
        filemenu.add_separator()
        filemenu.add_command(label="exit", command=quit)
        menubar.add_cascade(label="File", menu=filemenu)

        tk.Tk.config(self, menu=menubar)

        self.frames = {}

        for i in range(self.thread_numb):
            frame = MatrixPage(container, self, i)

            self.frames[i] = frame

            frame.grid(row=0, column=0, sticky="nsew")

        self.show_frame(0)

    def show_frame(self, i):
        if i < 0 or i >= self.thread_numb:
            return

        frame = self.frames[i]
        frame.tkraise()

    def popupmsg(self, msg):
        popup = tk.Tk()

        popup.wm_title("!")
        label = ttk.Label(popup, text=msg)
        label.pack()
        b1 = ttk.Button(popup, text="Okay", command=popup.destroy)
        b1.pack()
        popup.mainloop()