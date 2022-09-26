"""
Author: Thomas Tendron

This file implements a GUI in Python. For now just a basic test.
"""
import tkinter as tk
import yfinance as yf
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.figure import Figure
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from sklearn.linear_model import LinearRegression as LR


class App(tk.Tk):
    def __init__(self):
        tk.Tk.__init__(self)
        self.title("Financial Data Exploratory Analysis Dashboard")

        self.s_selected = tk.StringVar()
        self.p_selected = tk.StringVar()
        self.v_selected = tk.StringVar()

        self.stock_selected = tk.Entry(self, textvariable=self.s_selected)
        self.stock_selected.grid(row=0, column=0)

        self.period_selected = tk.Entry(self, textvariable=self.p_selected)
        self.period_selected.grid(row=0, column=1)

        self.variable_selected = tk.Entry(self, textvariable=self.v_selected)
        self.variable_selected.grid(row=0, column=2)

        self.plot_stock_close = tk.Button(self, text='Plot closing price series', command=self.plot_close) 
        self.plot_stock_close.grid(row=0, column=3)

    def plot_close(self):
        """
        improvement: dropdown menu with what to plot
        """
        stock = self.s_selected.get()
        period = self.p_selected.get()
        variable = self.v_selected.get()
        # Get data from Yahoo finance
        s = yf.Ticker(stock)
        data = s.history(period=period)
        # plot
        fig, ax = plt.subplots(1,1)
        ax.plot(data.index, data[variable])
        """
        Train on the first half of the perido and predict the second half, plot to compare with actual
        First example of a backtest
        """

        # Plot mean
        # ax.plot(data.index, [data[variable].mean() for _ in range(len(data.index))])
        # ax.label()
        canvas = FigureCanvasTkAgg(fig, master=self)
        canvas.draw()
        canvas.get_tk_widget().grid(row=1,column=1, columnspan=3)

if __name__ == "__main__":
    app = App()
    app.mainloop()