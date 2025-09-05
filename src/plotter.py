# Author: Berkan Mertan
# Copyright (c) 2025 Berkan Mertan. All rights reserved.
# Simple plotting class to avoid redundant matplotlib code

import matplotlib.pyplot as plt

class Plotter():
    def __init__(
            self, 
            xFields=[[]],
            yFields=[[]],
            xLabel="X Axis",
            yLabel="Y Axis",
            title="Classical Physics Plot"
    ):
        self._x = xFields
        self._y = yFields
        self._xlabel = xLabel
        self._ylabel = yLabel
        self._title = title
    
    def plot(self):
        """
        Convenient plotting function, invoke to plot x and y fields.
        """
        plt.figure(figsize=(10, 6))

        label_index=1
        for xField, yField in zip(self._x, self._y):
            plt.plot(xField, yField, label=f"Plot #{label_index}")
            label_index+=1
        
        plt.title(self._title)
        plt.xlabel(self._xlabel)
        plt.ylabel(self._ylabel)
        plt.legend()
        plt.grid(True)
        plt.show()

    @property
    def xFields(self):
        return self._x
    
    @xFields.setter
    def xField(self, xFields):
        self._x = xFields

    @property
    def yFields(self):
        return self._y

    @yFields.setter
    def yField(self, yFields):
        self._y = yFields
    
