from typing import List
from dataclasses import dataclass


@dataclass
class PlotData:
    master: dict
    studies: dict | None = None
    master_size: int = 3
    study_size: int = 1


class IPlotter:

    def get_plots(self) -> List[str]:
        """
        Get the list of plots that this object can generate.
        """
        ...
    
    def get_plot_data(self, plot_name: str) -> PlotData:
        """
        Get the data for the specified plot.
        """
        ...
