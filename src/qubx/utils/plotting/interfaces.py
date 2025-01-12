from typing import List, Tuple
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

    def get_plot_data(self, plot_name: str) -> Tuple[str, dict]:
        """
        Get the data for the specified plot.
        """
        ...
