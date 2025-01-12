from dataclasses import dataclass, field
from typing import Literal, Dict, List, Optional, Union, Any
from datetime import datetime
import pandas as pd

PlotType = Literal["candlestick", "line", "scatter", "bar", "area", "signal", "execution", "track", "trend"]


@dataclass
class PlotStyle:
    """Visual properties of a plot"""

    color: Optional[str] = None
    width: float = 1.0
    opacity: float = 1.0
    dash: Optional[str] = None  # solid, dash, dot
    marker: Optional[str] = None  # triangle-up, triangle-down etc
    marker_size: float = 10.0
    fill_color: Optional[str] = None


@dataclass
class SeriesData:
    """Single data series"""

    data: Union[pd.Series, pd.DataFrame]
    type: PlotType
    name: str
    style: PlotStyle = field(default_factory=PlotStyle)
    hover_text: Optional[List[str]] = None


@dataclass
class SubplotData:
    """Group of series in one subplot"""

    series: List[SeriesData]
    height_ratio: float = 1.0
    y_axis_range: Optional[tuple[float, float]] = None
    show_legend: bool = True


@dataclass
class PlotData:
    """Complete plot specification"""

    main: SubplotData  # Main chart (usually price)
    studies: Dict[str, SubplotData] = field(default_factory=dict)  # Additional studies
    title: str = ""

    def to_looking_glass(self) -> Dict[str, Any]:
        """Convert to LookingGlass format"""
        return {
            "master": self._convert_subplot(self.main),
            "studies": (
                {name: self._convert_subplot(study) for name, study in self.studies.items()} if self.studies else None
            ),
        }

    def _convert_subplot(self, subplot: SubplotData) -> Union[pd.DataFrame, List[Any]]:
        """Convert SubplotData to LookingGlass format"""
        if not subplot.series:
            return pd.DataFrame()

        # If single OHLCV series, return as main dataframe
        if len(subplot.series) == 1 and subplot.series[0].type == "candlestick":
            return (
                subplot.series[0].data
                if isinstance(subplot.series[0].data, pd.DataFrame)
                else subplot.series[0].data.to_frame()
            )

        # Convert multiple series into list format
        converted = []
        for series in subplot.series:
            if series.type == "candlestick":
                converted.append(
                    {
                        "data": series.data,
                        "type": "candlestick",
                        "name": series.name,
                        "style": {
                            "increasing_line_color": series.style.color or "green",
                            "decreasing_line_color": series.style.color or "red",
                            "opacity": series.style.opacity,
                        },
                    }
                )

            elif series.type in ["line", "area", "step"]:
                converted.append(
                    {
                        "data": series.data,
                        "type": series.type,
                        "name": series.name,
                        "style": {
                            "color": series.style.color,
                            "width": series.style.width,
                            "dash": series.style.dash,
                            "fill": series.style.fill_color if series.type == "area" else None,
                            "opacity": series.style.opacity,
                        },
                    }
                )

            elif series.type in ["signal", "execution"]:
                converted.append(
                    {
                        "data": series.data,
                        "type": "scatter",
                        "name": series.name,
                        "style": {
                            "color": series.style.color,
                            "marker": series.style.marker or "circle",
                            "size": series.style.marker_size,
                            "opacity": series.style.opacity,
                        },
                        "hover_text": series.hover_text,
                    }
                )

            elif series.type == "trend":
                converted.append(
                    {
                        "data": series.data,
                        "type": "line",
                        "name": series.name,
                        "style": {
                            "color": series.style.color,
                            "width": series.style.width,
                            "dash": series.style.dash or "dash",
                            "opacity": series.style.opacity,
                        },
                    }
                )

        return converted
