import pytest
import pandas as pd
import numpy as np
from datetime import datetime
from qubx.plotting.data import PlotStyle, SeriesData, SubplotData, PlotData

@pytest.fixture
def sample_ohlcv_data():
    dates = pd.date_range(start='2023-01-01', periods=3)
    return pd.DataFrame({
        'open': [100, 102, 101],
        'high': [103, 104, 103],
        'low': [99, 101, 100],
        'close': [102, 101, 102],
        'volume': [1000, 1200, 1100]
    }, index=dates)

@pytest.fixture
def sample_line_data():
    dates = pd.date_range(start='2023-01-01', periods=3)
    return pd.Series([100, 101, 102], index=dates)

def test_plot_style_initialization():
    style = PlotStyle(color='red', width=2.0)
    assert style.color == 'red'
    assert style.width == 2.0
    assert style.opacity == 1.0

def test_series_data_initialization(sample_line_data):
    series = SeriesData(
        data=sample_line_data,
        type='line',
        name='test_series',
        style=PlotStyle(color='blue')
    )
    assert series.name == 'test_series'
    assert series.type == 'line'
    assert series.style.color == 'blue'

def test_subplot_data_initialization(sample_line_data):
    series = SeriesData(data=sample_line_data, type='line', name='test')
    subplot = SubplotData(series=[series], height_ratio=0.7)
    assert subplot.height_ratio == 0.7
    assert len(subplot.series) == 1

def test_plot_data_candlestick_conversion(sample_ohlcv_data):
    series = SeriesData(data=sample_ohlcv_data, type='candlestick', name='BTCUSD')
    subplot = SubplotData(series=[series])
    plot = PlotData(main=subplot)
    
    result = plot.to_looking_glass()
    assert isinstance(result['master'], pd.DataFrame)
    assert result['studies'] is None

def test_plot_data_line_conversion(sample_line_data):
    series = SeriesData(
        data=sample_line_data,
        type='line',
        name='MA',
        style=PlotStyle(color='blue', width=2.0)
    )
    subplot = SubplotData(series=[series])
    plot = PlotData(main=subplot)
    
    result = plot.to_looking_glass()
    converted_series = result['master'][0]
    assert converted_series['type'] == 'line'
    assert converted_series['name'] == 'MA'
    assert converted_series['style']['color'] == 'blue'
    assert converted_series['style']['width'] == 2.0

def test_plot_data_with_studies(sample_ohlcv_data, sample_line_data):
    main_series = SeriesData(data=sample_ohlcv_data, type='candlestick', name='BTCUSD')
    study_series = SeriesData(data=sample_line_data, type='line', name='MA')
    
    main_subplot = SubplotData(series=[main_series])
    study_subplot = SubplotData(series=[study_series])
    
    plot = PlotData(
        main=main_subplot,
        studies={'MA Study': study_subplot}
    )
    
    result = plot.to_looking_glass()
    assert isinstance(result['master'], pd.DataFrame)
    assert 'MA Study' in result['studies']
    assert result['studies']['MA Study'][0]['type'] == 'line'
