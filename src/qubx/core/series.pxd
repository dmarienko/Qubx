import numpy as np
cimport numpy as np


cdef np.ndarray nans(int dims)


cdef class Indexed:
    cdef public list values
    cdef public float max_series_length
    cdef unsigned short _is_empty


cdef class TimeSeries:
    cdef public long long timeframe
    cdef public Indexed times
    cdef public Indexed values
    cdef float max_series_length
    cdef unsigned short _is_new_item
    cdef public str name
    cdef dict indicators        # it's used for indicators caching
    cdef list calculation_order # calculation order as list: [ (input_id, indicator_obj, indicator_id) ]
    cdef double _process_every_update
    cdef double _last_bar_update_value
    cdef long long _last_bar_update_time

    cdef _update_indicators(TimeSeries self, long long time, object value, short new_item_started)


cdef class Indicator(TimeSeries):
    cdef TimeSeries series
    cdef TimeSeries parent


cdef class RollingSum:
    """
    Rolling fast summator
    """
    cdef unsigned int period
    cdef np.ndarray __s
    cdef unsigned int __i
    cdef double rsum
    cdef unsigned short is_init_stage 

    cpdef double update(RollingSum self, double value, short new_item_started)


cdef class Bar:
    cdef public long long time
    cdef public double open
    cdef public double high
    cdef public double low
    cdef public double close
    cdef public double volume          # total volume (in quote asset)
    cdef public double bought_volume   # volume bought (in quote asset) if presented

    cpdef Bar update(Bar self, double price, double volume, double bought_volume=*)

    cpdef dict to_dict(Bar self, unsigned short skip_time=*)


cdef class OHLCV(TimeSeries):
    cdef public TimeSeries open
    cdef public TimeSeries high
    cdef public TimeSeries low
    cdef public TimeSeries close
    cdef public TimeSeries volume
    cdef public TimeSeries bvolume

    cpdef short update(OHLCV self, long long time, double price, double volume=*, double bvolume=*)

    cpdef short update_by_bar(OHLCV self, long long time, double open, double high, double low, double close, double vol_incr=*, double b_vol_incr=*)

    cpdef _update_indicators(OHLCV self, long long time, object value, short new_item_started)

    cpdef object append_data(
        OHLCV self, 
        np.ndarray times, 
        np.ndarray opens,
        np.ndarray highs, 
        np.ndarray lows,
        np.ndarray closes, 
        np.ndarray volumes,
        np.ndarray bvolumes
    )


cdef class Trade:
    cdef public long long time
    cdef public double price
    cdef public double size
    cdef public short taker
    cdef public long long trade_id


cdef class Quote:
    cdef public long long time
    cdef public double bid
    cdef public double ask
    cdef public double bid_size
    cdef public double ask_size

    cpdef double mid_price(Quote self)
