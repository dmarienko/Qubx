import threading
import time
from datetime import datetime, timedelta
from time import sleep

import ntplib
import numpy as np

from qubx import logger

NTP_SERVERS_LIST = ["time.windows.com", "pool.ntp.org", "europe.pool.ntp.org", "time.google.com"]

__CORRECT_INTERVAL = timedelta(seconds=30)
__SLEEP_CORRECT_THREAD = 10

_offset = None  # never use it explicitly but for tests! Always use get_offset()
_controlling_thread = None


def __correct_offset():
    global _offset
    ntp_client = ntplib.NTPClient()
    for ntp_url in NTP_SERVERS_LIST:
        try:
            response = ntp_client.request(ntp_url)
            _offset = response.offset
            return
        except Exception as e:
            logger.warning(f"{ntp_url} NTP server request exception: {e}")
    logger.error(f"Unable to get ntp offset from neither of NTP servers list {NTP_SERVERS_LIST}")


def __correct_offset_runnable():
    logger.debug("NTP offset controller thread is started")
    last_corrected_dt = None
    while True:
        # do correction every specified interval
        if last_corrected_dt is None or datetime.now() - last_corrected_dt > __CORRECT_INTERVAL:
            __correct_offset()
            last_corrected_dt = datetime.now()
        sleep(__SLEEP_CORRECT_THREAD)


def start_ntp_thread():
    global _controlling_thread
    if _controlling_thread is not None:
        return
    _controlling_thread = threading.Thread(target=__correct_offset_runnable, daemon=True)
    _controlling_thread.start()


def time_now() -> np.datetime64:
    return np.datetime64(int((time.time() + get_offset()) * 1_000_000_000), "ns")


def get_offset():
    global _offset
    if _offset is None:
        __correct_offset()
        if _offset is None:  # if something really went wrong
            logger.warning("Unable to get ntp offset value. Very unexpected!")
            _offset = 0.0
    return _offset
