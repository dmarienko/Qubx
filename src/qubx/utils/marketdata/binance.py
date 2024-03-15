from os import unlink
import numpy as np
import pandas as pd
from os.path import exists, join, split, basename
from tqdm.notebook import tqdm
from typing import Any, Callable, Dict, List
from binance.client import Client, HistoricalKlinesType, BinanceAPIException
import requests

from qubx import logger
from qubx.utils.misc import makedirs, get_local_qubx_folder
from qubx.utils.pandas import generate_equal_date_ranges, srows


DEFALT_LOCAL_FILE_STORAGE = makedirs(get_local_qubx_folder(), 'data/import/binance_history/')
DEFALT_LOCAL_CSV_STORAGE = makedirs(get_local_qubx_folder(), 'data/binance/')

# _DEFAULT_MARKET_DATA_DB = 'md'
BINANCE_DATA_STORAGE = "https://s3-ap-northeast-1.amazonaws.com"
BINANCE_DATA_URL = "https://data.binance.vision/"

    
def get_binance_symbol_info_for_type(market_types: List[str]) -> Dict[str, Dict[str, Any]]:
    """
    Get list of all symbols from binance for given list of market types:
    possible types are: SPOT, FUTURES, COINSFUTURES
    
    >>> get_binance_symbol_info_for_type('FUTURES')
    
    :param market_type: SPOT, FUTURES (UM) or COINSFUTURES (CM)
    """
    client = Client()
    infos = {}
    for market_type in (market_types if not isinstance(market_types, str) else [market_types]):
        if market_type in ['FUTURES', 'UM']:
            infos['binance.um'] = client.futures_exchange_info()

        elif market_type in ['COINSFUTURES', 'CM']:
            infos['binance.cm'] = client.futures_coin_exchange_info()

        elif market_type == 'SPOT':
            infos['binance'] = client.get_exchange_info()
        else:
            raise ValueError("Only 'FUTURES | UM', 'COINSFUTURES | CM' or 'SPOT' are supported for market_type")

    return infos


def fetch_file(url, local_file_storage, chunk_size=1024*1024, progress_bar=True):
    """
    Load file from url and store it to specified storage
    """
    file = split(url)[-1]
    
    # if dest location not exists create it
    if not exists(local_file_storage):
        makedirs(local_file_storage)
        
    response = requests.get(url, stream=True)
    fpath = join(local_file_storage, file)
    with open(fpath, "wb") as handle:
        iters = response.iter_content(chunk_size=chunk_size)
        for data in tqdm(iters) if progress_bar else iters:
            handle.write(data)
    return fpath


def get_trades_files(symbol: str, instr_type: str, instr_subtype: str):
    """
    Get list of trades files for specified instrument from Binance datastorage
    """
    if instr_type.lower() == 'spot':
        instr_subtype = ''
    filter_str = join("data", instr_type.lower(), instr_subtype.lower(), "monthly", "trades", symbol.upper())
    pg = requests.get(f"{BINANCE_DATA_STORAGE}/data.binance.vision?prefix={filter_str}/")
    info = pd.read_xml(pg.text)
    return [k for k in info.Key.dropna() if k.endswith('.zip')]


def load_trades_for(symbol, instr_type='futures', instr_subtype='um', local_file_storage=DEFALT_LOCAL_FILE_STORAGE):
    """
    Load trades from Binance data storage
    >>> load_trades_for('ETHUSDT', 'futures', 'um')
    """
    local_file_storage = makedirs(local_file_storage)

    f_list = get_trades_files(symbol, instr_type, instr_subtype)
    for r_file in tqdm(f_list):
        dest_dir = join(local_file_storage, symbol.upper())
        dest_file = join(dest_dir, basename(r_file))
        if not exists(dest_file):
            fetch_file(join(BINANCE_DATA_URL, r_file), dest_dir)
        else:
            logger.info(f"{dest_file} already loaded, skipping ...")


def parse_kl_file(fpath: str) -> pd.DataFrame:
    _reader = lambda fp, hdr: pd.read_csv(fp, names = [
            'open_time', 'open', 'high', 'low', 'close', 'volume', 'close_time', 
            'quote_volume', 'count', 'taker_buy_volume', 'taker_buy_quote_volume', 'ignore'
        ], 
        usecols=[
            'open_time', 'open', 'high', 'low', 'close', 'volume', 'quote_volume', 
            'count', 'taker_buy_volume', 'taker_buy_quote_volume', 
        ], 
        index_col='open_time', header=hdr
    )
    d = _reader(fpath, None)

    # - if there is header 
    if isinstance(d.index[0], str): 
        d = d.iloc[1:, :].astype(float)

    d.index = pd.to_datetime(d.index, unit='ms').rename('timestamp')
    return d


def load_binance_kl_history(symbol: str, start: str, stop: str = 'now',
                            instr_type='futures', instr_subtype='um', 
                            timeframe='1m', temp_storage=DEFALT_LOCAL_FILE_STORAGE) -> pd.DataFrame:
    """
    Loads binance 1m KLine history from AWS storage

    Parameters
    ----------
    symbol : str
        The symbol to load data for
    start : str
        The start date in format %Y-%m-%d
    stop : str, optional
        The end date in format %Y-%m-%d, by default 'now'
    instr_type : str, optional
        The instrument type, one of 'futures', 'spot', by default 'futures'
    instr_subtype : str, optional
        The instrument subtype, one of 'um', 'cm', by default 'um'
    timeframe : str, optional
        The timeframe, one of '1m', '3m', '5m', '15m', '30m', '1h', '2h', '4h', '6h', '8h', '12h', '1d', '3d', '1w', '1M', by default '1m'
    temp_storage : str, optional
        The temporary storage location, by default 'DEFALT_LOCAL_FILE_STORAGE'

    Returns
    -------
    pd.DataFrame
        The loaded data
    """
    start = pd.Timestamp(start)
    stop = pd.Timestamp(stop)
    if instr_type.lower() == 'futures':
        subt = f"{instr_type}/{instr_subtype}"
    else:
        subt = 'spot'
    temp_storage =  join(temp_storage, subt)
 
    def _loader(start: pd.Timestamp, stop: pd.Timestamp, f_units: str):
        curr_date = pd.Timestamp('now').ceil('1d')
        data = pd.DataFrame()
        continue_from = None
        for t, _ in generate_equal_date_ranges(start, stop, 1, f_units[0].upper()):
            if f_units =='monthly':
                dt = pd.Timestamp(t)
                # stop when we got into current month
                if dt.year == curr_date.year and dt.month == curr_date.month:
                    continue_from = t
                    break
                dt = dt.strftime("%Y-%m")
            else:
                dt = pd.Timestamp(t).strftime("%Y-%m-%d")

            fname = f"{symbol.upper()}-{timeframe}-{dt}.zip"
            zf = join(temp_storage, fname)

            if not exists(zf):
                logger.info(f'[green]{(symbol)}[/green] {instr_subtype} {instr_type} loading data for [yellow]{dt}[/yellow] -> [red]{fname}[/red]')
                u = f'{BINANCE_DATA_URL}data/{subt}/{f_units}/klines/{symbol.upper()}/{timeframe}/{fname}'
                zf = fetch_file(u, temp_storage, progress_bar=False)
            else:
                logger.info(f'[green]{symbol}[/green] {instr_subtype} {instr_type} parsing data from [red]{fname}[/red]')
            if zf and exists(zf):
                try:
                    data = srows(data, parse_kl_file(zf), keep='last')
                except Exception as err:
                    logger.warning(err)
                    unlink(zf)
        return data, continue_from

    # - load by months
    data, cont_time = _loader(start, stop, 'monthly')

    # - rest data load by days
    if cont_time is not None:
        data_cont, cont_time = _loader(cont_time, stop, 'daily')
        data = srows(data, data_cont, keep='last')

    return data


def update_binance_data_storage(coins=[], quoted_in=['USDT'], market='futures', data_storage=DEFALT_LOCAL_CSV_STORAGE):
    """
    Fetch data from the Binance data storage and save it as local csv files
    TODO: csv is just temporary solution and we need to keep data in DB
    """
    info = get_binance_symbol_info_for_type(['UM', 'CM', 'SPOT'])
    if market.lower() == 'futures':
        for sy in info['binance.um']['symbols']:
            if sy['quoteAsset'] in quoted_in:
                if coins and sy['baseAsset'] not in coins:
                    continue
                symbol = sy['symbol'] 
                start = pd.Timestamp(sy['onboardDate'], unit='ms')
                data = load_binance_kl_history(symbol, start.strftime('%Y-%m-%d'), instr_type='futures', instr_subtype='um')
                # - update in mongo db
                if data is not None and not data.empty:
                    data.to_csv(join(data_storage, 'BINANCE.UM'))
                    # path = f'm1/BINANCEF:{symbol}'
                    # z_del(path, dbname=_DEFAULT_MARKET_DATA_DB)
                    # z_save(path, data, dbname=_DEFAULT_MARKET_DATA_DB)
                    del data

    if market.lower() == 'spot':
        client = Client()
        for sy in info['binance']['symbols']:
            if sy['quoteAsset'] in quoted_in:
                if coins and sy['baseAsset'] not in coins:
                    continue
                symbol = sy['symbol'] 
                # - some dirty way to get historical data start for spot
                d = client.get_historical_klines(
                    symbol, '1M', '2017-01-01', '2100-01-01', klines_type=HistoricalKlinesType.SPOT, limit=1000
                )
                start = pd.Timestamp(d[0][0], unit='ms')
                data = load_binance_kl_history(symbol, start.strftime('%Y-%m-%d'), instr_type='spot', instr_subtype='-')
                # - update in mongo db
                if data is not None and not data.empty:
                    data.to_csv(join(data_storage, 'BINANCE'))
                    # path = f'm1/BINANCE:{symbol}'
                    # z_del(path, dbname=_DEFAULT_MARKET_DATA_DB)
                    # z_save(path, data, dbname=_DEFAULT_MARKET_DATA_DB)
                    del data

# class BinanceHist:
#     START_SPOT_HIST = pd.Timestamp('2017-01-01')
#     START_FUT_HIST = pd.Timestamp('2019-09-08')
    
#     def __init__(self, api_key, secret_key):
#         self.client = Client(api_key=api_key, api_secret=secret_key)
    
#     @staticmethod
#     def from_env(path):
#         if exists(path):
#             return BinanceHist(**get_env_data_as_dict(path))
#         else:
#             raise ValueError(f"Can't find env file at {path}")
    
#     @staticmethod
#     def _t2s(time):
#         _t = pd.Timestamp(time) if not isinstance(time, pd.Timestamp) else time
#         return _t.strftime('%d %b %Y %H:%M:%S')
    
#     def load_hist_spot_data(self, symbol, timeframe, start: pd.Timestamp, stop: pd.Timestamp, step='4W', timeout_sec=2):
#         return self.load_hist_data(symbol, timeframe, start, stop, HistoricalKlinesType.SPOT, step, timeout_sec)
    
#     def load_hist_futures_data(self, symbol, timeframe, start: pd.Timestamp, stop: pd.Timestamp, step='4W', timeout_sec=2):
#         return self.load_hist_data(symbol, timeframe, start, stop, HistoricalKlinesType.FUTURES, step, timeout_sec)

#     def update_hist_data(self, symbol, stype, timeframe, exchange_id, drop_prev=False, step='4W', timeout_sec=2):
#         """
#         Update loaded data in z db:
        
#         >>> bh = BinanceHist.from_env('.env')
#         >>> bh.update_hist_data('BTCUSDT', 'FUTURES', '1Min', 'BINANCEF')
#         >>> bh.update_hist_data('BTCUSDT', 'SPOT', '1Min', 'BINANCE')
#         """
#         if isinstance(symbol, (tuple, list)):
#             for s in symbol:
#                 print(f"Working on {symbol.index(s)}th from {len(symbol)} symbols...")
#                 self.update_hist_data(s, stype, timeframe, exchange_id, drop_prev=drop_prev, step=step, timeout_sec=timeout_sec)
#             return
                
#         klt = {
#             'futures': [HistoricalKlinesType.FUTURES, BinanceHist.START_FUT_HIST],
#             'spot':    [HistoricalKlinesType.SPOT, BinanceHist.START_SPOT_HIST]
#         }.get(stype.lower()) 
        
#         if klt is None:
#             raise ValueError(f"Unknown instrument type '{stype}' !")
            
#         tD = pd.Timedelta(timeframe)
#         now = self._t2s(pd.Timestamp(datetime.datetime.now(pytz.UTC).replace(second=0)) - tD)
        
#         path = f'm1/{exchange_id}:{symbol}'
#         if drop_prev:
#             print(f' > Deleting data at {path} ...')
#             z_del(path, dbname=_DEFAULT_MARKET_DATA_DB)
        
#         mc = MongoController(dbname=_DEFAULT_MARKET_DATA_DB)
        
#         hist = z_ld(path, dbname=_DEFAULT_MARKET_DATA_DB)
#         if hist is None:
#             last_known_date = klt[1]
#         else:   
#             last_known_date = hist.index[-1]
            
#             # 're-serialize' data to be accessible through gridfs
#             if mc.get_count(path) == 0:
#                 z_save(path, hist, is_serialize=False, dbname=_DEFAULT_MARKET_DATA_DB)

#         s_time = last_known_date.round(tD)
            
#         # pull up recent data
#         def __cb(data):
#             # print(">>>>>> new data: ", len(data))
#             mc.append_data(path, data)
            
#         self.load_hist_data(symbol, timeframe, s_time, now, klt[0], step=step, timeout_sec=timeout_sec, new_data_callback=__cb)
#         mc.close()
        
#         # drop _id (after update through MongoController)
#         pdata = z_ld(path, dbname=_DEFAULT_MARKET_DATA_DB)
#         if pdata is not None:
#             pdata = pdata.drop(columns='_id')
#         z_save(path, pdata, dbname=_DEFAULT_MARKET_DATA_DB)
    
#     def load_hist_data(self, symbol, timeframe, start: pd.Timestamp, stop: pd.Timestamp, ktype, step='4W', timeout_sec=2,
#                        new_data_callback: Callable[[pd.DataFrame], None]=None):
#         start = pd.Timestamp(start) if not isinstance(start, pd.Timestamp) else start
#         stop = pd.Timestamp(stop) if not isinstance(stop, pd.Timestamp) else stop
#         df = pd.DataFrame()
        
#         _tf_str = timeframe[:2].lower()
#         tD = pd.Timedelta(timeframe)
#         now = self._t2s(stop)
#         tlr = pd.DatetimeIndex([start]).append(pd.date_range(start, now, freq=step).append(pd.DatetimeIndex([now])))
        
#         print(f' >> Loading {green(symbol)} {yellow(timeframe)} for [{red(start)}  -> {red(now)}]')
#         s = tlr[0]
#         for e in tqdm(tlr[1:]):
#             if s + tD < e:
#                 _start, _stop = self._t2s(s + tD), self._t2s(e)
#                 # print(_start, _stop)
#                 chunk = None
#                 nerr = 0
#                 while nerr < 3:
#                     try:
#                         chunk = self.client.get_historical_klines(symbol, _tf_str, _start, _stop, klines_type=ktype, limit=1000)
#                         nerr = 100
#                     except BinanceAPIException as a:
#                         print('.', end='')
#                         break
#                     except Exception as err:
#                         nerr +=1
#                         print(red(str(err)))
#                         time.sleep(10)

#                 if chunk:
#                     data = pd.DataFrame(chunk, columns = ['timestamp', 'open', 'high', 'low', 'close', 'volume', 'close_time', 'quote_av', 'trades', 'tb_base_av', 'tb_quote_av', 'ignore' ])
#                     data.index = pd.to_datetime(data['timestamp'].rename('time'), unit='ms')
#                     data = data.drop(columns=['timestamp', 'close_time', 'quote_av', 'trades', 'tb_base_av', 'tb_quote_av', 'ignore' ]).astype(float)
#                     df = df.append(data)
#                     # callback on new data
#                     if new_data_callback is not None:
#                         new_data_callback(data)
                        
#                 s = e
#                 time.sleep(timeout_sec)
#         return df
    