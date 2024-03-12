import glob, re
import json, os, dataclasses
from datetime import datetime
from typing import Dict, List, Optional
import configparser

from qubx.core.basics import Instrument, FuturesInfo, TransactionCostsCalculator
from qubx.utils.marketdata.binance import get_binance_symbol_info_for_type
from qubx import logger
from qubx.utils.misc import makedirs, get_local_qubx_folder

_DEF_INSTRUMENTS_FOLDER = "instruments"
_DEF_FEES_FOLDER = "fees"


class _InstrumentEncoder(json.JSONEncoder):
    def default(self, obj):
        if dataclasses.is_dataclass(obj):
            return {k:v for k,v in dataclasses.asdict(obj).items() if not k.startswith('_')}
        if isinstance(obj, (datetime)):
            return obj.isoformat()
        return super().default(obj)


class _InstrumentDecoder(json.JSONDecoder):
    def _preprocess(d, ks):
        fi = d.get(ks)
        if fi:
            fi['delivery_date'] = datetime.strptime(fi.get('delivery_date', '5000-01-01T00:00:00'),'%Y-%m-%dT%H:%M:%S')
            fi['onboard_date'] = datetime.strptime(fi.get('onboard_date', '1970-01-01T00:00:00'),'%Y-%m-%dT%H:%M:%S')
        return d | {ks: FuturesInfo(**fi) if fi else None}

    def decode(self, json_string):
        obj = super(_InstrumentDecoder, self).decode(json_string)
        if isinstance(obj, dict):
            return Instrument(**_InstrumentDecoder._preprocess(obj, 'futures_info'))
        elif isinstance(obj, list):
            return [Instrument(**_InstrumentDecoder._preprocess(o, 'futures_info')) for o in obj]
        return obj


class InstrumentsLookup:
    _lookup: Dict[str, Instrument]
    _path: str

    def __init__(self, path: str=makedirs(get_local_qubx_folder(), _DEF_INSTRUMENTS_FOLDER)) -> None:
        self._path = path
        if not self.load():
            self.refresh()
        self.load()

    def load(self) -> bool:
        self._lookup = {}
        data_exists = False
        for fs in glob.glob(self._path + '/*.json'):
            try:
                with open(fs, 'r') as f:
                    instrs = json.load(f, cls=_InstrumentDecoder)
                    for i in instrs:  
                        self._lookup[f"{i.exchange}:{i.symbol}"] = i
                    data_exists = True
            except Exception as ex:
                logger.warning(ex)

        return data_exists

    def find(self, exchange: str, base: str, quote: str) -> Optional[Instrument]:
        for i in self._lookup.values():
            if i.exchange == exchange and (
                (i.base == base and i.quote == quote) or (i.base == quote and i.quote == base)
            ):
                return i
        return None
    
    def find_aux_instrument_for(self, instrument: Instrument, base_currency: str) -> Optional[Instrument]:
        """
        Tries to find aux instrument (for conversions to funded currency)
        for example: 
            ETHBTC -> BTCUSDT for base_currency USDT
            EURGBP -> GBPUSD for base_currency USD
            ...
        """
        base_currency = base_currency.upper()
        if instrument.quote != base_currency and instrument._aux_instrument is None:
            return self.find(instrument.exchange, instrument.quote, base_currency)
        return instrument._aux_instrument

    def __getitem__(self, spath: str) -> List[Instrument]:
        res = []
        c = re.compile(spath)
        for k, v in self._lookup.items():
            if re.match(c, k):
                res.append(v)
        return res
    
    def refresh(self):
        for mn in dir(self):
            if mn.startswith('_update_'):
                getattr(self, mn)(self._path)

    def _update_kraken(self, path: str):
        # TODO
        pass

    def _update_dukas(self, path: str):
        instruments = [
            Instrument('EURUSD', 'FX', 'DUKAS', 'EUR', 'USD', 'USD', 0.00001, 1, 1000),
            Instrument('GBPUSD', 'FX', 'DUKAS', 'GBP', 'USD', 'USD', 0.00001, 1, 1000),
            Instrument('USDJPY', 'FX', 'DUKAS', 'USD', 'JPY', 'USD', 0.001,   1, 1000),
            Instrument('USDCAD', 'FX', 'DUKAS', 'USD', 'CAD', 'USD', 0.00001, 1, 1000),
            Instrument('AUDUSD', 'FX', 'DUKAS', 'AUD', 'USD', 'USD', 0.00001, 1, 1000),
            Instrument('USDPLN', 'FX', 'DUKAS', 'USD', 'PLN', 'USD', 0.00001, 1, 1000),
            Instrument('EURGBP', 'FX', 'DUKAS', 'EUR', 'GBP', 'USD', 0.00001, 1, 1000),
            # TODO: addd all or find how to get it from site
        ]
        logger.info(f'Updates {len(instruments)} for DUKASCOPY')
        with open(os.path.join(path, f'dukas.json'), 'w') as f:
            json.dump(instruments, f, cls=_InstrumentEncoder)

    def _update_binance(self, path: str):
        infos = get_binance_symbol_info_for_type(['UM', 'CM', 'SPOT'])
        for exchange, info in infos.items():
            instruments = []
            for s in info['symbols']:
                tick_size, size_step = None, None 
                for i in s['filters']:
                    if i['filterType'] == 'PRICE_FILTER':
                        tick_size = float(i['tickSize'])
                    if i['filterType'] == 'LOT_SIZE':
                        size_step = float(i['stepSize'])

                fut_info = None
                if 'contractType' in s:
                    fut_info = FuturesInfo( 
                        s.get('contractType', 'UNKNOWN'),
                        datetime.fromtimestamp(s.get('deliveryDate', 0)/1000.0),
                        datetime.fromtimestamp(s.get('onboardDate', 0)/1000.0),
                        float(s.get('contractSize', 1)), 
                        float(s.get('maintMarginPercent', 0)), 
                        float(s.get('requiredMarginPercent', 0)), 
                        float(s.get('liquidationFee', 0)),
                    )

                instruments.append(
                    Instrument(
                        s['symbol'], 'CRYPTO', exchange.upper(), s['baseAsset'], s['quoteAsset'], s.get('marginAsset', None), 
                        tick_size, size_step, 
                        min_size=size_step,    # TODO: not sure about minimal position for Binance
                        futures_info=fut_info
                ))

            logger.info(f'Loaded {len(instruments)} for {exchange}')

            with open(os.path.join(path, f'{exchange}.json'), 'w') as f:
                json.dump(instruments, f, cls=_InstrumentEncoder)
  

# - TODO: need to find better way to extract actual data !!
_DEFAULT_FEES = """
[binance]
# SPOT (maker, taker)
spot_vip0_usdt = 0.1000,0.1000
spot_vip1_usdt = 0.0900,0.1000
spot_vip2_usdt = 0.0800,0.1000
spot_vip3_usdt = 0.0420,0.0600
spot_vip4_usdt = 0.0420,0.0540
spot_vip5_usdt = 0.0360,0.0480
spot_vip6_usdt = 0.0300,0.0420
spot_vip7_usdt = 0.0240,0.0360
spot_vip8_usdt = 0.0180,0.0300
spot_vip9_usdt = 0.0120,0.0240

# UM futures (maker, taker)
um_vip0_usdt = 0.0200,0.0500
um_vip1_usdt = 0.0160,0.0400
um_vip2_usdt = 0.0140,0.0350
um_vip3_usdt = 0.0120,0.0320
um_vip4_usdt = 0.0100,0.0300
um_vip5_usdt = 0.0080,0.0270
um_vip6_usdt = 0.0060,0.0250
um_vip7_usdt = 0.0040,0.0220
um_vip8_usdt = 0.0020,0.0200
um_vip9_usdt = 0.0000,0.0170

[bitmex]
tierb_xbt=0.02,0.075
tierb_usdt=-0.015,0.075
tieri_xbt=0.01,0.05
tieri_usdt=-0.015,0.05
tiert_xbt=0.0,0.04
tiert_usdt=-0.015,0.04
tierm_xbt=0.0,0.035
tierm_usdt=-0.015,0.035
tiere_xbt=0.0,0.03
tiere_usdt=-0.015,0.03
tierx_xbt=0.0,0.025
tierx_usdt=-0.015,0.025
tierd_xbt=-0.003,0.024
tierd_usdt=-0.015,0.024
tierw_xbt=-0.005,0.023
tierw_usdt=-0.015,0.023
tierk_xbt=-0.008,0.022
tierk_usdt=-0.015,0.022
tiers_xbt=-0.01,0.0175
tiers_usdt=-0.015,0.02

[dukas]
regular=0.0035,0.0035
premium=0.0017,0.0017
"""


class FeesLookup:
    """
    Fees lookup
    """
    _lookup: Dict[str, TransactionCostsCalculator]
    _path: str

    def __init__(self, path: str=makedirs(get_local_qubx_folder(), _DEF_FEES_FOLDER)) -> None:
        self._path = path
        if not self.load():
            self.refresh()
        self.load()

    def load(self) -> bool:
        self._lookup = {}
        data_exists = False
        parser = configparser.ConfigParser()
        # - load all avaliable configs
        for fs in glob.glob(self._path + '/*.ini'):
            parser.read(fs)
            data_exists = True

        for exch in parser.sections():
            for spec, info in parser[exch].items():
                try:
                    maker, taker = info.split(',')
                    self._lookup[f"{exch}_{spec}"] = (float(maker), float(taker))
                except:
                    logger.warning(f'Wrong spec format for {exch}: "{info}". Should be spec=maker,taker')

        return data_exists

    def __getitem__(self, spath: str) -> List[Instrument]:
        res = []
        c = re.compile(spath)
        for k, v in self._lookup.items():
            if re.match(c, k):
                res.append((k,v))
        return res
    
    def refresh(self):
        with open(os.path.join(self._path, 'default.ini'), 'w') as f:
            f.write(_DEFAULT_FEES)

    def find(self, exchange: str, spec: str) -> Optional[TransactionCostsCalculator]:
        key = f"{exchange}_{spec}"
        vals = self._lookup.get(key)
        return TransactionCostsCalculator(key, *self._lookup.get(key)) if vals is not None else None

    def __repr__(self) -> str:
        s = "Name:\t\t\t(maker, taker)\n"
        for k, v in self._lookup.items():
            s += f"{k.ljust(25)}: {v}\n"
        return s


@dataclasses.dataclass(frozen=True)
class GlobalLookup:
    instruments: InstrumentsLookup
    fees: FeesLookup

    def find_fees(self, exchange: str, spec: str) -> Optional[TransactionCostsCalculator]:
        return self.fees.find(exchange, spec)

    def find_aux_instrument_for(self, instrument: Instrument, base_currency: str) -> Optional[Instrument]:
        return self.instruments.find_aux_instrument_for(instrument, base_currency)

    def find_instrument(self, exchange: str, base: str, quote: str) -> Optional[Instrument]:
        return self.instruments.find(exchange, base, quote)