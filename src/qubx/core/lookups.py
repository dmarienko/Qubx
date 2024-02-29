import glob, re
import json, os, dataclasses
from datetime import datetime
from typing import Dict, List, Optional

from qubx.core.basics import Instrument, FuturesInfo
from qubx.utils.marketdata.binance import get_binance_symbol_info_for_type
from qubx import logger
from qubx.utils.misc import makedirs, get_local_qubx_folder


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
    lookup: Dict[str, Instrument]
    path: str

    def __init__(self, path: str=makedirs(get_local_qubx_folder(), 'instruments')) -> None:
        self.path = path
        if not self.load():
            self.refresh()
        self.load()

    def load(self) -> bool:
        self.lookup = {}
        data_exists = False
        for fs in glob.glob(self.path + '/*.json'):
            try:
                with open(fs, 'r') as f:
                    instrs = json.load(f, cls=_InstrumentDecoder)
                    for i in instrs:  
                        self.lookup[f"{i.exchange}:{i.symbol}"] = i
                    data_exists = True
            except Exception as ex:
                logger.warning(ex)

        return data_exists

    def find(self, exchange: str, base: str, quote: str) -> Optional[Instrument]:
        for i in self.lookup.values():
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
        for k, v in self.lookup.items():
            if re.match(c, k):
                res.append(v)
        return res
    
    def refresh(self):
        for mn in dir(self):
            if mn.startswith('_update_'):
                getattr(self, mn)(self.path)

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
  