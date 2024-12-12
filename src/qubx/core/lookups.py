import configparser
import dataclasses
import glob
import json
import os
import re
from datetime import datetime
from typing import Dict, List, Optional

import pandas as pd
import stackprinter

from qubx import logger
from qubx.core.basics import ZERO_COSTS, AssetType, Instrument, MarketType, TransactionCostsCalculator
from qubx.utils.marketdata.dukas import SAMPLE_INSTRUMENTS
from qubx.utils.misc import get_local_qubx_folder, makedirs

_DEF_INSTRUMENTS_FOLDER = "instruments"
_DEF_FEES_FOLDER = "fees"


class _InstrumentEncoder(json.JSONEncoder):
    def default(self, obj):
        if dataclasses.is_dataclass(obj):
            return {k: v for k, v in dataclasses.asdict(obj).items() if not k.startswith("_")}
        if isinstance(obj, (datetime)):
            return obj.isoformat()
        return super().default(obj)


class _InstrumentDecoder(json.JSONDecoder):
    def decode(self, json_string):
        obj = super(_InstrumentDecoder, self).decode(json_string)
        if isinstance(obj, dict):
            # Convert delivery_date and onboard_date strings to datetime
            delivery_date = obj.get("delivery_date")
            onboard_date = obj.get("onboard_date")
            if delivery_date:
                obj["delivery_date"] = datetime.strptime(delivery_date.split(".")[0], "%Y-%m-%dT%H:%M:%S")
            if onboard_date:
                obj["onboard_date"] = datetime.strptime(onboard_date.split(".")[0], "%Y-%m-%dT%H:%M:%S")
            return Instrument(
                symbol=obj["symbol"],
                asset_type=AssetType[obj["asset_type"]],
                market_type=MarketType[obj["market_type"]],
                exchange=obj["exchange"],
                base=obj["base"],
                quote=obj["quote"],
                settle=obj["settle"],
                exchange_symbol=obj.get("exchange_symbol", obj["symbol"]),
                tick_size=float(obj["tick_size"]),
                lot_size=float(obj["lot_size"]),
                min_size=float(obj["min_size"]),
                min_notional=float(obj.get("min_notional", 0.0)),
                initial_margin=float(obj.get("initial_margin", 0.0)),
                maint_margin=float(obj.get("maint_margin", 0.0)),
                liquidation_fee=float(obj.get("liquidation_fee", 0.0)),
                contract_size=float(obj.get("contract_size", 1.0)),
                onboard_date=obj.get("onboard_date"),
                delivery_date=obj.get("delivery_date"),
            )
        elif isinstance(obj, list):
            return [self.decode(json.dumps(item)) for item in obj]
        return obj


class InstrumentsLookup:
    _lookup: Dict[str, Instrument]
    _path: str

    def __init__(self, path: str = makedirs(get_local_qubx_folder(), _DEF_INSTRUMENTS_FOLDER)) -> None:
        self._path = path
        if not self.load():
            self.refresh()
        self.load()

    def load(self) -> bool:
        self._lookup = {}
        data_exists = False
        for fs in glob.glob(self._path + "/*.json"):
            try:
                with open(fs, "r") as f:
                    instrs: list[Instrument] = json.load(f, cls=_InstrumentDecoder)
                    for i in instrs:
                        self._lookup[f"{i.exchange}:{i.symbol}"] = i
                    data_exists = True
            except Exception as ex:
                stackprinter.show_current_exception()
                logger.warning(ex)

        return data_exists

    def find(self, exchange: str, base: str, quote: str, settle: Optional[str] = None) -> Optional[Instrument]:
        for i in self._lookup.values():
            if i.exchange == exchange and (
                (i.base == base and i.quote == quote) or (i.base == quote and i.quote == base)
            ):
                if settle is not None and i.settle is not None:
                    if i.settle == settle:
                        return i
                else:
                    return i
        return None

    def find_symbol(self, exchange: str, symbol: str) -> Optional[Instrument]:
        for i in self._lookup.values():
            if (i.exchange == exchange) and (i.symbol == symbol):
                return i
        return None

    def find_instruments(self, exchange: str, quote: str | None = None) -> list[Instrument]:
        return [i for i in self._lookup.values() if i.exchange == exchange and (quote is None or i.quote == quote)]

    def _save_to_json(self, path, instruments: List[Instrument]):
        with open(path, "w") as f:
            json.dump(instruments, f, cls=_InstrumentEncoder, indent=4)
        logger.info(f"Saved {len(instruments)} to {path}")

    def find_aux_instrument_for(self, instrument: Instrument, base_currency: str) -> Instrument | None:
        """
        Tries to find aux instrument (for conversions to funded currency)
        for example:
            ETHBTC -> BTCUSDT for base_currency USDT
            EURGBP -> GBPUSD for base_currency USD
            ...
        """
        base_currency = base_currency.upper()
        if instrument.quote != base_currency:
            return self.find(instrument.exchange, instrument.quote, base_currency)
        return None

    def __getitem__(self, spath: str) -> List[Instrument]:
        res = []
        c = re.compile(spath)
        for k, v in self._lookup.items():
            if re.match(c, k):
                res.append(v)
        return res

    def refresh(self):
        for mn in dir(self):
            if mn.startswith("_update_"):
                getattr(self, mn)(self._path)

    def _ccxt_update(
        self,
        path: str,
        file_name: str,
        exchange_to_ccxt_name: dict[str, str],
        keep_types: list[MarketType] | None = None,
    ):
        import ccxt as cx

        from qubx.utils.marketdata.ccxt import ccxt_symbol_to_instrument

        instruments = []
        for exch, ccxt_name in exchange_to_ccxt_name.items():
            exch = exch.upper()
            ccxt_name = ccxt_name.lower()
            ex: cx.Exchange = getattr(cx, ccxt_name)()
            mkts = ex.load_markets()
            for v in mkts.values():
                if v["index"]:
                    continue
                instr = ccxt_symbol_to_instrument(exch, v)
                if not keep_types or instr.market_type in keep_types:
                    instruments.append(instr)

        # - drop to file
        self._save_to_json(os.path.join(path, f"{file_name}.json"), instruments)

    def _update_kraken(self, path: str):
        self._ccxt_update(path, "kraken.f", {"kraken.f": "krakenfutures"})
        self._ccxt_update(path, "kraken", {"kraken": "kraken"})

    def _update_binance(self, path: str):
        self._ccxt_update(path, "binance", {"binance": "binance"}, keep_types=[MarketType.SPOT, MarketType.MARGIN])
        self._ccxt_update(path, "binance.um", {"binance.um": "binanceusdm"})
        self._ccxt_update(path, "binance.cm", {"binance.cm": "binancecoinm"})

    def _update_bitfinex(self, path: str):
        self._ccxt_update(path, "bitfinex.f", {"bitfinex.f": "bitfinex"}, keep_types=[MarketType.SWAP])

    def _update_dukas(self, path: str):
        self._save_to_json(os.path.join(path, "dukas.json"), SAMPLE_INSTRUMENTS)


# - TODO: need to find better way to extract actual data !!
_DEFAULT_FEES = """
[binance]
# SPOT (maker, taker)
vip0_usdt = 0.1000,0.1000
vip1_usdt = 0.0900,0.1000
vip2_usdt = 0.0800,0.1000
vip3_usdt = 0.0420,0.0600
vip4_usdt = 0.0420,0.0540
vip5_usdt = 0.0360,0.0480
vip6_usdt = 0.0300,0.0420
vip7_usdt = 0.0240,0.0360
vip8_usdt = 0.0180,0.0300
vip9_usdt = 0.0120,0.0240

# SPOT (maker, taker)
vip0_bnb = 0.0750,0.0750
vip1_bnb = 0.0675,0.0750
vip2_bnb = 0.0600,0.0750
vip3_bnb = 0.0315,0.0450
vip4_bnb = 0.0315,0.0405
vip5_bnb = 0.0270,0.0360
vip6_bnb = 0.0225,0.0315
vip7_bnb = 0.0180,0.0270
vip8_bnb = 0.0135,0.0225
vip9_bnb = 0.0090,0.0180

# UM futures (maker, taker)
[binance.um]
vip0_usdt = 0.0200,0.0500
vip1_usdt = 0.0160,0.0400
vip2_usdt = 0.0140,0.0350
vip3_usdt = 0.0120,0.0320
vip4_usdt = 0.0100,0.0300
vip5_usdt = 0.0080,0.0270
vip6_usdt = 0.0060,0.0250
vip7_usdt = 0.0040,0.0220
vip8_usdt = 0.0020,0.0200
vip9_usdt = 0.0000,0.0170

# CM futures (maker, taker)
[binance.cm]
vip0 = 0.0200,0.0500
vip1 = 0.0160,0.0400
vip2 = 0.0140,0.0350
vip3 = 0.0120,0.0320
vip4 = 0.0100,0.0300
vip5 = 0.0080,0.0270
vip6 = 0.0060,0.0250
vip7 = 0.0040,0.0220
vip8 = 0.0020,0.0200
vip9 = 0.0000,0.0170

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

[kraken]
K0=0.25,0.40
K10=0.20,0.35
K50=0.14,0.24
K100=0.12,0.22
K250=0.10,0.20
K500=0.08,0.18
M1=0.06,0.16
M2.5=0.04,0.14
M5=0.02,0.12
M10=0.0,0.10

[kraken.f]
K0=0.0200,0.0500
K100=0.0150,0.0400
M1=0.0125,0.0300
M5=0.0100,0.0250
M10=0.0075,0.0200
M20=0.0050,0.0150
M50=0.0025,0.0125
M100=0.0000,0.0100
"""


class FeesLookup:
    """
    Fees lookup
    """

    _lookup: Dict[str, TransactionCostsCalculator]
    _path: str

    def __init__(self, path: str = makedirs(get_local_qubx_folder(), _DEF_FEES_FOLDER)) -> None:
        self._path = path
        if not self.load():
            self.refresh()
        self.load()

    def load(self) -> bool:
        self._lookup = {}
        data_exists = False
        parser = configparser.ConfigParser()
        # - load all avaliable configs
        for fs in glob.glob(self._path + "/*.ini"):
            parser.read(fs)
            data_exists = True

        for exch in parser.sections():
            for spec, info in parser[exch].items():
                try:
                    maker, taker = info.split(",")
                    self._lookup[f"{exch}_{spec}"] = (float(maker), float(taker))
                except:
                    logger.warning(f'Wrong spec format for {exch}: "{info}". Should be spec=maker,taker')

        return data_exists

    def __getitem__(self, spath: str) -> List[Instrument]:
        res = []
        c = re.compile(spath)
        for k, v in self._lookup.items():
            if re.match(c, k):
                res.append((k, v))
        return res

    def refresh(self):
        with open(os.path.join(self._path, "default.ini"), "w") as f:
            f.write(_DEFAULT_FEES)

    def find(self, exchange: str, spec: str | None) -> Optional[TransactionCostsCalculator]:
        if spec is None:
            return ZERO_COSTS
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

    def find_fees(self, exchange: str, spec: str | None) -> Optional[TransactionCostsCalculator]:
        return self.fees.find(exchange, spec)

    def find_aux_instrument_for(self, instrument: Instrument, base_currency: str) -> Optional[Instrument]:
        return self.instruments.find_aux_instrument_for(instrument, base_currency)

    def find_instrument(self, exchange: str, base: str, quote: str) -> Optional[Instrument]:
        return self.instruments.find(exchange, base, quote)

    def find_instruments(self, exchange: str, quote: str | None = None) -> list[Instrument]:
        return self.instruments.find_instruments(exchange, quote)

    def find_symbol(self, exchange: str, symbol: str) -> Optional[Instrument]:
        return self.instruments.find_symbol(exchange, symbol)
