import glob, re
import json, os, dataclasses
from datetime import datetime
from typing import Dict, List, Optional
import configparser
import stackprinter

from qubx.core.basics import Instrument, FuturesInfo, TransactionCostsCalculator, ZERO_COSTS
from qubx.utils.marketdata.binance import get_binance_symbol_info_for_type
from qubx import logger
from qubx.utils.misc import makedirs, get_local_qubx_folder

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
    def _preprocess(d, ks):
        fi = d.get(ks)
        _preproc = lambda x: x[: x.find(".")] if x.endswith("Z") else x
        if fi:
            fi["delivery_date"] = datetime.strptime(
                _preproc(fi.get("delivery_date", "5000-01-01T00:00:00")), "%Y-%m-%dT%H:%M:%S"
            )
            fi["onboard_date"] = datetime.strptime(
                _preproc(fi.get("onboard_date", "1970-01-01T00:00:00")), "%Y-%m-%dT%H:%M:%S"
            )
        return d | {ks: FuturesInfo(**fi) if fi else None}

    def decode(self, json_string):
        obj = super(_InstrumentDecoder, self).decode(json_string)
        if isinstance(obj, dict):
            return Instrument(**_InstrumentDecoder._preprocess(obj, "futures_info"))
        elif isinstance(obj, list):
            return [Instrument(**_InstrumentDecoder._preprocess(o, "futures_info")) for o in obj]
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
                    instrs = json.load(f, cls=_InstrumentDecoder)
                    for i in instrs:
                        self._lookup[f"{i.exchange}:{i.symbol}"] = i
                    data_exists = True
            except Exception as ex:
                stackprinter.show_current_exception()
                logger.warning(ex)

        return data_exists

    def find(self, exchange: str, base: str, quote: str, margin: Optional[str] = None) -> Optional[Instrument]:
        for i in self._lookup.values():
            if i.exchange == exchange and (
                (i.base == base and i.quote == quote) or (i.base == quote and i.quote == base)
            ):
                if margin is not None and i.margin_symbol is not None:
                    if i.margin_symbol == margin:
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
            json.dump(instruments, f, cls=_InstrumentEncoder)
        logger.info(f"Saved {len(instruments)} to {path}")

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
            if mn.startswith("_update_"):
                getattr(self, mn)(self._path)

    def _update_kraken(self, path: str):
        import ccxt as cx

        kf = cx.krakenfutures()
        ks = cx.kraken()
        f_mkts = kf.load_markets()
        s_mkts = ks.load_markets()

        # - process futures
        f_instruments = []
        for _, v in f_mkts.items():
            info = v["info"]
            # - we skip index as it's not traded
            if v["index"]:
                continue
            maint_margin = 0
            required_margin = 0
            if "marginLevels" in info:
                margins = info["marginLevels"][0]
                maint_margin = float(margins["maintenanceMargin"])
                required_margin = float(margins["initialMargin"])
            f_instruments.append(
                Instrument(
                    v["symbol"],
                    v["type"],
                    "KRAKEN.F",
                    v["base"],
                    v["quote"],
                    v["settle"],
                    min_tick=float(info["tickSize"]),
                    min_size_step=float(v["precision"]["price"]),
                    min_size=v["precision"]["amount"],
                    futures_info=FuturesInfo(
                        contract_type=info["type"],
                        contract_size=float(info["contractSize"]),
                        onboard_date=info["openingDate"],
                        delivery_date=v["expiryDatetime"] if "expiryDatetime" in info else "2100-01-01T00:00:00",
                        maint_margin=maint_margin,
                        required_margin=required_margin,
                    ),
                )
            )
        # - drop to file
        self._save_to_json(os.path.join(path, "kraken.f.json"), f_instruments)

        # - process spots
        s_instruments = []
        for _, v in s_mkts.items():
            info = v["info"]
            s_instruments.append(
                Instrument(
                    v["symbol"],
                    v["type"],
                    "KRAKEN",
                    v["base"],
                    v["quote"],
                    v["settle"],
                    min_tick=float(info["tick_size"]),
                    min_size_step=float(v["precision"]["price"]),
                    min_size=float(v["precision"]["amount"]),
                )
            )
        # - drop to file
        self._save_to_json(os.path.join(path, "kraken.json"), s_instruments)

    def _update_dukas(self, path: str):
        instruments = [
            Instrument("EURUSD", "FX", "DUKAS", "EUR", "USD", "USD", 0.00001, 1, 1000),
            Instrument("GBPUSD", "FX", "DUKAS", "GBP", "USD", "USD", 0.00001, 1, 1000),
            Instrument("USDJPY", "FX", "DUKAS", "USD", "JPY", "USD", 0.001, 1, 1000),
            Instrument("USDCAD", "FX", "DUKAS", "USD", "CAD", "USD", 0.00001, 1, 1000),
            Instrument("AUDUSD", "FX", "DUKAS", "AUD", "USD", "USD", 0.00001, 1, 1000),
            Instrument("USDPLN", "FX", "DUKAS", "USD", "PLN", "USD", 0.00001, 1, 1000),
            Instrument("EURGBP", "FX", "DUKAS", "EUR", "GBP", "USD", 0.00001, 1, 1000),
            # TODO: addd all or find how to get it from site
        ]
        self._save_to_json(os.path.join(path, "dukas.json"), instruments)

    def _update_binance(self, path: str):
        infos = get_binance_symbol_info_for_type(["UM", "CM", "SPOT"])
        for exchange, info in infos.items():
            instruments = []
            for s in info["symbols"]:
                tick_size, size_step = None, None
                for i in s["filters"]:
                    if i["filterType"] == "PRICE_FILTER":
                        tick_size = float(i["tickSize"])
                    if i["filterType"] == "LOT_SIZE":
                        size_step = float(i["stepSize"])

                fut_info = None
                if "contractType" in s:
                    fut_info = FuturesInfo(
                        s.get("contractType", "UNKNOWN"),
                        datetime.fromtimestamp(s.get("deliveryDate", 0) / 1000.0),
                        datetime.fromtimestamp(s.get("onboardDate", 0) / 1000.0),
                        float(s.get("contractSize", 1)),
                        float(s.get("maintMarginPercent", 0)),
                        float(s.get("requiredMarginPercent", 0)),
                        float(s.get("liquidationFee", 0)),
                    )

                instruments.append(
                    Instrument(
                        s["symbol"],
                        "CRYPTO",
                        exchange.upper(),
                        s["baseAsset"],
                        s["quoteAsset"],
                        s.get("marginAsset", None),
                        tick_size,
                        size_step,
                        min_size=size_step,  # TODO: not sure about minimal position for Binance
                        futures_info=fut_info,
                    )
                )
            # - store to file
            self._save_to_json(os.path.join(path, f"{exchange}.json"), instruments)


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

    def find_fees(self, exchange: str, spec: str) -> Optional[TransactionCostsCalculator]:
        return self.fees.find(exchange, spec)

    def find_aux_instrument_for(self, instrument: Instrument, base_currency: str) -> Optional[Instrument]:
        return self.instruments.find_aux_instrument_for(instrument, base_currency)

    def find_instrument(self, exchange: str, base: str, quote: str) -> Optional[Instrument]:
        return self.instruments.find(exchange, base, quote)

    def find_instruments(self, exchange: str, quote: str | None = None) -> list[Instrument]:
        return self.instruments.find_instruments(exchange, quote)

    def find_symbol(self, exchange: str, symbol: str) -> Optional[Instrument]:
        return self.instruments.find_symbol(exchange, symbol)
