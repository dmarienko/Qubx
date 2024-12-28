from pathlib import Path

import pytest

from qubx.utils.runner.configs import load_strategy_config_from_yaml

CONFIGS_DIR = Path(__file__).parent / "configs"


def test_strategy_config_parsing():
    # test basic config
    basic_yaml = CONFIGS_DIR / "basic.yaml"
    config = load_strategy_config_from_yaml(basic_yaml)
    assert config.strategy == "sty.models.portfolio.pigone.TestPig1"
    assert config.parameters["top_capitalization_percentile"] == 2
    assert set(config.exchanges.keys()) == {"BINANCE.UM", "KRAKEN.F"}
    assert config.aux is not None
    assert config.aux.reader == "mqdb::nebula"

    # test config without aux reader (ok)
    no_aux_yaml = CONFIGS_DIR / "no_aux.yaml"
    config = load_strategy_config_from_yaml(no_aux_yaml)
    assert config.aux is None

    # test config without exchanges (throw exception)
    no_exchanges_yaml = CONFIGS_DIR / "no_exchanges.yaml"
    with pytest.raises(ValueError):
        load_strategy_config_from_yaml(no_exchanges_yaml)
