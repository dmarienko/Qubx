from pathlib import Path

import yaml
from pydantic import BaseModel, Field


class ExchangeConfig(BaseModel):
    connector: str
    universe: list[str]


class AuxConfig(BaseModel):
    reader: str
    args: dict = Field(default_factory=dict)


class LoggingConfig(BaseModel):
    logger: str
    position_interval: str
    portfolio_interval: str
    heartbeat_interval: str = "1m"


class StrategyConfig(BaseModel):
    strategy: str
    parameters: dict = Field(default_factory=dict)
    exchanges: dict[str, ExchangeConfig]
    logging: LoggingConfig
    aux: AuxConfig | None = None


def load_strategy_config_from_yaml(path: Path | str, key: str | None = None) -> StrategyConfig:
    """
    Loads a strategy configuration from a YAML file.

    Args:
        path (str | Path): The path to the YAML file.
        key (str | None): The key to extract from the YAML file.

    Returns:
        StrategyConfig: The parsed configuration.
    """
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"File {path} not found.")
    with path.open("r") as f:
        config_dict = yaml.safe_load(f)
        if key:
            config_dict = config_dict[key]
        return StrategyConfig(**config_dict)


class StrategySimulationConfig(BaseModel):
    strategy: str | list[str]
    parameters: dict = Field(default_factory=dict)
    data: dict = Field(default_factory=dict)
    simulation: dict = Field(default_factory=dict)
    description: str | list[str] | None = None
    variate: dict = Field(default_factory=dict)


def load_simulation_config_from_yaml(path: Path | str) -> StrategySimulationConfig:
    with open(path, "r") as f:
        cfg = yaml.safe_load(f)
    return StrategySimulationConfig(**cfg)
