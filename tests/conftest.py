import os
import pytest
import dotenv


def pytest_addoption(parser):
    add_env_option(parser)


def add_env_option(parser):
    parser.addoption(
        "--env",
        default=".env.integration",
        help="Path to the environment file to use for integration tests",
    )


@pytest.fixture
def exchange_credentials(request):
    EXCHANGE_MAPPINGS = {"BINANCE_SPOT": "BINANCE", "BINANCE_FUTURES": "BINANCE.UM"}
    env_path = request.config.getoption("--env")
    options = dotenv.dotenv_values(env_path)
    api_keys = {k: v for k, v in options.items() if k.endswith("_API_KEY")}
    api_secrets = {k: v for k, v in options.items() if k.endswith("_SECRET")}
    exchange_credentials = {}
    for key, api_key in api_keys.items():
        exchange = key[: -len("_API_KEY")]
        exchange_credentials[EXCHANGE_MAPPINGS.get(exchange, exchange)] = {
            "apiKey": api_key,
            "secret": api_secrets[f"{exchange}_SECRET"],
        }
    return exchange_credentials
