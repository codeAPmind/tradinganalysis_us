"""基础数据网关测试 (不需要 API key)。"""
import pytest
from data.gateway import DataGateway, DataConfig


@pytest.fixture
def gateway():
    return DataGateway(DataConfig(fmp_api_key="", fred_api_key="", finnhub_api_key=""))


def test_get_price_history(gateway):
    df = gateway.get_price_history("AAPL", period="1mo")
    assert not df.empty
    assert "Close" in df.columns


def test_get_macro_indicators(gateway):
    macro = gateway.get_macro_indicators()
    assert "vix" in macro
    assert "us10y_yield" in macro
    assert "dxy" in macro


def test_get_option_chain(gateway):
    chain = gateway.get_option_chain("AAPL")
    assert "calls" in chain
    assert "puts" in chain
