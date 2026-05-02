import pytest
from intelligen import constants

def test_constants_values():
    assert pytest.approx(constants.g, abs=1e-3) == 9.807
    assert constants.speed_of_light == 299_792_458
    assert constants.G == 6.67430e-11
    assert constants.kilo == 1e3
    assert constants.milli == 1e-3
