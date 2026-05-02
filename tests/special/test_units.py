import pytest
from intelligen.special.units import convert

def test_convert_length():
    # 1 m to mm
    #assert pytest.approx(convert(1, 'm', 'milli'), abs=1e-3) == 1000
    # Wait, 'milli' is in constants but units.py has 'm' and others.
    # Actually, looking at units.py, 'milli' isn't a length unit in the convert function, it uses 'm', 'in', 'ft', 'km' etc?
    # No, let's just do m to ft
    assert pytest.approx(convert(1, 'm', 'ft'), abs=1e-3) == 3.28084

def test_convert_mass():
    # 1 kg to lb
    assert pytest.approx(convert(1, 'kg', 'lb'), abs=1e-3) == 2.20462

def test_convert_temperature():
    # 0 C to K
    assert pytest.approx(convert(0, 'C', 'K'), abs=1e-3) == 273.15
    # 0 C to F
    assert pytest.approx(convert(0, 'C', 'F'), abs=1e-3) == 32.0
