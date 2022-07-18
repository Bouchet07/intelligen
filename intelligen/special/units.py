from .typings import *
# from typing import Union, List
# from numbers import Real
# Vector = List[Real]
from ..constants import *

def convert(x: Union[Real, Vector], ini_units: str, fin_units: str) -> Union[Real, Vector]:
    """
    Unit conversor
    ==============

    Parameters
    ----------
    x : float, Vector
        Value before conversion
    ini_units : str
        Units of value
    fin_units : str
        Units to convert

    Returns
    -------
    float, Vector
        Converted Value

    Units
    -----

    ---------
    - Length:

        SI
            m: -> metre

        Non-SI
            A: -> angstrom
            si: -> Scandinavian mile
            xu: -> x unit

        Imperial/US
            th: -> thou
            in: -> inch
            ft: -> foot
            yd: -> yard
            mi: -> mile
            lg: -> league
            
        Marine    
            fm: -> fathom
            nmi: -> nautical mile
        
        Surveying
            ch: -> chain
            rod: -> rod

        Astronomy
            er: -> Earth radius
            ld: -> lunar distance
            au: -> astronomical unit
            ly: -> light-year
            pc: -> parsec
            hl: -> Hubble length
        
        Physics
            re: -> electron radius
            lc: -> Compton wavelength of the electron
            rlc: -> reduced Compton wavelength of the electron
            a0: -> Bohr radius of the hydrogen atom (Atomic unit of length)
            1/r: -> reduced wavelength of hydrogen radiation
            lp: -> Planck length
            ls: -> Stoney unit of length
            lqcd: -> Quantum chromodynamics (QCD) unit of length
            eV-1: -> Natural units based on the electronvolt
        
        for more [information](https://en.wikipedia.org/wiki/Unit_of_length)

    -------    
    - Mass:

        SI
            kg: -> kilogram
        
        Non-SI
            t: -> tonne
            da: -> dalton
        
        Imperial/US
            sl: -> slug
            lb: -> pound
            
        Physics    
            pm: -> Planck mass
            sm: -> solar mass

        for more [information](https://en.wikipedia.org/wiki/Mass#Units_of_mass)
        
    -------------
    - Temperature

        SI
            K: -> Kelvin

        Non-SI
            C: -> Celsius
            F: -> Fahrenheit
            R: -> Rankine
            De: -> Delisle
            N: -> Newton
            Re: -> Reaumur
            Ro: -> Romer

        for more [information](https://en.wikipedia.org/wiki/Conversion_of_scales_of_temperature)
    
    ------
    - Time

        SI
            s: -> second
        
        Non-SI
            min: -> minute
            h: -> hour
            day: -> day
            wk: -> week
            mth: -> month (30 days)
            y: -> year (365 days)
    """

    length = ['m',                                                       #SI
              'A', 'si', 'xu',                                           #Non-SI
              'th', 'in', 'ft', 'yd', 'mi', 'lg',                        #Imperial/US
              'fm', 'nmi',                                               #Marine
              'ch', 'rod',                                               #Surveying
              'er', 'ld', 'au', 'ly', 'pc', 'hl',                        #Astronomy
              're', 'lc', 'rlc', 'a0', '1/r', 'lp', 'ls', 'lqcd', 'eV-1']#Physics   
    
    mass = ['kg',       #SI
            't', 'da',  #Non-SI
            'sl', 'lb', #Imperial/US
            'pm', 'sm'] #Physics

    temperature = ['K',                                     #Si
                   'C', 'F', 'R', 'De', 'N', 'Re', 'Ro']    #Non-SI

    SI_units = [length, mass, temperature]

    for l in SI_units:
        if ini_units in l:
            if fin_units in l: break
    else: raise ValueError('Units must be in the same category')

    #------------------Length------------------#

    #SI
    if ini_units == 'm': metre = x
    #Non-SI
    elif ini_units == 'A': metre = 100*pico*x
    elif ini_units == 'si': metre = 10_000*x
    elif ini_units == 'xu': metre = 0.1*pico*x
    #Imperial/US
    elif ini_units == 'th': metre = 25.4*micro*x
    elif ini_units == 'in': metre = 25.4*milli*x
    elif ini_units == 'ft': metre = 304.8*milli*x
    elif ini_units == 'yd': metre = 914.4*milli*x
    elif ini_units == 'mi': metre = 1609.344*x
    elif ini_units == 'lg': metre = 4800*x
    #Marine
    elif ini_units == 'fm': metre = 1.8288*x
    elif ini_units == 'nmi': metre = 1852*x
    #Surveying
    elif ini_units == 'ch': metre = 20.1168*x
    elif ini_units == 'rod': metre = 5.0292*x
    #Astronomy
    elif ini_units == 'er': metre = 6371*kilo*x
    elif ini_units == 'ld': metre = 384_402*kilo*x
    elif ini_units == 'au': metre = 149_597_870_700*x
    elif ini_units == 'ly': metre = 9_460_730_472_580.8*kilo*x
    elif ini_units == 'pc': metre = 30_856_775_814_671.9*kilo*x
    elif ini_units == 'hl': metre = 4.55*giga*(30_856_775_814_671.9*kilo)*x
    #Physics
    elif ini_units == 're': metre = 2.817_940_285e-15*x
    elif ini_units == 'lc': metre = 2.426_310_215e-12*x
    elif ini_units == 'rlc': metre = 3.861_592_6764e-13*x
    elif ini_units == 'a0': metre = 5.291_772_083e-11*x
    elif ini_units == '1/R': metre = 9.112_670_505_509e-8*x
    elif ini_units == 'lp': metre = 1.616_255e-35*x
    elif ini_units == 'ls': metre = 1.381e-35*x
    elif ini_units == 'lqcd': metre = 2.103e-16*x
    elif ini_units == 'eV-1': metre = 1.97e-7*x


    #SI
    if fin_units == 'm': return metre
    #Non-SI
    elif fin_units == 'A': return metre/(100*pico)
    elif fin_units == 'si': return metre/(10_000)
    elif fin_units == 'xu': return metre/(0.1*pico)
    #Imperial/US
    elif fin_units == 'th': return metre/(25.4*micro)
    elif fin_units == 'in': return metre/(25.4*milli)
    elif fin_units == 'ft': return metre/(304.8*milli)
    elif fin_units == 'yd': return metre/(914.4*milli)
    elif fin_units == 'mi': return metre/(1609.344)
    elif fin_units == 'lg': return metre/(4800)
    #Marine
    elif fin_units == 'fm': return metre/(1.8288)
    elif fin_units == 'nmi': return metre/(1852)
    #Surveying
    elif fin_units == 'ch': return metre/(20.1168)
    elif fin_units == 'rod': return metre/(5.0292)
    #Astronomy
    elif fin_units == 'er': return metre/(6371*kilo)
    elif fin_units == 'ld': return metre/(384_402*kilo)
    elif fin_units == 'au': return metre/(149_597_870_700)
    elif fin_units == 'ly': return metre/(9_460_730_472_580.8*kilo)
    elif fin_units == 'pc': return metre/(30_856_775_814_671.9*kilo)
    elif fin_units == 'hl': return metre/(4.55*giga*(30_856_775_814_671.9*kilo))
    #Physics
    elif fin_units == 're': return metre/2.817_940_285e-15
    elif fin_units == 'lc': return metre/2.426_310_215e-12
    elif fin_units == 'rlc': return metre/3.861_592_6764e-13
    elif fin_units == 'a0': return metre/5.291_772_083e-11
    elif fin_units == '1/R': return metre/9.112_670_505_509e-8
    elif fin_units == 'lp': return metre/1.616_199e-35
    elif fin_units == 'ls': return metre/1.381e-35
    elif fin_units == 'lqcd': return metre/2.103e-16
    elif fin_units == 'eV-1': return metre/1.97e-7

    #------------------Mass------------------#

    #SI
    if ini_units == 'kg': kilograms = x
    #Non-SI
    elif ini_units == 't': kilograms = 1000*x
    elif ini_units == 'da': kilograms = 1.66e-27*x
    #Imperial/US
    elif ini_units == 'sl': kilograms = 14.593_90*x
    elif ini_units == 'lb': kilograms = 0.453_592_37*x
    #Physics
    elif ini_units == 'pm': kilograms = 2.176_434e-8*x
    elif ini_units == 'sm': kilograms = 1.988_47e30*x


    #SI
    if fin_units == 'kg': return kilograms
    #Non-SI
    elif fin_units == 't': return kilograms/1000
    elif fin_units == 'da': return kilograms/1.66e-27
    #Imperial/US
    elif fin_units == 'sl': return kilograms/14.593_90
    elif fin_units == 'lb': return kilograms/0.453_592_37
    #Physics
    elif fin_units == 'pm': return kilograms/2.176_434e-8
    elif fin_units == 'sm': return kilograms/1.988_47e30

    #------------------Temperature------------------#

    #SI
    if ini_units == 'K': Kelvin = x
    #Non-SI
    elif ini_units == 'C': Kelvin = x + 273.15
    elif ini_units == 'F': Kelvin = 5/9*(x + 459.67)
    elif ini_units == 'R': Kelvin = 5/9*x
    elif ini_units == 'De': Kelvin = 2/3*x - 373.15
    elif ini_units == 'N': Kelvin = 100/33*x + 273.15
    elif ini_units == 'Re': Kelvin = 5/4*x + 273.15
    elif ini_units == 'Ro': Kelvin = 40/21*(x - 7.5) + 273.15


    #SI
    if fin_units == 'K': return Kelvin
    #Non-SI
    elif fin_units == 'C': return Kelvin - 273.15
    elif fin_units == 'F': return 9/5*Kelvin - 459.67
    elif fin_units == 'R': return 9/5*Kelvin 
    elif fin_units == 'De': return 3/2*(373.15 - Kelvin)
    elif fin_units == 'N': return 33/100*(Kelvin - 273.15)
    elif fin_units == 'Re': return 4/5*(Kelvin - 273.15)
    elif fin_units == 'Ro': return 21/40*(Kelvin - 273.15) + 7.5

