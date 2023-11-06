# -*- coding: utf-8 -*-
"""
Created on Fri Dec  6 14:04:25 2019

@author: xv18766
"""

import numpy as np

epsilon_0 = 8.854E-12 # electric permittivity of free space
chi = 9 # electric susceptibility
N = 2E26 # density of oscillators
e = 1.6022E-19 # electron charge
m_0 = 9.109E-31 # electron mass
omega_0 = 6E14 # oscillator resonance
gamma = 5E13 # damping rate
c = 2.997930E17 # speed of light (nm/s)

# constants for Lorentzian approximation
def epsilon_st():
    return 1 + chi + ((N*e**2)/(epsilon_0*m_0))/omega_0**2

def epsilon_inf():
    return 1 + chi

# relative dielectric constant
def epsilon_1(omega, approx=False): # real part
    if not approx:
        return 1 + chi + ((N*e**2)/(epsilon_0*m_0))*(omega_0**2 - omega**2)/(
                        (omega_0**2 - omega**2)**2 + (gamma*omega)**2)
    else:
        domega = omega - omega_0
        return epsilon_inf - (
            (epsilon_st() - epsilon_inf())
                *(2*omega_0*domega)/(4*domega**2 + gamma**2))

def epsilon_2(omega, approx=False): # imaginary part
    if not approx:
        return ((N*e**2)/(epsilon_0*m_0))*(gamma*omega/
                 ((omega_0**2 - omega**2)**2 + (gamma*omega)**2))
    else:
        domega = omega - omega_0
        return (epsilon_st() - epsilon_inf())*gamma*omega_0/(
                4*domega**2 + gamma**2)

# refractive index
def n(omega, epsilon1=None, epsilon2=None, approx=False): # real part
    if not epsilon1:
        epsilon1 = epsilon_1(omega, approx=approx)
    if not epsilon2:
        epsilon2 = epsilon_2(omega, approx=approx)
    return 2**(-1/2)*np.sqrt(epsilon1 + np.sqrt(epsilon1**2 + epsilon2**2))

def k(omega, epsilon1=None, epsilon2=None, approx=False): # imaginary part
    if not epsilon1:
        epsilon1 = epsilon_1(omega, approx=approx)
    if not epsilon2:
        epsilon2 = epsilon_2(omega, approx=approx)
    return 2**(-1/2)*np.sqrt(-epsilon1 + np.sqrt(epsilon1**2 + epsilon2**2))
