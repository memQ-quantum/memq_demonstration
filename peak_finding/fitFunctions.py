##############################
# Created by Manish Singh
# 3/6/2021

# Modified by Dana Lin
# 1/11/2023
###############################
import numpy as np


def linear(x, a, b):
    return a * x + b


def quadratic(x, a, h, k):
    return a * ((x - h) ** 2) + k


def fnLorentzianSimple(x, a, gamma):
    return a * (gamma / 2) / ((gamma / 2) ** 2 + (x) ** 2)


def fnLorentzian(x, a, gamma, offset):
    return a * (gamma / 2) / ((gamma / 2) ** 2 + (x) ** 2) + offset


def fnLorentzianX0(x, a, gamma, offset, x0):
    # gamma is full width half max (FWHM)
    # x0 is peak of Lorentzian
    return a * (gamma / 2) / ((gamma / 2) ** 2 + (x - x0) ** 2) + offset


def fnLorentzianBSL(x, a, gamma, offset, x0, m, c):
    # x correction based on linear y = mx+c
    return a * gamma / 2 / ((gamma / 2) ** 2 + (m * (x - x0) + c) ** 2) + offset


def fnLorentzianBi(x, a, gamma, a2, offset):
    return (
        a * (gamma / 2 / ((gamma / 2) ** 2 + (x) ** 2))
        + a2 * (6.24 / 2 / ((6.24 / 2) ** 2 + (x) ** 2))
        + offset
    )


def fnExponential(x, a, t0, offset):
    return a * np.exp(-x / t0)


def fnBiExponential(x, a, aDash, t0, offset):
    return a * np.exp(-x / t0) + aDash * np.exp(-x / 2.3) + offset


def fnBiExponentialAmp(x, a, t0, offset):
    return a * (np.exp(-x / t0) + np.exp(-x / 2.3)) + offset


def sinusoid(x, a, w, phi, offset):
    return a * np.sin(w * x + phi) + offset


# def fnBiExponential(x, a, aDash, t0, t0Dash, offset):
#     return (a*np.exp(-x/t0) + aDash*np.exp(-x/t0Dash)+ offset)

# def fnBiExponentialCommonAmp(x, a, t0, t0Dash, offset):
#     return (a*(np.exp(-x/t0) + np.exp(-x/t0Dash))+ offset)
