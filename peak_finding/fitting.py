"""Code to reformat the csv/json data files containing information from the laser sweep, and
to fit the data to a Lorentzian distribution and plot.

To run this Python script, type the following line in the terminal (this is just an example;
please modify with your customized paths):
python fitting.py --data_path ../sample_data/B1D73.csv --plot_path /Users/diangenlin/Desktop/B1D73-20230206.jpg

One fit for which quadratic subtraction of background fails.
python fitting.py --data_path ../sample_data/B1D73.csv --plot_path /Users/diangenlin/Desktop/B1D73-20230206.jpg
"""

import fitFunctions as fit
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from argparse import ArgumentParser, Namespace
from pathlib import Path
from scipy.optimize import curve_fit
from typing import Callable, List, Optional, Tuple

# preset figure size, font size and font type
plt.rcParams["figure.figsize"] = [8, 6]
plt.rcParams.update({"font.size": 11})
plt.rcParams.update({"font.family": "Arial"})


def format(csv_path: Path) -> Tuple[np.ndarray, np.ndarray]:
    """iven a csv file path, reformat the csv to obtain only the wavelengths and
    corresponding laser powers.

    Parameters
    ----------
    csv_path : Path
        Path to the csv file.

    Returns
    -------
    Tuple[np.ndarray, np.ndarray]
        X: Laser wavelength in nm.
        Y: Detected power in dBm.
    """

    df = pd.read_csv(
        filepath_or_buffer=csv_path, skiprows=16, index_col=False, header=None
    )
    df = df.transpose()  # to make data values vertical instead of horizontal
    df = df.iloc[1:, :]
    X = np.array(df[0])  # laser wavelength in nm
    Y = np.array(df[1])  # detected power in dBm
    # Y = POW  # (POW-np.max(POW))/(np.max(POW)-np.min(POW))  ## normalize power
    # X = 2.9979e8 / (WL * 1e-9) * 1e-12  # converts from WL in nm to frequency in THz.
    return X, Y


def get_effective_data(
    X: np.ndarray,
    Y: np.ndarray,
    close_dist_from_peak: float = 0.25,
    far_dist_from_peak: float = 1,
) -> Tuple[Tuple[np.ndarray, np.ndarray], Tuple[np.ndarray, np.ndarray]]:
    """Given sweep data of laser wavelength with corresponding laser power, determine
    where the Lorentzian peak is approximately going to be, and get the effective background and data
    for easier background subtraction. We define the effective background to be approximately
    a distance of close_dist_from_peak to a distance of far_dist_from_peak from either side
    of the peak.

    Assume that the Lorentzian peak is at the wavelength with the minimum detected power.

    To select close or far dist from peak, get a sense of the maximum width of the peak from
    actual data as well as how far out from the peak we should consider the oscillatory background
    noise.

    Parameters
    ----------
    X : np.ndarray
        Laser wavelength in nm.
    Y : np.ndarray
        Detected power in dBm.
    close_dist_from_peak : float, optional
        Close distance from peak, by default 0.25 nm. Must be a positive value.
    far_dist_from_peak : float, optional
        Far distance from peak, by default 1 nm. far_dist_from_peak must be greater than
        close_dist_from_peak and must be a positive value.

    Returns
    -------
    Tuple[Tuple[np.ndarray, np.ndarray], Tuple[np.ndarray, np.ndarray]]
        X_eff_bkgd: Effective background laser wavelength in nm.
        Y_eff_bkgd: Effecitve background detected power in dBm.
        X_eff: Effective laser wavelength in nm.
        Y_eff: Effective detected powers in dBm.
    """

    # TO-DO:
    # Note that in the current version of the function, we assumed the Lorentzian peak
    # to be at the wavelength with the minimum detected power. This assumption may not be correct
    # because there could be an outlier data point that has the minimum detected power, but
    # does not actually reflect the actual peak of the curve.
    # A better way to do this is to use the second derivative to fit a concave up curvature
    # on the data, to determine where the actual peak is likely to be.

    # index of peak wavelength in data; note: only index of the first match is returned
    # assume indices of a wavelength and its corresponding detected power are the same in X and Y data
    ind_peak_x = list(Y).index(np.min(Y))

    # wavelength step size is 0.01 nm; that is, x data points are spaced 0.01 nm apart.
    # wavelength step size can be obtained from uncleaned csv data from Fotonica.
    wl_step_size = 0.01
    inds_from_peak_x_close = int(
        np.ceil(close_dist_from_peak / wl_step_size)
    )  # number of indices away from index of peak_x (close)
    inds_from_peak_x_far = int(
        np.ceil(far_dist_from_peak / wl_step_size)
    )  # number of indices away from index of peak_x (far)

    # index of effective background laser wavelength to the left of peak
    far_left = ind_peak_x - inds_from_peak_x_far
    close_left = ind_peak_x - inds_from_peak_x_close

    # index of effective background laser wavelength to the right of peak
    close_right = ind_peak_x + inds_from_peak_x_close
    far_right = ind_peak_x + inds_from_peak_x_far

    # effective background, laser wavelength
    X_eff_bkgd_left = X[far_left:close_left]
    X_eff_bkgd_right = X[close_right:far_right]
    X_eff_bkgd = np.concatenate((X_eff_bkgd_left, X_eff_bkgd_right))

    # effective background, detected powers
    Y_eff_bkgd_left = Y[far_left:close_left]
    Y_eff_bkgd_right = Y[close_right:far_right]
    Y_eff_bkgd = np.concatenate((Y_eff_bkgd_left, Y_eff_bkgd_right))

    # effective data
    X_eff = X[far_left:far_right]
    Y_eff = Y[far_left:far_right]

    return (X_eff_bkgd, Y_eff_bkgd), (X_eff, Y_eff)


def fit_background(
    X_eff: np.ndarray, Y_eff: np.ndarray, X_eff_bkgd: np.ndarray, Y_eff_bkgd: np.ndarray
) -> Tuple:
    """Fit effective background data with a quadratic function.

    Parameters
    ----------
    X_eff : np.ndarray
        Effective laser wavelength in nm (peak section included).
    Y_eff : np.ndarray
        Effective detected powers in dBm.
    X_eff_bkgd : np.ndarray
        Effective background laser wavelength in nm.
    Y_eff_bkgd : np.ndarray
        Effecitve background detected power in dBm.

    Returns
    -------
    Tuple[np.ndarray, float]
        Y_fit_bkgd: np.ndarray
            Quadratic fit of the effective background data.
        r2_bkgd: float
            R squared representing the goodness of fit, rounded to 3 decimal places.
    """
    # try perform quadratic fitting of the background data;
    # if there is an error because scipy cannot find optimal parameters for quadratic fit,
    # use linear fit instead

    # fit the background to a quadratic function
    # y = a * ((x - h) ** 2) + k
    try:
        popt = curve_fit(fit.quadratic, X_eff_bkgd, Y_eff_bkgd)[0]
        a_fit = popt[0]
        h_fit = popt[1]
        k_fit = popt[2]
        # over the domain of all effective (truncated data, including peak)
        Y_fit = fit.quadratic(x=X_eff, a=a_fit, h=h_fit, k=k_fit)
        # over only the effective background; calculate again to calcuate R^2
        Y_fit_bkgd = fit.quadratic(x=X_eff_bkgd, a=a_fit, h=h_fit, k=k_fit)
        bkgd_fit_type = "Quadratic"
        print("Background fit with quadratic fitting.")
    except:
        popt = curve_fit(fit.linear, X_eff_bkgd, Y_eff_bkgd)[0]
        a_fit = popt[0]  # slope
        b_fit = popt[1]  # intercept
        # over the domain of all effective (truncated data, including peak)
        Y_fit = fit.linear(x=X_eff, a=a_fit, b=b_fit)
        # over only the effective background; calculate again to calcuate R^2
        Y_fit_bkgd = fit.linear(x=X_eff_bkgd, a=a_fit, b=b_fit)
        bkgd_fit_type = "Linear"
        print("Background fit with linear fitting.")

    # calculate R^2
    r2_bkgd = 1 - np.var(Y_eff_bkgd - Y_fit_bkgd) / np.var(Y_eff_bkgd)

    return Y_fit, r2_bkgd, bkgd_fit_type


def subtract_background(Y_eff: np.ndarray, Y_fit_bkgd: np.ndarray) -> np.ndarray:
    """Remove backgroumd noise from Fabry-Perot oscillations, to make Lorenztian fitting easier later.
    Subtract the original Y data from the fit background

    Parameters
    ----------
    Y : np.ndarray
        Detected power in dBm.
    Y_fit_bkgd : np.ndarray
        Quadratic fit of the effective background data.

    Returns
    -------
    np.ndarray
        Detected power with effective background subtracted.
    """
    Y_bkgd_subtracted = Y_eff - Y_fit_bkgd
    return Y_bkgd_subtracted


def get_fit_bounds(
    X: np.ndarray, Y: np.ndarray
) -> Tuple[Tuple[float, float, float, float], Tuple[float, float, float, float]]:
    """Get the Lorenztian fit bounds, specifically for the function fitFunctions.fnLorentzianX0.

    Parameters
    ----------
    X : np.ndarray
        X data to be fit (laser wavelength in nm).
    Y : np.ndarray
        Corresponding Y data to be fit (detected powers in dBm).

    Returns
    -------
    Tuple[Tuple[float, float, float, float], Tuple[float, float, float, float]]
        Lower and upper bounds for fit parameters.
    """
    # for fnLorentzianX0(x, a, gamma, offset, x0)
    aMin, aMax = -np.inf, 0  # 5 * min(Y), 0 # prefactor spread
    gammaMin, gammaMax = 0, 0.5 * (max(X) - min(X))  # 0.0001, max(X) - min(X)
    offsetMin, offsetMax = -np.inf, np.inf  # -1, 0
    x0Min, x0Max = X[np.argmin(Y)] - 0.01 * (max(X) - min(X)), X[
        np.argmin(Y)
    ] + 0.01 * (
        max(X) - min(X)
    )  # min(X), max(X)
    bounds = ((aMin, gammaMin, offsetMin, x0Min), (aMax, gammaMax, offsetMax, x0Max))
    print("Lorentzian fit parameters:")
    print("Fit bounds: ", bounds)
    return bounds


def get_initial_params(X: np.ndarray, Y: np.ndarray) -> List[float]:
    """Create initial guess for Lorentzian parameters, specifically for the function fitFunctions.fnLorentzianX0.

    Parameters
    ----------
    X : np.ndarray
        X data to be fit (laser wavelength in nm).
    Y : np.ndarray
        Corresponding Y data to be fit (detected powers in dBm).

    Returns
    -------
    List[float]
        Initial guesses for each of the Lorentzian parameters: a0, gamma0, offset0, x00.
    """
    # currently params are just input manually based on the graph, but this should be updated to make an informed guess.
    a0 = -1
    gamma0 = 1e-3 * (max(X) - min(X))
    offset0 = 0
    x00 = X[np.argmin(Y)]
    p0 = [a0, gamma0, offset0, x00]
    print("Initial params [a0, gamma0, offset0, x00]: ", p0)
    return p0


def fit_lorentzian(
    X: np.ndarray,
    Y: np.ndarray,
    p0: List[float],
    bounds: Tuple[Tuple[float, float, float, float], Tuple[float, float, float, float]],
    bkgd_fit_type: str,
    plot: bool = True,
    plot_path: Optional[Path] = None,
) -> Tuple[np.ndarray, np.ndarray, float, float]:
    """Fit data to a Lorentzian distribution, calculate the goodness of fit by R^2, and
    calculate the Q factor. If user specifies to plot the Lorentzian fit with the original
    data, then plot.

    The Q factor is defined by the resonant frequency / full width at half max (FWHM), unitless.
    It is a measure of the rate of energy loss and thus how fast the oscillation dies out.
    Higher Q: lower rate of energy loss, oscillation dies out slowly.
    Lower Q: higher rate of energy loss, oscillation dies out quickly.

    Parameters
    ----------
    lorentzian : Callable
        Callable Lorentzian function.
    X : np.ndarray
        Independent data, laser frequency in THz.
    Y : np.ndarray
        Dependent data, laser power in dBm.
    p0 : List[float]
        Curve_fit's initial guess.
    bounds : Tuple[Tuple[float, float, float, float], Tuple[float, float, float, float]]
        Lower and upper bounds for fit parameters.
    plot : bool
        Specify whether to plot the original data with the Lorentzian fit.
    plot_path : Optional[Path], optional
        Path to save plot.

    Returns
    -------
    np.ndarray
        Optimal parameter values of x0 (peak of distribution) and gamma (FWHM).
    """
    # popt: array, optimal values for the parameters which minize the sum of the square of residuals
    # of the callable function
    # pcov: 2-D array, estimated covariance of popt
    popt = curve_fit(f=fit.fnLorentzianX0, xdata=X, ydata=Y, p0=p0, bounds=bounds)[0]
    a_fit = popt[0]
    gamma_fit = popt[1]
    offset_fit = popt[2]
    x0_fit = popt[3]

    # calculate goodness of fit using R^2
    Y_fit = fit.fnLorentzianX0(X, a_fit, gamma_fit, offset_fit, x0_fit)

    # calculate R^2
    r2 = round(1 - np.var(Y - Y_fit) / np.var(Y), 3)

    # calculate the Q factor: Q = resonant frequency/FWHM = freq(x0) / freq(gamma)
    res_freq = 2.9979e8 / (x0_fit * 1e-9)  # convert wavelength to Hz
    FWHM = 2.9979e8 / (gamma_fit * 1e-9)  # convert wavelength to Hz

    print("x0_fit:", x0_fit)
    print("gamma_fit:", gamma_fit)
    print("resonant frequency (Hz):", res_freq)
    print("FWHM (Hz):", FWHM)

    Q = round(x0_fit / gamma_fit, 3)  # Q factor, round to 3 decimal places
    print("r2:", r2)
    print("Q factor:", Q)

    if plot:
        # plot the original data
        plt.scatter(X, Y, color="black", marker=".", label="Data")
        # plot Lorentzian fit
        plt.plot(
            X,
            Y_fit,
            color="red",
            label=f"Lorentzian fit \n x0={round(x0_fit,3)} \n R2={r2} \n Q={Q} \n Bkgd fit type: {bkgd_fit_type}",
        )
        plt.xlabel("Wavelength (nm)")
        plt.ylabel("Background Subtracted Detected Power (dBm)")
        plt.legend()
        # plt.show()
        print("Finished plotting Lorentzian")

        if plot_path is not None:
            plt.savefig(fname=plot_path, dpi=300)
        print("Finished saving plot")

    return Y_fit, popt, r2, Q


def fit_and_plot(data_path: Path, plot_path: Optional[Path] = None) -> Tuple:
    """Get data from CSV file, perform background subtraction, fit to Lorentzian, and plot.
    This function integrates everything in this script.

    Parameters
    ----------
    data_path : Path
        Path to get CSV data.
    plot_path : Optional[Path], optional
        Path to save plots, by default None

    """
    X, Y = format(csv_path=data_path)
    result = get_effective_data(X, Y)
    X_eff_bkgd = result[0][0]
    Y_eff_bkgd = result[0][1]
    X_eff = result[1][0]
    Y_eff = result[1][1]
    Y_fit_bkgd, r2_bkgd, bkgd_fit_type = fit_background(
        X_eff, Y_eff, X_eff_bkgd, Y_eff_bkgd
    )
    Y_bkgd_subtracted = subtract_background(Y_eff, Y_fit_bkgd)

    # fit Lorentzian
    bounds = get_fit_bounds(X_eff, Y_bkgd_subtracted)
    p0 = get_initial_params(X_eff, Y_bkgd_subtracted)
    Y_fit, Lorentzian_popt, r2, Q = fit_lorentzian(
        X_eff,
        Y_bkgd_subtracted,
        p0,
        bounds,
        bkgd_fit_type=bkgd_fit_type,
        plot=True,
        plot_path=plot_path,
    )
    return X, Y, Y_fit, Lorentzian_popt, r2, Q


def parse_args() -> Namespace:
    """Parse command line arguments.
    Returns
    -------
    Namespace
        Parsed arguments.
    """
    parser = ArgumentParser(description="Analyze and visualize sweep data.")

    parser.add_argument(
        "--data_path",
        type=Path,
        required=True,
        help="Path to access data file containing sweep results.",
    )
    parser.add_argument(
        "--plot_path",
        type=Path,
        required=True,
        help="Path to save Lorentzian fit plot.",
    )
    return parser.parse_args()


def main() -> None:
    X, Y, Y_fit, Lorentzian_popt, r2, Q = fit_and_plot(
        data_path=args.data_path, plot_path=args.plot_path
    )


if __name__ == "__main__":
    args = parse_args()
    main()
