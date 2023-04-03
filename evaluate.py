import numpy as np
import pandas as pd


def regression_errors(y, yhat):
    """
    Calculates regression errors and returns the following values:
    sum of squared errors (SSE)
    explained sum of squares (ESS)
    total sum of squares (TSS)
    mean squared error (MSE)
    root mean squared error (RMSE)
    """
    # Calculate SSE, ESS, and TSS
    SSE = np.sum((y - yhat) ** 2)
    ESS = np.sum((yhat - np.mean(y)) ** 2)
    TSS = SSE + ESS

    # Calculate MSE and RMSE
    n = len(y)
    MSE = SSE / n
    RMSE = np.sqrt(MSE)

    return SSE, ESS, TSS, MSE, RMSE


def baseline_mean_errors(y):
    """
    Calculates the errors for the baseline model and returns the following values:
    sum of squared errors (SSE)
    mean squared error (MSE)
    root mean squared error (RMSE)
    """
    # Calculate baseline prediction
    y_mean = np.mean(y)
    yhat_baseline = np.full_like(y, y_mean)

    # Calculate SSE, MSE, and RMSE
    SSE_bl = np.sum((y - yhat_baseline) ** 2)
    n = len(y)
    MSE_bl = SSE / n
    RMSE_bl = np.sqrt(MSE)

    return SSE_bl, MSE_bl, RMSE_bl


def better_than_baseline(y, yhat):
    """
    Checks if your model performs better than the baseline and returns a boolean value.
    """
    # Calculate errors for model and baseline
    SSE_model = np.sum((y - yhat) ** 2)
    SSE_baseline = np.sum((y - np.mean(y)) ** 2)

    # Check if model SSE is less than baseline SSE
    if SSE_model < SSE_baseline:
        return True
    else:
        return False


def plot_residuals(y, yhat):
    """
    Creates a residual plot using matplotlib.
    """
    # Calculate residuals
    residuals = y - yhat

    # Create residual plot
    plt.scatter(yhat, residuals)
    plt.axhline(y=0, color='red', linestyle='--')
    plt.xlabel('Predicted Values')
    plt.ylabel('Residuals')
    plt.title('Residual Plot')
    plt.show()