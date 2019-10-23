import matplotlib.pyplot as plt
import numpy as np

def make_meshgrid(x, y, h=.001):
    """Create a mesh of points to plot in

    Parameters
    ----------
    x: data to base x-axis meshgrid on
    y: data to base y-axis meshgrid on
    h: stepsize for meshgrid, optional

    Returns
    -------
    xx, yy : ndarray
    """
    x_min, x_max = -0.05, 1.05
    y_min, y_max = -0.05, 1.05
    xx, yy = np.meshgrid(np.arange(x_min, x_max, h),
                         np.arange(y_min, y_max, h))
    return xx, yy


def plot_contours(ax, model, xx, yy, **params):
    """Plot the decision boundaries for a classifier.

    Parameters
    ----------
    ax: matplotlib axes object
    clf: a classifier
    xx: meshgrid ndarray
    yy: meshgrid ndarray
    params: dictionary of params to pass to contourf, optional
    """
    X = pd.DataFrame
    model.eval()
    Z = model(input_data)
    model.train()
    Z = torch.round(Z)
    Z = Z.reshape(xx.shape)    
    out = ax.contourf(xx, yy, Z.detach().numpy(), **params)
    return out, Z
    