import torch
from scipy.signal import savgol_coeffs

def interp1dTorch(tOld, xOld, tNew):
    # Calculate slopes between adjacent points
    slopes = (xOld[1:] - xOld[:-1]) / (tOld[1:] - tOld[:-1])[:, None]

    # Find indices of closest values in tOld for tNew
    ids = torch.searchsorted(tOld, tNew)

    # Perform linear interpolation
    return xOld[ids] + slopes[ids - 1] * (tNew - tOld[ids - 1])[:, None]

def savGolFilterTorch(x, windowSize, polyOrder, dim=0, mode='constant'):
    # Generate the Savitzky-Golay filter kernel
    kernel = torch.tensor(savgol_coeffs(windowSize, polyOrder)).unsqueeze(0).unsqueeze(0)

    # Reshape and transpose input tensor
    xT = x.reshape(x.shape[dim], -1).transpose(0, 1).unsqueeze(1)

    # Apply 1D convolution using the kernel
    xFilt = torch.nn.functional.conv1d(xT, kernel, padding='same')[:, 0]

    # Transpose and reshape the filtered tensor back to the original shape
    xFilt = xFilt.transpose(0, 1).reshape(x.shape)

    return xFilt

def adaptiveMedFilterTorch(x, windowSize=3, windowSizeMax=11, dim=0):
    # Ensure dim is a tuple
    if isinstance(dim, int):
        dim = (dim,)

    # Ensure windowSize is a tuple with the same length as dim
    if isinstance(windowSize, int):
        windowSize = len(dim) * (windowSize,)

    # Create a copy of x
    xFilt = torch.clone(x)

    # Reshape and transpose input tensor
    xT = x.reshape(x.shape[dim[0]], -1).transpose(0, 1).unsqueeze(1).unsqueeze(-1)

    # Extract sliding windows from x along the specified dimension
    xStride = torch.nn.functional.unfold(xT, (windowSize[0], 1), padding=(windowSize[0] // 2, 0))

    # Calculate minimum, median, and maximum values within each window
    xMin = torch.min(xStride, 1)[0].transpose(0, 1).reshape(x.shape)
    xMed = torch.median(xStride, 1)[0].transpose(0, 1).reshape(x.shape)
    xMax = torch.max(xStride, 1)[0].transpose(0, 1).reshape(x.shape)

    # Create masks for adaptive filtering
    mask1 = (xMed - xMin > 0) * (xMed - xMax < 0)
    mask2 = (x - xMin > 0) * (x - xMax < 0)

    # Perform adaptive filtering
    xFilt[mask1 * ~mask2] = xMed[mask1 * ~mask2]
    xFilt[~mask1] = adaptiveMedFilterTorch(x, [windowSize[axis] + 2 for axis in range(len(dim))], windowSizeMax, dim)[~mask1] \
        if all([windowSize[axis] <= windowSizeMax for axis in range(len(dim))]) else xMed[~mask1]

    return xFilt