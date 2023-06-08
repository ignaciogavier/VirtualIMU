import numpy as np
import pandas as pd
from scipy.interpolate import interp1d


def preprocessIMU(
    # File names
    IMUfileNameLeft = '/data/rawIMU/shimmer_left_wrist_filtered.csv',
    IMUfileNameRight = '/data/rawIMU/shimmer_right_wrist_filtered.csv',
    IMUfileName = 'data/realIMU.csv',
    # Temporal variables
    tStartUsr = -np.inf,
    tStopUsr = np.inf,
    samplingRate = 100.,
):

    # Load raw IMU dataframes
    dataFrameLeft = pd.read_csv(IMUfileNameLeft, index_col=0)
    dataFrameRight = pd.read_csv(IMUfileNameRight, index_col=0)

    tStart = np.max([tStartUsr, dataFrameLeft['time'].min(), dataFrameRight['time'].min()])
    tStop = np.min([tStopUsr, dataFrameLeft['time'].max(), dataFrameRight['time'].max()])
    timeVec = np.arange(tStart, tStop, 1 / samplingRate, dtype=np.float128)

    dataFrameReal = pd.DataFrame(timeVec, columns=['time'])
    for type in ['acc_', 'gyr_']:
        for axis in ['x', 'y', 'z']:
            f = interp1d(dataFrameLeft['time'], dataFrameLeft[type+axis], kind='slinear')
            dataFrameReal[type+axis+'F_left'] = f(timeVec)
        for axis in ['x', 'y', 'z']:
            f = interp1d(dataFrameRight['time'], dataFrameRight[type+axis], kind='slinear')
            dataFrameReal[type+axis+'F_right'] = f(timeVec)

    dataFrameReal.to_csv(IMUfileName, index=False)

    return
