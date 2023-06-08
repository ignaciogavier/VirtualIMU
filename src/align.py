import numpy as np
import torch
import pandas as pd
from utils.signal_process import interp1dTorch, savGolFilterTorch, adaptiveMedFilterTorch

torch.autograd.set_detect_anomaly(True)


def optimizeAlignment(
    # File names
    WristsCoordFileName = 'data/wristCoord.csv',
    IMUfileName = 'data/realIMU.csv',
    VidIMUfileName = 'data/synthVidIMU.csv',
    # Temporal variables
    windowSize = 3000,
    windowStride = 6000,
    tStartUsr = 3.,
    tStopUsr = 34.,
    samplingRate = 100.,
    tShift = torch.tensor([0.1], requires_grad=True, dtype=torch.double),
    # Global coordinates transformations
    globalCoordTransf = torch.tensor([[1, 0, 0], [0, -1, 0], [0, 0, -1]], dtype=torch.double),
    localCoordTransfAccL = torch.tensor([[1, 0, 0], [0, 0, -1], [0, 1, 0]], dtype=torch.double),
    localCoordTransfAccR = torch.tensor([[-1, 0, 0], [0, 0, 1], [0, 1, 0]], dtype=torch.double),
    localCoordTransfGyrL = torch.tensor([[-1, 0, 0], [0, 0, -1], [0, -1, 0]], dtype=torch.double),
    localCoordTransfGyrR = torch.tensor([[1, 0, 0], [0, 0, 1], [0, -1, 0]], dtype=torch.double),
    # Initial sensor alignment
    alphaW = torch.tensor([0.6, 0.6], requires_grad=True, dtype=torch.double),
    alphaF = torch.tensor([-1.6, 1.6], requires_grad=True, dtype=torch.double),
    pW = torch.tensor([0.043, 0.043], requires_grad=True, dtype=torch.double),
    pF = torch.tensor([0.075, 0.075], requires_grad=True, dtype=torch.double),
    # Define optimizer method
    optimizerMethod = torch.optim.Adam,
    # Total number of iterations
    totalIter = 100,
    # Others
    verbose = True
):

    # Initialize optimizer
    optimizer = optimizerMethod([alphaW, alphaF, pW, pF], lr=.1)

    # RMS and NRMS errors
    accelRMSVids = [0 for it in range(totalIter)]
    gyroRMSVids = [0 for it in range(totalIter)]
    accelNRMSVids = [0 for it in range(totalIter)]
    gyroNRMSVids = [0 for it in range(totalIter)]

    for it in range(totalIter):
        optimizer.zero_grad()
        simulatedSignalsVid = []

        alphaW_copy = alphaW.clone()
        alphaF_copy = alphaF.clone()
        R0 = torch.stack([
            torch.stack([torch.cos(alphaW_copy), torch.zeros_like(alphaW_copy), -torch.sin(alphaW_copy)]),
            torch.stack([torch.sin(alphaW_copy)*torch.sin(alphaF_copy), torch.cos(alphaF_copy), torch.cos(alphaW_copy)*torch.sin(alphaF_copy)]),
            torch.stack([torch.sin(alphaW_copy)*torch.cos(alphaF_copy), -torch.sin(alphaF_copy), torch.cos(alphaW_copy)*torch.cos(alphaF_copy)])
        ])
        
        pW_copy = pW.clone()
        pF_copy = pF.clone()
        p0 = torch.stack([-pF_copy, torch.zeros_like(pW_copy), pW_copy])

        # Load the reference IMU signal
        if it == 0:
            realSignals = []
            dataFrameReal = pd.read_csv(IMUfileName)
            
            # Define interval of time of interest
            tStart = np.max([tStartUsr, dataFrameReal['time'].min()])
            tStop = np.min([tStopUsr, dataFrameReal['time'].max()])
            timeVec = torch.arange(tStart.item(), tStop.item(), 1 / samplingRate, dtype=torch.float64)
            T = len(timeVec)
            
            # Apply windowing and convert to tensor
            realSignals.append(np.lib.stride_tricks.sliding_window_view(dataFrameReal.loc[:,'acc_xF_left':'gyr_zF_right'], windowSize, axis=0)[::windowStride])
            realSignals = torch.tensor(np.concatenate(realSignals, axis=0))
        
        # Retrieve wrist coordinates
        dataVid = np.genfromtxt(WristsCoordFileName, skip_header=1, delimiter=',')

        timeVec = torch.tensor(dataVid[:,0])
        deltaT = timeVec[1] - timeVec[0]

        coordinatesLeft = torch.tensor(dataVid[:,1:4])
        coordinatesRight = torch.tensor(dataVid[:,13:16])
        rotationsLeft = torch.tensor(np.stack([dataVid[:,4:7], dataVid[:,7:10], dataVid[:,10:13]], 1))
        rotationsRight = torch.tensor(np.stack([dataVid[:,16:19], dataVid[:,19:22], dataVid[:,22:25]], 1))
        
        # Change of global coordinates MediaPipe [X (left), Y (down), Z (backward)] to Standard [X (left), Y (up), Z (forward)]
        coordinatesLeft = torch.matmul(globalCoordTransf, coordinatesLeft[...,None]).squeeze(2)
        coordinatesRight = torch.matmul(globalCoordTransf, coordinatesRight[...,None]).squeeze(2)
        rotationsLeft = torch.matmul(globalCoordTransf, rotationsLeft)
        rotationsRight = torch.matmul(globalCoordTransf, rotationsRight)

        # Sensor alignment (position protocol and geometric forearm features)
        rotationsLeft = torch.matmul(rotationsLeft, R0[...,0])
        rotationsRight = torch.matmul(rotationsRight, R0[...,1])
        coordinatesLeft += torch.matmul(rotationsLeft, p0[...,0][...,None]).squeeze(2)
        coordinatesRight += torch.matmul(rotationsRight, p0[...,1][...,None]).squeeze(2)

        # Savitzky-Golay filter
        M, N = 31, 3
        coordinatesLeft = savGolFilterTorch(coordinatesLeft, M, N, dim=0)
        coordinatesRight = savGolFilterTorch(coordinatesRight, M, N, dim=0)
        rotationsLeft = savGolFilterTorch(rotationsLeft, M, N, dim=0)
        rotationsRight = savGolFilterTorch(rotationsRight, M, N, dim=0)

        # Adaptive median filter
        M, N = 3, 11
        coordinatesLeft = adaptiveMedFilterTorch(coordinatesLeft, M, N, 0, 'constant')
        coordinatesRight = adaptiveMedFilterTorch(coordinatesRight, M, N, 0, 'constant')
        rotationsLeft = adaptiveMedFilterTorch(rotationsLeft, M, N, 0, 'constant')
        rotationsRight = adaptiveMedFilterTorch(rotationsRight, M, N, 0, 'constant')

        # Derivatives
        accelLeft = torch.gradient(torch.gradient(coordinatesLeft, spacing=deltaT, edge_order=2, dim=0)[0], spacing=deltaT, edge_order=2, dim=0)[0]
        accelLeft += torch.tensor([[0., 9.81, 0.]]) # Gravity
        accelLeft = torch.matmul(rotationsLeft.transpose(1,2), accelLeft[...,None]).squeeze(2)
        rotationsLeftDer = torch.gradient(rotationsLeft, spacing=deltaT, edge_order=2, dim=0)[0]
        gyroLeft = torch.matmul(rotationsLeft.transpose(1,2), rotationsLeftDer)
        gyroLeft = torch.stack([
            (gyroLeft[:,2,1] - gyroLeft[:,1,2]) / 2,
            (gyroLeft[:,0,2] - gyroLeft[:,2,0]) / 2,
            (gyroLeft[:,1,0] - gyroLeft[:,0,1]) / 2
        ], dim=1) * 180 / np.pi # In degrees

        accelRight = torch.gradient(torch.gradient(coordinatesRight, spacing=deltaT, edge_order=2, dim=0)[0], spacing=deltaT, edge_order=2, dim=0)[0]
        accelRight += torch.tensor([[0., 9.81, 0.]]) # Gravity
        accelRight = torch.matmul(rotationsRight.transpose(1,2), accelRight[...,None,]).squeeze(2)
        rotationsRightDer = torch.gradient(rotationsRight, spacing=deltaT, edge_order=2, dim=0)[0]
        gyroRight = torch.matmul(rotationsRight.transpose(1,2), rotationsRightDer)
        gyroRight = torch.stack([
            (gyroRight[:,2,1] - gyroRight[:,1,2]) / 2,
            (gyroRight[:,0,2] - gyroRight[:,2,0]) / 2,
            (gyroRight[:,1,0] - gyroRight[:,0,1]) / 2
        ], dim=1) * 180 / np.pi # In degrees

        # Align with IMU axes
        accelLeft = torch.matmul(localCoordTransfAccL, accelLeft[..., None]).squeeze(2)
        accelRight = torch.matmul(localCoordTransfAccR, accelRight[..., None]).squeeze(2)
        gyroLeft = torch.matmul(localCoordTransfGyrL, gyroLeft[..., None]).squeeze(2)
        gyroRight = torch.matmul(localCoordTransfGyrR, gyroRight[..., None]).squeeze(2)

        # Savitzky-Golay filter
        M, N = 31, 3
        accelLeftFiltered = savGolFilterTorch(accelLeft, M, N, dim=0)
        accelRightFiltered = savGolFilterTorch(accelRight, M, N, dim=0)
        gyroLeftFiltered = savGolFilterTorch(gyroLeft, M, N, dim=0)
        gyroRightFiltered = savGolFilterTorch(gyroRight, M, N, dim=0)

        # Interpolate to have the same sampling points as the real IMU
        timeVecNew = torch.linspace(tStart, tStop, T) - tShift
        accelLeftResampled = interp1dTorch(timeVec, accelLeftFiltered, timeVecNew).transpose(0,1)
        accelRightResampled = interp1dTorch(timeVec, accelRightFiltered, timeVecNew).transpose(0,1)
        gyroLeftResampled = interp1dTorch(timeVec, gyroLeftFiltered, timeVecNew).transpose(0,1)
        gyroRightResampled = interp1dTorch(timeVec, gyroRightFiltered, timeVecNew).transpose(0,1)
        
        # Write to file
        dataFrame = pd.DataFrame(
            np.concatenate([
                timeVec[:,None].detach().numpy(),
                accelLeftResampled.detach().numpy(), accelRightResampled.detach().numpy(),
                gyroLeftResampled.detach().numpy(), gyroRightResampled.detach().numpy()
            ], axis=1),
            columns=[
                'time', 'acc_xF_left', 'acc_yF_left', 'acc_zF_left',
                'acc_xF_right', 'acc_yF_right', 'acc_zF_right',
                'gyr_xF_left', 'gyr_yF_left', 'gyr_zF_left',
                'gyr_xF_right', 'gyr_yF_right', 'gyr_zF_right'
            ])

        dataFrame.to_csv(VidIMUfileName, index=False)

        # Put them in a tensor
        simulatedSignalsVid.append(torch.cat([
            accelLeftResampled, accelRightResampled,
            gyroLeftResampled, gyroRightResampled
        ], dim=0)[...,:windowSize])

        simulatedSignalsVid = torch.stack(simulatedSignalsVid, dim=0)
        
        # Redefine temporal 
        timeVec = torch.arange(0, windowSize / samplingRate, 1 / samplingRate, dtype=torch.float64)
        T = len(timeVec)
        
        # Calculate RMS and NRMS errors
        accelRMSVid = torch.sqrt(torch.nanmean((realSignals[:,:3,:] - simulatedSignalsVid[:,:3,:]) ** 2, dim=(1,2)))
        gyroRMSVid = torch.sqrt(torch.nanmean((realSignals[:,6:9,:] - simulatedSignalsVid[:,6:9,:]) ** 2, dim=(1,2)))
        
        accelNRMSVid = torch.sqrt(torch.nanmean((realSignals[:,:3,:] - simulatedSignalsVid[:,:3,:]) ** 2 / (torch.max(realSignals[:,:3,:]) ** 2 + 1e-5), dim=(1,2)))
        gyroNRMSVid = torch.sqrt(torch.nanmean((realSignals[:,6:9,:] - simulatedSignalsVid[:,6:9,:]) ** 2 / (torch.max(realSignals[:,6:9,:]) ** 2 + 1e-5), dim=(1,2)))
        
        # Store them in a list
        accelRMSVids[it] = accelRMSVid.item()
        gyroRMSVids[it] = gyroRMSVid.item()
        
        accelNRMSVids[it] = accelNRMSVid.item()
        gyroNRMSVids[it] = gyroNRMSVid.item()
            
        objective = 5 * accelNRMSVid.mean() + 1 * gyroNRMSVid.mean()
        
        if verbose:
            print(f'Iteration: {it}')
            print(f'\t\t\taccelNRMSVid: {accelNRMSVid.data.numpy()}')
            print(f'\t\t\tgyroNRMSVid: {gyroNRMSVid.data.numpy()}')
            print(f'\t\t\talphaW: {alphaW.detach().numpy()}')
            print(f'\t\t\talphaF: {alphaF.detach().numpy()}')
            print(f'\t\t\tpW: {pW.detach().numpy()}')
            print(f'\t\t\tpF: {pF.detach().numpy()}')
        
        objective.backward(retain_graph=True)
        optimizer.step()
    
    return alphaW, alphaF, pW, pF