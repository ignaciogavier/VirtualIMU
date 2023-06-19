import numpy as np
import torch
import pandas as pd
from utils.signal_process import interp1dTorch, savGolFilterTorch, adaptiveMedFilterTorch

torch.autograd.set_detect_anomaly(True)


def augmentMonteCarlo(
    # File names
    WristsCoordFileName = 'data/wristCoord.csv',
    VidIMUdirName = 'data/augmentedVidIMU/',
    # Temporal variables
    tStartUsr = 3.,
    tStopUsr = 34.,
    samplingRate = 100.,
    # Global coordinates transformations
    globalCoordTransf = torch.tensor([[1, 0, 0], [0, -1, 0], [0, 0, -1]], dtype=torch.double),
    localCoordTransfAccL = torch.tensor([[1, 0, 0], [0, 0, -1], [0, 1, 0]], dtype=torch.double),
    localCoordTransfAccR = torch.tensor([[-1, 0, 0], [0, 0, 1], [0, 1, 0]], dtype=torch.double),
    localCoordTransfGyrL = torch.tensor([[-1, 0, 0], [0, 0, -1], [0, -1, 0]], dtype=torch.double),
    localCoordTransfGyrR = torch.tensor([[1, 0, 0], [0, 0, 1], [0, -1, 0]], dtype=torch.double),
    # Mean sensor alignment
    alphaWMean = torch.tensor([0.6, 0.6], dtype=torch.double),
    alphaFMean = torch.tensor([-1.6, 1.6], dtype=torch.double),
    pWMean = torch.tensor([0.043, 0.043], dtype=torch.double),
    pFMean = torch.tensor([0.075, 0.075], dtype=torch.double),
    # Standard deviation sensor alignment
    alphaWStd = torch.tensor([0.2, 0.2], dtype=torch.double),
    alphaFStd = torch.tensor([0.22, 0.22], dtype=torch.double),
    pWStd = torch.tensor([0.002, 0.002], dtype=torch.double),
    pFStd = torch.tensor([0.003, 0.003], dtype=torch.double),
    # Total number of generations
    totalAug = 100,
    # Others
    verbose = True
):

    timeVec = torch.arange(tStart, tStop, 1 / samplingRate, dtype=torch.float64)
    T = len(timeVec)

    for aug in range(totalAug):
        alphaW = alphaWMean + alphaWStd * torch.randn_like(alphaWStd)
        alphaF = alphaFMean + alphaFStd * torch.randn_like(alphaFStd)
        R0 = torch.stack([
            torch.stack([torch.cos(alphaW), torch.zeros_like(alphaW), -torch.sin(alphaW)]),
            torch.stack([torch.sin(alphaW)*torch.sin(alphaF), torch.cos(alphaF), torch.cos(alphaW)*torch.sin(alphaF)]),
            torch.stack([torch.sin(alphaW)*torch.cos(alphaF), -torch.sin(alphaF), torch.cos(alphaW)*torch.cos(alphaF)])
        ])
        
        pW = pWMean + pWStd * torch.randn_like(pWStd)
        pF = pFMean + pFStd * torch.randn_like(pFStd)
        p0 = torch.stack([-pF, torch.zeros_like(pW), pW])

        # Retrieve wrist coordinates
        dataVid = np.genfromtxt(WristsCoordFileName, skip_header=1, delimiter=',')

        timeVec = torch.tensor(dataVid[:,0])
        deltaT = timeVec[1] - timeVec[0]

        # Define interval of time of interest
        tStart = max([tStartUsr, timeVec[0]])
        tStop = min([tStopUsr, timeVec[-1]])
        
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
        coordinatesLeft = adaptiveMedFilterTorch(coordinatesLeft, M, N, 0)
        coordinatesRight = adaptiveMedFilterTorch(coordinatesRight, M, N, 0)
        rotationsLeft = adaptiveMedFilterTorch(rotationsLeft, M, N, 0)
        rotationsRight = adaptiveMedFilterTorch(rotationsRight, M, N, 0)

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

        # Interpolate to have the same sampling points as the reference
        timeVecNew = torch.linspace(tStart, tStop, T)
        accelLeftResampled = interp1dTorch(timeVec, accelLeftFiltered, timeVecNew).transpose(0,1)
        accelRightResampled = interp1dTorch(timeVec, accelRightFiltered, timeVecNew).transpose(0,1)
        gyroLeftResampled = interp1dTorch(timeVec, gyroLeftFiltered, timeVecNew).transpose(0,1)
        gyroRightResampled = interp1dTorch(timeVec, gyroRightFiltered, timeVecNew).transpose(0,1)
        
        # Write to file
        dataFrame = pd.DataFrame(
            np.concatenate([
                timeVecNew[:,None].detach().numpy(),
                accelLeftResampled.detach().numpy().T, accelRightResampled.detach().numpy().T,
                gyroLeftResampled.detach().numpy().T, gyroRightResampled.detach().numpy().T
            ], axis=1),
            columns=[
                'time', 'acc_xF_left', 'acc_yF_left', 'acc_zF_left',
                'acc_xF_right', 'acc_yF_right', 'acc_zF_right',
                'gyr_xF_left', 'gyr_yF_left', 'gyr_zF_left',
                'gyr_xF_right', 'gyr_yF_right', 'gyr_zF_right'
            ])

        dataFrame.to_csv(VidIMUdirName+f'synthAug{aug}VidIMU.csv', index=False)

        if verbose: print(f'Generated {aug+1}/{totalAug}')
    
    return