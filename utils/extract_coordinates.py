import csv
import cv2
import mediapipe as mp
import numpy as np
import pandas as pd

def extractWristCoordinatesVideo(
        videoFile = 'data/video.mp4',
        coordinatesFile = 'data/wristCoord.csv'
):
    # Set up the Mediapipe pose model
    mediaPipePose = mp.solutions.pose
    pose = mediaPipePose.Pose(static_image_mode=False, min_detection_confidence=0.5, min_tracking_confidence=0.5, model_complexity=0)

    # Set up the drawing tools
    mediaPipeDrawing = mp.solutions.drawing_utils
    drawSpec = mediaPipeDrawing.DrawingSpec(color=(13, 218, 253), thickness=4, circle_radius=3)

    # Open the video file
    videoCapture = cv2.VideoCapture(videoFile)

    # Open the CSV file
    csvFile = open(coordinatesFile, 'w', newline='')
    csvWriter = csv.writer(csvFile)

    # Write the CSV header row
    csvWriter.writerow([
        'frame',
        'p1L','p2L','p3L', # position left
        'R11L','R12L','R13L', # rotation left
        'R21L','R22L','R23L',
        'R31L','R32L','R33L',
        'p1R','p2R','p3R', # position right
        'R11R','R12R','R13R', # rotation right
        'R21R','R22R','R23R',
        'R31R','R32R','R33R'
    ])

    # Loop through the video frames
    while videoCapture.isOpened():
        # Read a frame from the video
        retVal, image = videoCapture.read()

        if not retVal:
            break

        # Convert the image to RGB format
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        # Process the image with Mediapipe
        results = pose.process(image)

        # Extract the pose landmarks and pose world landmarks from the results
        poseLandmarks = results.pose_landmarks
        poseWorldLandmarks = results.pose_world_landmarks

        # Skip the iteration if pose landmarks or pose world landmarks are missing
        if poseLandmarks is None or poseWorldLandmarks is None:
            continue

        # Skip the iteration if any of the required landmarks have low visibility
        if (
            poseWorldLandmarks.landmark[mediaPipePose.PoseLandmark.LEFT_SHOULDER].visibility < 0.6 or
            poseWorldLandmarks.landmark[mediaPipePose.PoseLandmark.LEFT_ELBOW].visibility < 0.6 or
            poseWorldLandmarks.landmark[mediaPipePose.PoseLandmark.LEFT_THUMB].visibility < 0.6 or
            poseWorldLandmarks.landmark[mediaPipePose.PoseLandmark.RIGHT_SHOULDER].visibility < 0.6 or
            poseWorldLandmarks.landmark[mediaPipePose.PoseLandmark.RIGHT_ELBOW].visibility < 0.6 or
            poseWorldLandmarks.landmark[mediaPipePose.PoseLandmark.RIGHT_THUMB].visibility < 0.6
        ):
            continue
        
        # Get the position of the left and right thumbs
        leftThumbPosition = np.array([
            poseWorldLandmarks.landmark[mediaPipePose.PoseLandmark.LEFT_THUMB].x,
            poseWorldLandmarks.landmark[mediaPipePose.PoseLandmark.LEFT_THUMB].y,
            poseWorldLandmarks.landmark[mediaPipePose.PoseLandmark.LEFT_THUMB].z
        ])
        rightThumbPosition = np.array([
            poseWorldLandmarks.landmark[mediaPipePose.PoseLandmark.RIGHT_THUMB].x,
            poseWorldLandmarks.landmark[mediaPipePose.PoseLandmark.RIGHT_THUMB].y,
            poseWorldLandmarks.landmark[mediaPipePose.PoseLandmark.RIGHT_THUMB].z
        ])

        # Get the position of the left and right wrists
        leftWristPosition = np.array([
            poseWorldLandmarks.landmark[mediaPipePose.PoseLandmark.LEFT_WRIST].x,
            poseWorldLandmarks.landmark[mediaPipePose.PoseLandmark.LEFT_WRIST].y,
            poseWorldLandmarks.landmark[mediaPipePose.PoseLandmark.LEFT_WRIST].z
        ])
        rightWristPosition = np.array([
            poseWorldLandmarks.landmark[mediaPipePose.PoseLandmark.RIGHT_WRIST].x,
            poseWorldLandmarks.landmark[mediaPipePose.PoseLandmark.RIGHT_WRIST].y,
            poseWorldLandmarks.landmark[mediaPipePose.PoseLandmark.RIGHT_WRIST].z
        ])

        # Get the position of the left and right elbows
        leftElbowPosition = np.array([
            poseWorldLandmarks.landmark[mediaPipePose.PoseLandmark.LEFT_ELBOW].x,
            poseWorldLandmarks.landmark[mediaPipePose.PoseLandmark.LEFT_ELBOW].y,
            poseWorldLandmarks.landmark[mediaPipePose.PoseLandmark.LEFT_ELBOW].z
        ])
        rightElbowPosition = np.array([
            poseWorldLandmarks.landmark[mediaPipePose.PoseLandmark.RIGHT_ELBOW].x,
            poseWorldLandmarks.landmark[mediaPipePose.PoseLandmark.RIGHT_ELBOW].y,
            poseWorldLandmarks.landmark[mediaPipePose.PoseLandmark.RIGHT_ELBOW].z
        ])

        # Calculate the vectors between elbows and wrists
        leftElbowWrist = leftWristPosition - leftElbowPosition
        rightElbowWrist = rightWristPosition - rightElbowPosition

        # Calculate the vectors between wrists and thumbs
        leftWristThumb = leftThumbPosition - leftWristPosition
        rightWristThumb = rightThumbPosition - rightWristPosition

        # Calculate the normalized X, Y, and Z axes for left wrist rotation
        leftVersorX = leftElbowWrist / np.linalg.norm(leftElbowWrist)
        leftVersorZ = np.cross(leftWristThumb, leftElbowWrist)
        leftVersorZ /= np.linalg.norm(leftVersorZ)
        leftVersorY = np.cross(leftVersorZ, leftVersorX)

        # Create the left wrist rotation matrix
        leftWristRotation = np.stack([leftVersorX, leftVersorY, leftVersorZ], axis=1)

        # Calculate the normalized X, Y, and Z axes for right wrist rotation
        rightVersorX = - rightElbowWrist / np.linalg.norm(rightElbowWrist)
        rightVersorZ = - np.cross(rightWristThumb, rightElbowWrist)
        rightVersorZ /= np.linalg.norm(rightVersorZ)
        rightVersorY = np.cross(rightVersorZ, rightVersorX)

        # Create the right wrist rotation matrix
        rightWristRotation = np.stack([rightVersorX, rightVersorY, rightVersorZ], axis=1)
        
        # Write the data to the CSV file
        csvWriter.writerow([
            videoCapture.get(cv2.CAP_PROP_POS_FRAMES),
            *[t for t in leftWristPosition.flatten()],
            *[t for t in leftWristRotation.flatten()],
            *[t for t in rightWristPosition.flatten()],
            *[t for t in rightWristRotation.flatten()],
        ])

        # Draw the pose landmarks on the image
        image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
        mediaPipeDrawing.draw_landmarks(
            image, results.pose_landmarks,
            mediaPipePose.POSE_CONNECTIONS,
            drawSpec, drawSpec
        )

        # Show the image with the pose landmarks
        cv2.imshow('Pose Detection', image)

        # Exit the loop if the 'q' key is pressed
        if cv2.waitKey(1) == ord('q'):
            break


def extractWristCoordinatesBVH(
        BVHFile = 'data/capture.bvh',
        coordinatesFile = 'data/wristCoord.csv'
):
    import bvhtoolbox as bvh
    import transforms3d as t3d

    # Open the BVH file
    with open(BVHFile) as f:
        BVHTree = bvh.BvhTree(f.read())
    
    # Compute Forward Kinematics
    root = next(BVHTree.root.filter('ROOT'))
    def get_world_positions(joint):
        if joint.value[0] == 'End':
            joint.world_transforms = np.tile(t3d.affines.compose(np.zeros(3), np.eye(3), np.ones(3)), (BVHTree.nframes, 1, 1))
        else:
            channels = BVHTree.joint_channels(joint.name)
            axes_order = ''.join([ch[:1] for ch in channels if ch[1:] == 'rotation']).lower()  # FixMe: This isn't going to work when not all rotation channels are present
            axes_order = 's' + axes_order[::-1]
            joint.world_transforms = bvh.get_affines(BVHTree, joint.name, axes=axes_order)
            
        if joint != root:
            # For joints substitute position for offsets.
            offset = [float(o) for o in joint['OFFSET']]
            joint.world_transforms[:, :3, 3] = offset
            joint.world_transforms = np.matmul(joint.parent.world_transforms, joint.world_transforms)
        
        end = list(joint.filter('End'))
        if end:
            get_world_positions(end[0])  # There can be only one End Site per joint.
        for child in joint.filter('JOINT'):
            get_world_positions(child)
    
    get_world_positions(root)
    
    # Get time vector
    sampleRate = int(1 / BVHTree.frame_time)
    timeVec = np.arange(BVHTree.nframes, dtype=float)[:,None] / sampleRate
    
    # Get transforms
    leftWristPosition = BVHTree.search('End', 'LeftForeArm_End')[0].world_transforms[:,:3,3] / 100
    rightWristPosition = BVHTree.search('End', 'RightForeArm_End')[0].world_transforms[:,:3,3] / 100
    leftWristRotation = BVHTree.search('End', 'LeftForeArm_End')[0].world_transforms[:,:3,:3].reshape(-1,9)
    rightWristRotation = BVHTree.search('End', 'RightForeArm_End')[0].world_transforms[:,:3,:3].reshape(-1,9)
    
    # Write the data to the CSV file
    data = pd.DataFrame(
        np.concatenate([
            timeVec,
            leftWristPosition,
            leftWristRotation,
            rightWristPosition,
            rightWristRotation
        ], axis=1),
        columns = [
            'frame',
            'p1L','p2L','p3L', # position left
            'R11L','R12L','R13L', # rotation left
            'R21L','R22L','R23L',
            'R31L','R32L','R33L',
            'p1R','p2R','p3R', # position right
            'R11R','R12R','R13R', # rotation right
            'R21R','R22R','R23R',
            'R31R','R32R','R33R'
        ])

    data.to_csv(coordinatesFile, index=False)
    


if __name__ == '__main__':
    for i in range(5):
        extractWristCoordinatesBVH(
            BVHFile = f'/Users/igavier/Documents/GitHub/VirtualIMU/data/captureOrig{i+1}.bvh',
            coordinatesFile = f'/Users/igavier/Documents/GitHub/VirtualIMU/data/captureOrigWristCoord{i+1}.csv'
        )