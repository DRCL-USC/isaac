import argparse
import numpy as np
import pinocchio as pin
import trimesh  # If mesh loading is needed

def load_robot(robot_type):
    if robot_type == 'g1':
        robot = pin.RobotWrapper.BuildFromURDF(
            'resources/robots/g1_description/g1_29dof_rev_1_0.urdf',
            'resources/robots/g1_description',
            pin.JointModelFreeFlyer()  # Free-flyer joint (7 degrees of freedom)
        )
        return robot
    else:
        raise ValueError("Unknown robot type: " + robot_type)

def get_dof_names():
    # Define the joint degrees of freedom (DOF) names according to the dataset (total 36)
    dof_names = [
       "root_x", "root_y", "root_z", "root_qx", "root_qy", "root_qz", "root_qw",
       "left_hip_pitch", "left_hip_roll", "left_hip_yaw", "left_knee", "left_ankle_pitch", "left_ankle_roll",
       "right_hip_pitch", "right_hip_roll", "right_hip_yaw", "right_knee", "right_ankle_pitch", "right_ankle_roll",
       "waist_yaw", "waist_roll", "waist_pitch",
       "left_shoulder_pitch", "left_shoulder_roll", "left_shoulder_yaw", "left_elbow",
       "left_wrist_roll", "left_wrist_pitch", "left_wrist_yaw",
       "right_shoulder_pitch", "right_shoulder_roll", "right_shoulder_yaw", "right_elbow",
       "right_wrist_roll", "right_wrist_pitch", "right_wrist_yaw"
    ]
    return dof_names

def get_body_names(robot):
    # Use the names of the visual objects in the robot model (removing special characters at the end, such as ":0" or "_0") as body_names
    body_names = []
    for visual in robot.visual_model.geometryObjects:
        # Reference previous code to strip the last two characters from the name
        name = visual.name[:-2]
        if name not in body_names:
            body_names.append(name)
    return body_names

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--file_name', type=str, help="CSV file name (without extension)", default='dance1_subject1')
    parser.add_argument('--robot_type', type=str, help="Robot type", default='g1')
    args = parser.parse_args()

    # Set frame rate in fps (30 frames per second)
    fps = np.int64(30)
    dt = 1.0 / fps  # Time interval per frame

    # 1. Load the robot model
    robot = load_robot(args.robot_type)

    # 2. Load CSV data to obtain joint configurations (dof_positions), with shape (N, 36)
    csv_file = 'humanoid/envs/custom/motions/' + args.robot_type + '/' + args.file_name + '.csv'
    dof_positions = np.genfromtxt(csv_file, delimiter=',').astype(np.float32)
    N, D = dof_positions.shape
    print("Loaded {} frames of data, each with {} degrees of freedom".format(N, D))

    # 3. Get the list of joint names (dof_names)
    dof_names = get_dof_names()
    print("dof_names:", dof_names)

    # 4. Compute joint velocities (dof_velocities) using finite differences
    dof_velocities = np.zeros_like(dof_positions)
    # Use forward/backward difference for the first/last frame, and central difference for internal frames
    dof_velocities[0] = (dof_positions[1] - dof_positions[0]) / dt
    dof_velocities[-1] = (dof_positions[-1] - dof_positions[-2]) / dt
    for i in range(1, N - 1):
        dof_velocities[i] = (dof_positions[i + 1] - dof_positions[i - 1]) / (2 * dt)

    # 5. Get the list of body names
    body_names = get_body_names(robot)
    B = len(body_names)
    print("body_names:", body_names)

    # 6. Allocate position and rotation data for each body in each frame
    body_positions = np.zeros((N, B, 3), dtype=np.float32)      # (N, B, 3)
    body_rotations = np.zeros((N, B, 4), dtype=np.float32)      # (N, B, 4) Quaternion order: [w, x, y, z]
    # Also store rotation matrices for computing angular velocity
    body_rot_matrices = np.zeros((N, B, 3, 3), dtype=np.float32)

    # For each frame configuration, compute forward kinematics (framesForwardKinematics) and extract body poses
    for i in range(N):
        q = dof_positions[i]
        pin.framesForwardKinematics(robot.model, robot.data, q)
        for j, name in enumerate(body_names):
            frame_id = robot.model.getFrameId(name)
            oMf = robot.data.oMf[frame_id]
            body_positions[i, j, :] = oMf.translation
            body_rot_matrices[i, j, :, :] = oMf.rotation
            # Convert rotation matrix to quaternion (Pinocchio's Quaternion default order is [x, y, z, w])
            quat = pin.Quaternion(oMf.rotation)
            # Convert to [w, x, y, z]
            body_rotations[i, j, :] = np.array([quat.w, quat.x, quat.y, quat.z])

    # 7. Compute linear velocity for each body using finite differences (body_linear_velocities)
    body_linear_velocities = np.zeros_like(body_positions)
    body_linear_velocities[0] = (body_positions[1] - body_positions[0]) / dt
    body_linear_velocities[-1] = (body_positions[-1] - body_positions[-2]) / dt
    for i in range(1, N - 1):
        body_linear_velocities[i] = (body_positions[i + 1] - body_positions[i - 1]) / (2 * dt)

    # 8. Compute angular velocity for each body using finite differences (body_angular_velocities)
    # Use rotation matrix differences: for consecutive frames i and i+1,
    # compute relative rotation matrix R_delta = R_i^T * R_{i+1},
    # then use pin.log3(R_delta) to get the rotation vector (axis-angle representation),
    # and divide by dt to obtain angular velocity (unit: rad/s)
    body_angular_velocities = np.zeros((N, B, 3), dtype=np.float32)
    for i in range(N - 1):
        for j in range(B):
            R1 = body_rot_matrices[i, j]
            R2 = body_rot_matrices[i + 1, j]
            R_delta = R1.T @ R2
            ang_vel = pin.log3(R_delta) / dt
            body_angular_velocities[i, j] = ang_vel
    # For the last frame, choose to assign the same angular velocity as the second-to-last frame or set it to zero
    body_angular_velocities[-1] = body_angular_velocities[-2]

    # Print the shapes of each array for verification
    print("dof_positions shape:", dof_positions.shape)
    print("dof_velocities shape:", dof_velocities.shape)
    print("body_positions shape:", body_positions.shape)
    print("body_rotations shape:", body_rotations.shape)
    print("body_linear_velocities shape:", body_linear_velocities.shape)
    print("body_angular_velocities shape:", body_angular_velocities.shape)
    print("fps:", fps)

    # 9. Save all data to an .npz file
    np.savez("skeleton_data.npz",
             fps=fps,
             dof_names=np.array(dof_names),
             body_names=np.array(body_names),
             dof_positions=dof_positions,
             dof_velocities=dof_velocities,
             body_positions=body_positions,
             body_rotations=body_rotations,
             body_linear_velocities=body_linear_velocities,
             body_angular_velocities=body_angular_velocities)

if __name__ == "__main__":
    main()
