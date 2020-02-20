# here we use pybullt to calculate the inverse kinematics
import pybullet as p
import time
import math
from DateTime import DateTime
import random
import numpy as np
import scipy
from scipy.interpolate import interp1d

import os
from robosuite.models.base import MujocoXML
from mujoco_py import MjSim, MjRenderContextOffscreen
from robosuite.utils import SimulationError, XMLError, MujocoPyRenderer
from robosuite.models.arenas.table_arena import TableArena
import mujoco_py

from mujoco_py import load_model_from_xml, MjSim, functions, mjrenderpool, MjSimState
from scipy.interpolate import CubicSpline
import matplotlib.pyplot as plt

if __name__ == "__main__":


    # activate pybullet here

    fname = os.getcwd() + "/robosuite/models/assets/bullet_data/darias_description/robots/darias_no_hands.urdf"


    # initial parameters:
    ramp_ratio = 0.20  # Percentage of the time between policy time-steps used for interpolation
    control_freq = 20 # control steps per second

    # lower limits for null space
    ll = [-2.967059, -2.094395, -2.967059, -2.094395, -2.967059, -2.094395, -2.967059,
          -2.967059, -2.094395, -2.967059, -2.094395, -2.967059, -2.094395, -2.094395]
    # upper limits for null space
    ul = [2.967059, 2.094395, 2.967059, 2.094395, 2.967059, 2.094395, 2.967059,
          2.967059, 2.094395, 2.967059, 2.094395, 2.967059, 2.094395, 2.094395]
    # joint ranges for null space
    jr = [5.8, 4, 5.8, 4, 5.8, 4, 6]
    # initial joint position
    po = np.zeros(7)
    # joint damping coefficents
    jd = [0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1]


    # activate mujoco here
    fname2 = os.getcwd()+"/robosuite/models/assets/robots/darias/darias_nohands.xml"

    mujoco_darias = MujocoXML(fname2)

    model = mujoco_darias
    mjpy_model = model.get_model(mode="mujoco_py")
    sim = MjSim(mjpy_model)



    # Gravity compensation
    def Gravity():
        for i in range(14):
            sim.data.qfrc_applied[i] = sim.data.qfrc_bias[i]


    def goal_pos_joints():
        """
        input: the desired pos of end-effector

        :return: the desired pos of each joints

        """

        p.connect(p.DIRECT) # physicsClient = p.connect(p.GUI)

        p.setGravity(0, 0, -9.81)
        darias = p.loadURDF(fname, [0, 0, 0], [0, 0, 0, 1])  # set the basePosition and baseOrientation



        #joint_num = p.getNumJoints(darias)

        # 0: disable real-time simulation, the external forces=0 after each step
        # 1: to enable

        p.setRealTimeSimulation(1)

        # print("the number of joint in pybullet:", joint_num)

        #
        # # get the all index and joints of darias:
        # for i in range(joint_num):
        #     joint_name = p.getJointInfo(darias, i)
        #     print(joint_name[0], joint_name[1])

        # set the desired pos and orientations of end-effector



        # set x to a fixed value
        pos_right= [0.4, -0.2, 0.2]
        pos_left = [0.4, 0.2, 0.2]
        # end effector points down, not up
        orn_right = p.getQuaternionFromEuler([0, -math.pi, 0])
        orn_left = p.getQuaternionFromEuler([0, math.pi, 0])

        # the index of {right,left} end-effectors
        effector_right = 8
        effector_left = 16

        # this part is not sure
        jointPoses1 = list(p.calculateInverseKinematics(darias, effector_right, pos_right, orn_right, ll[0:7], ul[0:7],
                                                        jr, po))

        jointPoses2 = list(p.calculateInverseKinematics(darias, effector_left, pos_left, orn_left, ll[7:14], ul[7:14],
                                                        jr, po))

        for i in range(7, 14):
            jointPoses1[i] = jointPoses2[i]

        return np.array(jointPoses1)


    def interpolate_joint(starting_joint, last_goal_joint, interpolation_steps, current_vel, interpolation):
        # we interpolate to reach the commanded desired position in the ramp_ratio % of time we have this goal
        # joint position is a (interpolation_steps, 14) matrix

        if interpolation == "cubic":
            time = [0, interpolation_steps]
            position = np.vstack((starting_joint, last_goal_joint))
            cubic_joint = CubicSpline(time, position, bc_type=((1, current_vel), (1, np.zeros(14))), axis=0)
            interpolation_joint = np.array([cubic_joint(i) for i in range(interpolation_steps)])

        if interpolation == "linear":
            delta_x_per_step = (last_goal_joint - starting_joint) / interpolation_steps
            interpolation_joint = np.array(
                [starting_joint + i * delta_x_per_step for i in range(1, interpolation_steps + 1)])

        return interpolation_joint




    # set Kv and Kp
    def kp_cartisian(timestep, dimension):
        """
        input: timestep : how many steps in a command
        kp: 3-dim parameters, a cubic line connected with 5 points
        kv: kv=2*sqrt(kp)*damping
        damping: in range [0,1], here we set 0.1 for end-effector
        :return: kp, kv
        """
        damping = 0.1

        kp_cubic = []

        for dim in range(dimension):
            kp_dim = []
            random.seed(dim)
            for i in range(5):
                # here we choose the kp in range [25, 300]
                kp_dim.append(random.uniform(25, 300))

            # we sort kp from the largest to the smallest
            kp_dim.sort(reverse=True)

            x = np.linspace(0, timestep, num=5, endpoint=True)
            y = kp_dim
            f = interp1d(x, y, kind="cubic")

            # num 取值不确定
            xnew = np.linspace(0, timestep, num=timestep, endpoint=True)
            kp_cubic = np.hstack((kp_cubic, f(xnew)))
            kp_cubic = np.array(kp_cubic.reshape(dimension, timestep))

        # here we set the value of kp, which is smaller than 0, to 0
        for i in range(kp_cubic.shape[0]):
            for j in range(kp_cubic.shape[1]):
                if kp_cubic[i, j] < 0:
                    kp_cubic[i, j] = 0

        kv_cubic = 2 * np.sqrt(kp_cubic) * damping

        return kp_cubic, kv_cubic


    def kp_kv_joint():
        """

        :return: kp list of each joints
        """
        kp = np.ones(14) * 800

        kv = 2* np.sqrt(kp)


        return kp, kv

    def test_kp_joint(dimension=14, timesteps=2000):

        kp_joint, kv_joint = kp_cartisian(timesteps, dimension)

        return kp_joint, kv_joint


    def update_model_joint():
        """
        Updates the state of the robot used to compute the control command
        :param sim:
        :param joint_index:
        :param id_name:
        :return:
        """

        # calculate the position and velocity of each joint
        current_joint_position = [sim.data.qpos[x] for x in range(14)]
        current_joint_velocity = [sim.data.qvel[x] for x in range(14)]


        return np.array(current_joint_position), np.array(current_joint_velocity)


    def update_model_eeffector():
        # calculate the position, orientation, linear-velocity and angle_velocity of the end-effector

        body_name_right = "right_endeffector_link"
        body_name_left = "left_endeffector_link"

        current_position_right = sim.data.body_xpos[sim.model.body_name2id(body_name_right)]
        current_orientation_mat_right = sim.data.body_xmat[sim.model.body_name2id(body_name_right)].reshape([3, 3])
        current_lin_velocity_right = sim.data.body_xvelp[sim.model.body_name2id(body_name_right)]
        current_ang_velocity_right = sim.data.body_xvelr[sim.model.body_name2id(body_name_right)]
        list_right = [current_position_right, current_orientation_mat_right, current_lin_velocity_right, current_ang_velocity_right]

        current_position_left = sim.data.body_xpos[sim.model.body_name2id(body_name_left)]
        current_orientation_mat_left = sim.data.body_xmat[sim.model.body_name2id(body_name_left)].reshape([3, 3])
        current_lin_velocity_left = sim.data.body_xvelp[sim.model.body_name2id(body_name_left)]
        current_ang_velocity_left = sim.data.body_xvelr[sim.model.body_name2id(body_name_left)]
        list_left = [current_position_left, current_orientation_mat_left, current_lin_velocity_left, current_ang_velocity_left]

        return list_right, list_left




    def update_joint_pybullet():
        """

        :return: the position of each joints
        """

        current_joint_position = [p.getJointState(darias, x)[0] for x in range(joint_num)][0:14]
        current_joint_velocity = [p.getJointState(darias, x)[1] for x in range(joint_num)][0:14]


        return current_joint_position

    def update_eeffector_pybullet():
        """

        :return: the position of left and right end-effector
        """

        right_position = p.getLinkState(darias, 8)[0] # link world position
        left_position = p.getLinkState(darias, 16)[0]

        return right_position, left_position


    def update_massmatrix():

        mass_matrix = np.ndarray(shape=(len(sim.data.qvel) ** 2,), dtype=np.float64, order='C')
        mujoco_py.cymj._mj_fullM(sim.model, mass_matrix, sim.data.qM)
        mass_matrix = np.reshape(mass_matrix, (len(sim.data.qvel), len(sim.data.qvel)))
        # matrix_right= mass_matrix[joint_index, :][:, joint_index]
        # print("mass_matrix:", mass_matrix)
        # print("matrix of right", matrix_right)

        return mass_matrix


    def calculate_torques(current_joint_position, current_vel, desired_joint_position, kp_joint, kv_joint):


        desired_joint_velocity = np.zeros(14)
        position_joint_error = - current_joint_position + desired_joint_position
        velocity_joint_error = -current_vel + desired_joint_velocity

        torques = np.multiply(kp_joint, position_joint_error) + np.multiply(kv_joint, velocity_joint_error)

        # rescale normalized torques to control range
        ctrl_range = sim.model.actuator_ctrlrange

        bias = 0.5 * (ctrl_range[:, 1] + ctrl_range[:, 0])
        weight = 0.5 * (ctrl_range[:, 1] - ctrl_range[:, 0])

        applied_action = bias + weight * torques



        return np.array(torques), np.array(applied_action)


    def command_controll(current_pos, current_vel, desired_pos_joint, steps):
        pos = []
        errors = []
        print("desired pos joint:", desired_pos_joint)

        for i in range(steps):
            Gravity()

            torques, applied_torques = calculate_torques(current_pos, current_vel, desired_pos_joint, kp_joint, kv_joint)
            # print("torques:", torques[0])
            # print("applied torques:", applied_torques[0])
            #
            # #x = np.ones(14) * 100
            # #sim.data.ctrl[:] = x
            # a= torques[0]+torques[1]
            # torques[0]=0
            # torques[1]=a

            sim.data.ctrl[:] = torques
            #sim.data.ctrl[:] =
            sim.step()

           # sim.data.ctrl[:] = 3

            # print("ctrl:", sim.data.ctrl)


            current_pos, current_vel = update_model_joint()
            print("the pos of 1st joints:", current_pos[0])
            print("the pos of 8st joints:", current_pos[7])


            error = current_pos - desired_pos_joint
            pos.append(current_pos)
            errors.append(error)

            viewer.render()

        list_right, list_left = update_model_eeffector()
        eef_right_pos = list_right[0]
        eef_left_pos = list_left[0]
        pos = np.array(pos)
        errors = np.array(errors)

        right_end = current_pos[6]
        left_end = current_pos[13]


        return current_pos, current_vel


    def draw_picture(joint_index, steps, joint_pos, desired_pos_joint):

        x = np.linspace(0, steps, num=steps, endpoint=True)
        y = joint_pos[:, joint_index]
        y_des = np.ones((steps,)) * desired_pos_joint[joint_index]
        #z = errors[:, n]
        # print(y.shape)
        # print(y_des.shape)

        plt.plot(x, y, 'o', x, y_des, '--')
        plt.savefig("position_of_joint_%d.png" %joint_index)


    #----------------begin the mujoco--------------------#

    viewer = MujocoPyRenderer(sim)
   # set kp, kv to a fixed value:
    kp_joint, kv_joint = kp_kv_joint()

    # initialized the position and velocity of joints and end-effector:
    initial_pos = np.array([sim.data.qpos[x] for x in range(14)])
    initial_vel = np.array([sim.data.qvel[x] for x in range(14)])

    body_name_right = "right_endeffector_link"
    body_name_left = "left_endeffector_link"

    eff_pos_right_initial = sim.data.body_xpos[sim.model.body_name2id(body_name_right)]
    eff_pos_left_initial = sim.data.body_xpos[sim.model.body_name2id(body_name_left)]

    # calculate the desired position of joints using interpolation:

    last_goal_joint = np.zeros(14)
    # right-arm
    last_goal_joint[0] = 1
    last_goal_joint[3] = -1
    last_goal_joint[6]= -1

    #left-arm
    last_goal_joint[7] = -1
    last_goal_joint[10] = 1
    last_goal_joint[13] = 1

    #last_goal_joint = goal_pos_joints()

    interpolation_steps = 50
    #interpolation = "linear"
    interpolation = "cubic"
    desired_joint_pos = interpolate_joint(initial_pos, last_goal_joint, interpolation_steps,
                                          current_vel=initial_vel, interpolation = interpolation)


    # begin the command:
    current_pos = initial_pos
    current_vel = initial_vel
    for i in range(interpolation_steps):
        current_pos, current_vel = command_controll(current_pos, current_vel, desired_joint_pos[i], steps=500)



    pos_right = [0.4, -0.2, 0.2]
    pos_left = [0.4, 0.2, 0.2]
    print("desired pos of right-end:", pos_right)
    print("desired pos of left-end:", pos_left)

    list_right, list_left = update_model_eeffector()
    eef_right_pos = list_right[0]
    eef_left_pos = list_left[0]
    print("the pos of right-end  :", eef_right_pos)
    print("the pos of left-end:", eef_left_pos)

    print("current joint position:", current_pos)
    print("desired joint position:", last_goal_joint)
    print("the errors of joint position:", current_pos-last_goal_joint)
    print("current joint velcocity:", current_vel)






    #----------- in Pybullet version---------

    # right_end_pos, left_end_pos = update_eeffector_pybullet()
    # joint_pos_pybullet = update_joint_pybullet()

    # print("pos", pos.shape)
    # print("the pos of right-end:", eef_right_pos)
    # print("the pos of left-end:", eef_left_pos)
    # print("the pos of right-joint:", right_end )
    # print("the pos of left-joint:", left_end)
    # print("the pos of right in world of pybullet:", right_end_pos)
    # print("the pos of left in world of pybullet:", left_end_pos)
    # print("the pos of joints in puybullet:", joint_pos_pybullet)
    #
    # joint_indexs = np.arange(14)
    # ctrl_range = sim.model.actuator_ctrlrange
    # bias = 0.5 * (ctrl_range[:, 1] + ctrl_range[:, 0])
    # weight = 0.5 * (ctrl_range[:, 1] - ctrl_range[:, 0])
    # action = np.ones(14)
    # applied_action = bias + weight * action
    # print(applied_action)
    #
    # print("the pos before action:", update_model_joint()[0])
    #
    # for i in range(5000):
    #     Gravity()
    #
    #    # torques = calculate_torques(current_pos, current_vel, desired_pos_joint, kp_joint, kv_joint)
    #     # print("the torque of 1  right:", torques[0])
    #     # print("the torque of 1 left:", torques[7])
    #     # x = np.ones(14) * 100
    #     # sim.data.ctrl[:] = x
    #     sim.data.ctrl[joint_indexs] = applied_action
    #
    #     # print("ctrl:", sim.data.ctrl)
    #     sim.step()
    #     viewer.render()
    #
    # current_pos, current_vel = update_model_joint()
    # print("the pos of all joints after actions:", current_pos)


















