import numpy as np
import pybullet as p
import os
import math
from robosuite.models.base import MujocoXML
from mujoco_py import MjSim, MjRenderContextOffscreen
from robosuite.utils import SimulationError, XMLError, MujocoPyRenderer
from robosuite.models.arenas.table_arena import TableArena
import os
import numpy as np
import mujoco_py
import time
from mujoco_py import load_model_from_xml, MjSim, functions, mjrenderpool, MjSimState
from scipy.interpolate import CubicSpline


if __name__ == "__main__":

    fname = os.getcwd() + "/robosuite/models/assets/bullet_data/darias_description/robots/darias_no_hands.urdf"


    # physicsClient = p.connect(p.GUI)  # or p.DIRECT for non-graphical version
    physicsClient = p.connect(p.DIRECT)
    p.setGravity(0, 0, -9.81)

    darias = p.loadURDF(fname)
   # darias = p.loadURDF(fname, [0, 0, 0], [0, 0, 0, 1])  # set the basePosition and baseOrientation
    #darias = p.loadMJCF(fname2)

    joint_num = p.getNumJoints(darias)

    print("the number of joint:", joint_num)
    # print("the name of joints:", joint_name[1])

    # get the all index and joints of darias:
    for i in range(joint_num):
        joint_name = p.getJointInfo(darias, i)
        print(joint_name[0], joint_name[1])

     # lower limits for null space
    ll = [-2.967059, -2.094395, -2.967059, -2.094395, -2.967059, -2.094395, -2.967059,
          -2.967059, -2.094395, -2.967059, -2.094395, -2.967059, -2.094395, -2.094395]
    # upper limits for null space
    ul = [2.967059, 2.094395, 2.967059, 2.094395, 2.967059, 2.094395, 2.967059,
          2.967059, 2.094395, 2.967059, 2.094395, 2.967059, 2.094395, 2.094395]
    # joint ranges for null space
    jr = [5.8, 4, 5.8, 4, 5.8, 4, 6]
    # restposes for null space
    # rp = [0, 0, 0, 0.5 * math.pi, 0, -math.pi * 0.5 * 0.66, 0]
    # joint damping coefficents
    jd = np.eye(7) * 0.1

    # for i in range(7):
    #     p.resetJointState(darias, i + 2, rp[i])
    #     p.resetJointState(darias, i + 10, rp[i])

    p.setRealTimeSimulation(1)




    def goal_pos_joints():
        """
        input: the desired pos of end-effector

        :return: the desired pos of each joints
        """
        # set the desired pos and orientations of end-effector
        t = 0.05  # we can change time here

        # set x to a fixed value (in local frame or world frame)
        pos = [-0.4, 0.2 * math.cos(t), 0. + 0.2 * math.sin(t)]
        # end effector points down, not up
        orn = p.getQuaternionFromEuler([0, -math.pi, 0])

        # the index of {right,left} end-effectors
        effector_right = 8
        effector_left = 16


        # in local frame
        # current joint position
        initial_pos=np.zeros(7)

        # this part is not sure
        jointPoses1 = list(p.calculateInverseKinematics(darias, effector_right, pos, orn, ll[0:7], ul[0:7],
                                                        jr, initial_pos))

        print(jointPoses1)
        print(np.shape(jointPoses1))

        jointPoses2 = list(p.calculateInverseKinematics(darias, effector_left, pos, orn, ll[7:14], ul[7:14],
                                                        jr, initial_pos))



        print(jointPoses2)
        print(np.shape(jointPoses2))

        for i in range(7, 14):
            jointPoses1[i] = jointPoses2[i]

        print("joint desired position:", jointPoses1)  # desired joint position of two arms
        print("the number of joint:", len(jointPoses1))
        print("jointPoses1[1:]:", jointPoses1[1:])
        print(np.shape(jointPoses1[1:]))

        return jointPoses1



   # test= goal_pos_joints()





    eef_pos_in_world = np.array(p.getLinkState(darias, 8)[0])
    eef_orn_in_world = np.array(p.getLinkState(darias, 8)[1])

    eef_pos_in_local = np.array(p.getLinkState(darias, 8)[2])
    eef_orn_in_local = np.array(p.getLinkState(darias, 8)[3])

    linkframe_pos_in_world = np.array(p.getLinkState(darias, 8)[4])
    linkframe_orn_in_world = np.array(p.getLinkState(darias, 8)[5])


    print("right end-effector pos in world:", eef_pos_in_world)
    print("right orn in world:", eef_orn_in_world)
    print("right end-effector pos in local:", eef_pos_in_local)
    print("right orn in local:", eef_orn_in_local)
    print("link frame position in world:", linkframe_orn_in_world)
    print("link frame orientation in world:", linkframe_pos_in_world)


    joint_lower_limit=[]
    joint_upper_limit=[]

    joint_Max_Force=[]
    joint_Max_Velocity=[]
    joint_Damping=[]
    joint_Friction=[]

    for i in range(joint_num):
        if(i<=1 or i==9):
            continue
        else:
            info = p.getJointInfo(darias, i)
            joint_lower_limit.append(info[8])
            joint_upper_limit.append(info[9])
            joint_Max_Force.append(info[10])
            joint_Max_Velocity.append((info[11]))
            joint_Damping.append(info[6])
            joint_Friction.append(info[7])


    # print("joint lower limit:", joint_lower_limit)
    # print("joint upper limit:", joint_upper_limit)


    joint_current_pos=[p.getJointState(darias,x)[0] for x in range(joint_num)]
    print("joint current pos in pybullet:", joint_current_pos)

    print("joint damping:", joint_Damping)
    print("joint max force:", joint_Max_Force)
    print("joint max velocity:", joint_Max_Velocity)
    print("joint friction:", joint_Friction)


#############---------------mujoco version---------------------


    fname2 = os.getcwd() + "/robosuite/models/assets/robots/darias/darias_nohands.xml"

    mujoco_darias = MujocoXML(fname2)

    model = mujoco_darias
    mjpy_model = model.get_model(mode="mujoco_py")
    sim = MjSim(mjpy_model)


    joint_pos_mj=[sim.data.qpos[x] for x in range(14)]

    id_name = "right_endeffector_link"
    # id_name="R_SAA"
    current_position = []
    current_position = sim.data.body_xpos[sim.model.body_name2id(id_name)]
    print("right_endeffector_link position:", current_position)
    current_orientation_mat = sim.data.body_xmat[sim.model.body_name2id(id_name)].reshape([3, 3])
    print("right_endeffector_link orientation", current_position)

    print("joint current pos in mujoco:", joint_pos_mj)






