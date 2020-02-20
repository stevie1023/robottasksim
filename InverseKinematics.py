import numpy as np
import pybullet as p
import os
import math
import robosuite.utils.transform_utils as T

if __name__ == "__main__":

    fname = os.getcwd() + "/robosuite/models/assets/bullet_data/darias_description/robots/darias_no_hands.urdf"

    # physicsClient = p.connect(p.GUI)  # or p.DIRECT for non-graphical version
    physicsClient = p.connect(p.DIRECT)
    p.setGravity(0, 0, -9.81)

    # darias = p.loadURDF(fname)
    darias = p.loadURDF(fname, [0, 0, 0], [0, 0, 0, 1])  # set the basePosition and baseOrientation

    # set up inverse kinematics

    current_pos = p.getJointState(darias,)



