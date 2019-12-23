import numpy as np
# import robosuite as suite
from robosuite import Env_SawyerRmp

from robosuite.models.base import MujocoXML
from robosuite.environments import MujocoEnv

from mujoco_py import MjSim, MjRenderContextOffscreen

from robosuite.utils import SimulationError, XMLError, MujocoPyRenderer

import os
import xml.dom.minidom
import xml.etree.ElementTree as ET
import io
import numpy as np

from robosuite.utils import XMLError

import mujoco_py
import pybullet as p

if __name__ == "__main__":
    fname = "/home/ren/mypy/RobotTaskSim/robosuite/models/assets/robots/darias/darias.xml"

    # tree = ET.parse(fname)
    # root = tree.getroot()
    #
    # for child in root:
    #     print(child.tag, " -- ", child.attrib)

    # fname="/home/ren/mypy/RobotTaskSim/robosuite/models/assets/robots/baxter/robot.xml"

    mujoco_darias = MujocoXML(fname)

    model = mujoco_darias


    mjpy_model = model.get_model(mode="mujoco_py")
    sim = MjSim(mjpy_model)

    viewer = MujocoPyRenderer(sim)

    for i in range(5000):
        viewer.render()

    # model = mujoco_darias.get_model()

    # model = mujoco_py.load_model_from_xml(fname)

    # sim = mujoco_py.MjSim(model)

    # physicsClient = p.connect(p.GUI)  # or p.DIRECT for non-graphical version
    # p.setGravity(0, 0, -9.81)
    #
    # darias = p.loadURDF(fname)
    # joint_num = p.getNumJoints(darias)
    #
    # p.disconnect()
    #
    # print("joint_num", joint_num)
