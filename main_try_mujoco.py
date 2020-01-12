<<<<<<< HEAD
import numpy as np


import robosuite as suite
from robosuite import DariasClothGrasp

=======
>>>>>>> 1fab2027a49bdef4e8e6e2e07cf4f5e0b162894c
from robosuite.models.base import MujocoXML
from mujoco_py import MjSim, MjRenderContextOffscreen
from robosuite.utils import SimulationError, XMLError, MujocoPyRenderer
<<<<<<< HEAD

from robosuite.models.tasks import ClothGrasp, UniformRandomSampler

from robosuite.models.arenas.table_arena import TableArena
from robosuite.models.objects import BoxObject
from collections import OrderedDict
=======
from robosuite.models.arenas.table_arena import TableArena
>>>>>>> 1fab2027a49bdef4e8e6e2e07cf4f5e0b162894c
import os
import numpy as np
<<<<<<< HEAD

from robosuite.utils import XMLError

from robosuite.models.robots import Darias

from robosuite.environments.base import MujocoEnv

import mujoco_py
import pybullet as p
import time
if __name__ == "__main__":
    fname = "/home/mingyezhu/Desktop/robottasksim/robosuite/models/assets/robots/darias/darias.xml"

=======
import mujoco_py


if __name__ == "__main__":

    fname = os.getcwd() + "/robosuite/models/assets/robots/darias/darias.xml"
>>>>>>> 1fab2027a49bdef4e8e6e2e07cf4f5e0b162894c

    mujoco_darias = MujocoXML(fname)

    model = mujoco_darias

    table_full_size = (0.8, 0.8, 0.8)
    table_friction = (1., 5e-3, 1e-4)
    mujoco_arena = TableArena(
        table_full_size=table_full_size,
        table_friction=table_friction
    )

    # The sawyer robot has a pedestal, we want to align it with the table
    # mujoco_arena.set_origin([0.16 + table_full_size[0] / 2, 0, 0])

<<<<<<< HEAD
    filename_cloth = "/home/mingyezhu/Desktop/robottasksim/robosuite/models/assets/objects/cloth.xml"

    filename_supporter = "/home/mingyezhu/Desktop/robottasksim/robosuite/models/assets/objects/supporter.xml"
    mujoco_cloth = MujocoXML(filename_cloth)

    mujoco_supporter = MujocoXML(filename_supporter)
    filename_table = "/home/mingyezhu/Desktop/robottasksim/robosuite/models/assets/arenas/table_arena.xml"
    mujoco_table = MujocoXML(filename_table)
    # model.merge(mujoco_arena)
    model.merge(mujoco_cloth)
    model.merge(mujoco_table)
    model.merge(mujoco_supporter)

    env = suite.DariasClothGrasp
    #
    # env = suite.make(
    #     DariasClothGrasp,
    # )

=======
    filename_cloth = os.getcwd() + "/robosuite/models/assets/objects/cloth.xml"

    filename_supporter = os.getcwd() + "/robosuite/models/assets/objects/supporter.xml"
    mujoco_cloth = MujocoXML(filename_cloth)

    mujoco_supporter = MujocoXML(filename_supporter)
    filename_table = os.getcwd() + "/robosuite/models/assets/arenas/table_arena.xml"
    mujoco_table = MujocoXML(filename_table)

    # model.merge(mujoco_cloth)
    model.merge(mujoco_table)
    # model.merge(mujoco_supporter)
>>>>>>> 1fab2027a49bdef4e8e6e2e07cf4f5e0b162894c

    mjpy_model = model.get_model(mode="mujoco_py")
    sim = MjSim(mjpy_model)

<<<<<<< HEAD

    viewer = MujocoPyRenderer(sim)


    for i in range(10000):
=======
    print('number of contacts', sim.data.ncon)

    for i in range(sim.data.ncon):
        contact = sim.data.contact[i]
        print('contact', i)
        print('dist', contact.dist)
        print('geom1', contact.geom1, sim.model.geom_id2name(contact.geom1))
        print('geom2', contact.geom2, sim.model.geom_id2name(contact.geom2))
        # There's more stuff in the data structure
        # See the mujoco documentation for more info!
        geom2_body = sim.model.geom_bodyid[sim.data.contact[i].geom2]
        print(' Contact force on geom2 body', sim.data.cfrc_ext[geom2_body])
        print('norm', np.sqrt(np.sum(np.square(sim.data.cfrc_ext[geom2_body]))))
        # Use internal functions to read out mj_contactForce
        c_array = np.zeros(6, dtype=np.float64)
        print('c_array', c_array)
        mujoco_py.functions.mj_contactForce(sim.model, sim.data, i, c_array)
        print('c_array', c_array)

    viewer = MujocoPyRenderer(sim)

    for i in range(50000):
>>>>>>> 1fab2027a49bdef4e8e6e2e07cf4f5e0b162894c
        sim.step()

        viewer.render()

<<<<<<< HEAD
    a = sim.data.sensordata
    print(a)

    print('number of contacts', sim.data.ncon)

    for i in range(sim.data.ncon):
        contact = sim.data.contact[i]
        print('contact', i)
        print('dist', contact.dist)
        print('geom1', contact.geom1, sim.model.geom_id2name(contact.geom1))
        print('geom2', contact.geom2, sim.model.geom_id2name(contact.geom2))
        # There's more stuff in the data structure
        # See the mujoco documentation for more info!
        geom2_body = sim.model.geom_bodyid[sim.data.contact[i].geom2]
        print(' Contact force on geom2 body', sim.data.cfrc_ext[geom2_body])
        print('norm', np.sqrt(np.sum(np.square(sim.data.cfrc_ext[geom2_body]))))
        # Use internal functions to read out mj_contactForce
        c_array = np.zeros(6, dtype=np.float64)
        print('c_array', c_array)
        mujoco_py.functions.mj_contactForce(sim.model, sim.data, i, c_array)
        print('c_array', c_array)
    darias = p.loadURDF(fname)
    joint_num = p.getNumJoints(darias)

    p.disconnect()

    print("joint_num", joint_num)
=======
    # darias = p.loadURDF(fname)
    # joint_num = p.getNumJoints(darias)
    #
    # p.disconnect()
    #
    # print("joint_num", joint_num)
>>>>>>> 1fab2027a49bdef4e8e6e2e07cf4f5e0b162894c
