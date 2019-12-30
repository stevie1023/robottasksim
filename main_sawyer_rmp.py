import numpy as np
# import robosuite as suite
from robosuite import Env_SawyerRmp
from robosuite.utils.transform_utils import get_orientation_error, quat2mat, mat2quat

from rmp.rmp_base import RMP_Root, RMP_Node, RMP_robotLink
from rmp.rmp_leaf import CollisionAvoidanceDecentralized, GoalAttractorUni

if __name__ == "__main__":

    # initialize the task
    env = Env_SawyerRmp(has_renderer=True,
                        ignore_done=True,
                        use_camera_obs=False,
                        control_freq=100, )
    env.reset()
    env.viewer.set_camera(camera_id=0)

    joint_index = 6

    psi = lambda q: env.f_psi(joint_index, q)
    J = lambda q: env.f_jcb(joint_index, q)
    dJ = lambda q, dq: env.f_jcb_dot(joint_index, q, dq)

    rt = RMP_Root('root')

    link_frames = []

    for i in range(env.num_joint):
        link_frame = RMP_robotLink('rlk_' + str(i), i, rt, env.f_psi, env.f_jcb, env.f_jcb_dot)
        link_frames.append(link_frame)

    # goal control points
    # gcp_o=RMP_Node('gcp_0', link_frames[6], )
    #
    # goal_attractors = []

    # for i in range(N):
    #     goal_attractor = GoalAttractorUni('ga_robot_' + str(i),
    #                                       link_frames[i],
    #                                       x_g[i],
    #                                       alpha=1,
    #                                       gain=1,
    #                                       eta=2)
    #
    #     gas.append(goal_attractor)

    # do visualization
    for i in range(5000):
        action = np.random.randn(env.dof)
        action = [0.05, 0.05, 0.05, 0.05, 0.05, 0.05, 0.05, 0.0]

        obs, reward, done, _ = env.step(action)

        di = env.get_obv_for_planning()

        q = di["joint_pos"]
        dq = di["joint_vel"]

        id_name = 'right_l' + str(joint_index)
        current_position = env.sim.data.body_xpos[env.sim.model.body_name2id(id_name)]

        current_quat = env.sim.data.body_xquat[
            env.sim.model.body_name2id(id_name)]  # quaternion (w, x, y, z)
        current_rotmat = env.sim.data.body_xmat[env.sim.model.body_name2id(id_name)].reshape([3, 3])

        current_velp = env.sim.data.body_xvelp[env.sim.model.body_name2id(id_name)]
        current_velr = env.sim.data.body_xvelr[env.sim.model.body_name2id(id_name)]

        Jx = env.sim.data.get_body_jacp(id_name).reshape((3, -1))
        Jx = np.delete(Jx, [1, 8, 9], axis=1)
        Jr = env.sim.data.get_body_jacr(id_name).reshape((3, -1))
        Jr = np.delete(Jr, [1, 8, 9], axis=1)

        if i % 10 == 0:
            print()
            print()

            # print(psi(q))
            # print(J(q))
            # print(dJ(q, dq))

            # print('Jacobian_pos------------------------------')
            # print(current_velp)
            # print(np.dot(Jx, dq))
            # print(np.dot(J(q)[0], dq))
            # print('Jacobian_ori------------------------------')
            print(current_velr)
            quat_dot = np.dot(J(q), dq)[3:7]
            quat = psi(q)[3:7]

            print()

            # print('ForwardKinematics_pos------------------------------')
            # print(current_position)
            # print(link_frames[joint_index].psi(q)[0])
            # print(psi(q)[1])
            # print('ForwardKinematics_ori------------------------------')
            # print(mat2quat(current_rotmat))
            # print(psi(q)[2])

            print()
            print()

        env.render()
