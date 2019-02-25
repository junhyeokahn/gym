import logging
import math
import gym
from gym import spaces
from gym.utils import seeding
import numpy as np
import time
import subprocess
import pybullet as p
import pybullet_data
from pkg_resources import parse_version
import os
import pybullet_data

PROJECT_PATH = os.getcwd()

logger = logging.getLogger(__name__)

class DracoEnv(gym.Env):
  metadata = {
    'render.modes': ['human', 'rgb_array'],
    'video.frames_per_second' : 50
  }

  def __init__(self, renders=False):
    # ==========================================================================
    # Renderer
    # ==========================================================================
    self._renders = renders
    if (renders):
        p.connect(p.GUI)
    else:
        p.connect(p.DIRECT)

    # ==========================================================================
    # Robot Configuration
    # ==========================================================================
    self.hanging_pos = [0, 0, 0.893]
    self.draco = p.loadURDF(
            PROJECT_PATH+"/RobotModel/Robot/Draco/DracoFixed.urdf",
            self.hanging_pos, useFixedBase=False)
    p.setAdditionalSearchPath(pybullet_data.getDataPath())
    p.loadURDF("plane.urdf")
    self.feet = ['lFootFront', 'lFootFront2', 'lFootBack', 'lFootBack2',
                 'rFootFront', 'rFootFront2', 'rFootBack', 'rFootBack2']
    self.b_contact = [False]*len(self.feet)
    self.contact_force = [0]*len(self.feet)

    self.joint_list = {}
    self.link_list = {}
    self.dof_idx = []
    dof_lb = []
    dof_ub = []
    dof_max_force = []
    active_joint = 0
    for j in range (p.getNumJoints(self.draco)):
        info = p.getJointInfo(self.draco, j)
        joint_name = info[1]
        joint_type = info[2]
        link_name = info[12]
        self.joint_list[joint_name.decode()] = j
        self.link_list[link_name.decode()] = j
        if (joint_type==p.JOINT_PRISMATIC or joint_type==p.JOINT_REVOLUTE):
            self.dof_idx.append(j)
            dof_lb.append(info[8])
            dof_ub.append(info[9])
            dof_max_force.append(info[10])
            active_joint+=1
    self.n_dof = active_joint

    # ==========================================================================
    # Observation & Action
    # ==========================================================================
    self.observation_space = spaces.Box(np.array(dof_lb), np.array(dof_ub),
                                        dtype=np.float32)
    self.action_space = spaces.Box(np.array([-1]*self.n_dof),
                                   np.array([1]*self.n_dof), dtype=np.float32)
    self.action_sclae = np.array(dof_max_force)

    self.seed()
    self.viewer = None
    self._configure()

  def _configure(self, display=None):
    self.display = display

  def seed(self, seed=None):
    self.np_random, seed = seeding.np_random(seed)
    return [seed]

  def step(self, action):
    # ==========================================================================
    # Apply action
    # ==========================================================================
    base_pos, base_quat, _, _, _, _ = self.get_observation()
    pos_before = base_pos[0]
    clamped_action = np.clip(np.array(action), self.action_space.low,
                                               self.action_space.high)
    force = clamped_action * self.action_sclae

    p.setJointMotorControlArray(self.draco, self.dof_idx, p.TORQUE_CONTROL,
                                forces=force)
    p.stepSimulation()

    # ==========================================================================
    # Get observation
    # ==========================================================================
    base_pos, base_quat, q, base_vel, base_so3, qdot = self.get_observation()
    pos_after = base_pos[0]
    base_euler = p.getEulerFromQuaternion(base_quat)
    obs = base_pos + base_quat + q + base_vel + base_so3 + qdot

    # ==========================================================================
    # Termination condition
    # ==========================================================================
    done = bool( (np.abs(base_pos[2] - self.base_ini_pos[2]) > 0.05) or
                 (np.abs(base_pos[1] - self.base_ini_pos[1]) > 0.10) or
                 (np.abs(base_euler[0]) > 0.1) or
                 (np.abs(base_euler[1]) > 0.1) or
                 (np.abs(base_euler[2]) > 0.1)
               )

    # ==========================================================================
    # Reward
    # ==========================================================================
    vel_rew = (pos_after - pos_before) / self.timeStep
    alive_bonus = 10.0
    action_pen = 0.5 * np.square(clamped_action).sum()
    deviation_pen = 3 * np.square(base_pos[1:3]).sum()
    for foot_idx, foot_name in enumerate(self.feet):
        contact_result = p.getContactPoints(bodyA=self.draco,
                                            linkIndexA=self.link_list[foot_name])
        if len(contact_result) == 0:
            self.b_contact[foot_idx] = False
            self.contact_force[foot_idx] = 0
        else:
            self.b_contact[foot_idx] = True
            self.contact_force[foot_idx] = \
                sum([np.sqrt(cp[9]*cp[9] + cp[10]*cp[10] + cp[12]*cp[12]) \
                    for cp in contact_result])

    impact_pen = 1.0 * sum(self.contact_force) / 380.0
    impact_pen = min(impact_pen, 3)
    reward = vel_rew + alive_bonus - action_pen - deviation_pen - impact_pen
    if done:
        reward = 0.0

    # print("alive_bonus : {}, action_pen : {}, deviation_pen : {}, impact_pen : {}, total_rew : {}".format(alive_bonus, action_pen, deviation_pen, impact_pen, reward))
    # return np.array(obs), reward, False, {}
    return np.array(obs), reward, done, {}

  def reset(self):
    p.resetSimulation()
    p.setAdditionalSearchPath(pybullet_data.getDataPath())
    p.loadURDF("plane.urdf")
    self.draco = p.loadURDF(
            PROJECT_PATH+"/RobotModel/Robot/Draco/DracoFixed.urdf",
            self.hanging_pos, useFixedBase=False)
    p.changeDynamics(self.draco, -1, linearDamping=0, angularDamping=0)
    self.timeStep = 0.01
    p.setJointMotorControlArray(self.draco, self.dof_idx, p.TORQUE_CONTROL,
                                forces=np.zeros(self.n_dof))
    p.setGravity(0,0, -9.81)
    p.setTimeStep(self.timeStep)
    p.setRealTimeSimulation(0)

    alpha = -np.pi/4.0
    beta = np.pi/5.5
    self.ini_pos = np.array([0, 0, alpha, beta - alpha, np.pi/2 - beta,
                             0, 0, alpha, beta - alpha, np.pi/2 - beta])
    randpos = self.np_random.uniform(low=-0.05, high=0.05,
                                       size=(self.n_dof,))
    randvel = self.np_random.uniform(low=-0.05, high=0.05,
                                       size=(self.n_dof,))
    for i, dof_idx in enumerate(self.dof_idx):
        p.resetJointState(self.draco, dof_idx, self.ini_pos[i] + randpos[i],
                          randvel[i])
    base_pos, base_quat, q, base_vel, base_so3, qdot = self.get_observation()
    obs = base_pos + base_quat + q + base_vel + base_so3 + qdot
    self.base_ini_pos = base_pos

    return np.array(obs)

  def render(self, mode='human', close=False):
      return

  # ============================================================================
  # Joint State Query
  # ============================================================================
  def get_observation(self):
    joint_states = p.getJointStates(self.draco, self.dof_idx)
    q = [state[0] for state in joint_states]
    qdot = [state[1] for state in joint_states]
    base_pos, base_quat, base_vel, base_so3 = (), (), (), ()
    base_pos, base_quat= p.getBasePositionAndOrientation(self.draco)
    base_vel, base_so3 = p.getBaseVelocity(self.draco)

    return list(base_pos), list(base_quat), q, \
           list(base_vel), list(base_so3), qdot


  # ============================================================================
  # Link State Query
  # ============================================================================
  def get_link_com_pos(self, link_idx):
    return p.getLinkState(self.draco, link_idx,
                          computeLinkVelocity=False,
                          computeForwardKinematics=True)[0]

  def get_link_com_quat(self, link_idx):
    return p.getLinkState(self.draco, link_idx,
                          computeLinkVelocity=False,
                          computeForwardKinematics=True)[1]

  def get_link_local_com_pos(self, link_idx):
    return p.getLinkState(self.draco, link_idx,
                          computeLinkVelocity=False,
                          computeForwardKinematics=True)[2]

  def get_link_local_com_quat(self, link_idx):
    return p.getLinkState(self.draco, link_idx,
                          computeLinkVelocity=False,
                          computeForwardKinematics=True)[3]

  def get_link_pos(self, link_idx):
    return p.getLinkState(self.draco, link_idx,
                          computeLinkVelocity=False,
                          computeForwardKinematics=True)[4]

  def get_link_quat(self, link_idx):
    return p.getLinkState(self.draco, link_idx,
                          computeLinkVelocity=False,
                          computeForwardKinematics=True)[5]

  def get_link_vel(self, link_idx):
    return p.getLinkState(self.draco, link_idx,
                          computeLinkVelocity=True,
                          computeForwardKinematics=True)[6]

  def get_link_so3(self, link_idx):
    return p.getLinkState(self.draco, link_idx,
                          computeLinkVelocity=True,
                          computeForwardKinematics=True)[7]
