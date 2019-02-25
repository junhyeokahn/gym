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
    self.base_pos_ini = [0, 0, 1.193]
    self.draco = p.loadURDF(
            PROJECT_PATH+"/RobotModel/Robot/Draco/DracoFixed.urdf",
            self.base_pos_ini, useFixedBase=False)
    p.setAdditionalSearchPath(pybullet_data.getDataPath())
    p.loadURDF("plane.urdf")

    self.joint_list = {}
    self.dof_idx = []
    dof_lb = []
    dof_ub = []
    dof_max_force = []
    active_joint = 0
    for j in range (p.getNumJoints(self.draco)):
        info = p.getJointInfo(self.draco, j)
        joint_name = info[1]
        joint_type = info[2]
        self.joint_list[joint_name.decode()] = j
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
    done = bool( (np.abs(base_pos[2] - self.base_pos_ini[2]) > 0.05) or
                 (np.abs(base_pos[1] - self.base_pos_ini[1]) > 0.10) or
                 (np.abs(base_euler[0]) > 0.1) or
                 (np.abs(base_euler[1]) > 0.1) or
                 (np.abs(base_euler[2]) > 0.1)
               )

    # ==========================================================================
    # Reward
    # ==========================================================================
    vel_rew = (pos_after - pos_before) / self.timeStep
    alive_bonus = 2.0
    action_pen = 0.5 * np.abs(clamped_action).sum()
    deviation_pen = 3 * np.abs(base_pos[1])
    reward = vel_rew + alive_bonus - action_pen - deviation_pen
    if done:
        reward = 0.0
    return np.array(obs), reward, done, {}
    # return np.array(obs), reward, False, {}

  def reset(self):
    p.resetSimulation()
    p.setAdditionalSearchPath(pybullet_data.getDataPath())
    p.loadURDF("plane.urdf")
    self.draco = p.loadURDF(
            PROJECT_PATH+"/RobotModel/Robot/Draco/DracoFixed.urdf",
            self.base_pos_ini, useFixedBase=False)
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
    return np.array(obs)

  def render(self, mode='human', close=False):
      return

  def get_observation(self):
    joint_states = p.getJointStates(self.draco, self.dof_idx)
    q = [state[0] for state in joint_states]
    qdot = [state[1] for state in joint_states]
    base_pos, base_quat, base_vel, base_so3 = (), (), (), ()
    base_pos, base_quat= p.getBasePositionAndOrientation(self.draco)
    base_vel, base_so3 = p.getBaseVelocity(self.draco)

    return list(base_pos), list(base_quat), q, \
           list(base_vel), list(base_so3), qdot
