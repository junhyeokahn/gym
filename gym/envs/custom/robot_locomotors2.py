import os
PROJECT_PATH = os.getcwd()

import pybullet
import pybullet_data
from pybullet_envs.robot_bases import URDFBasedRobot
from pybullet_envs.gym_locomotion_envs import WalkerBaseBulletEnv

import numpy as np

class WalkerBase2(URDFBasedRobot):
	def __init__(self,  fn, robot_name, action_dim, obs_dim, base_pos, base_ori, power):
		URDFBasedRobot.__init__(self, fn, robot_name, action_dim, obs_dim, base_pos, base_ori)
		self.power = power
		self.camera_x = 0
		self.start_pos_x, self.start_pos_y, self.start_pos_z = 0, 0, 0
		self.walk_target_x = 1e3  # kilometer away
		self.walk_target_y = 0
		self.body_xyz=[0,0,0]

	def robot_specific_reset(self, bullet_client):
		self._p = bullet_client
		for j in self.ordered_joints:
			j.reset_current_position(self.np_random.uniform(low=-0.1, high=0.1), 0)

		self.feet = [self.parts[f] for f in self.foot_list]
		self.feet_contact = np.array([0.0 for f in self.foot_list], dtype=np.float32)
		self.scene.actor_introduce(self)
		self.initial_z = None

	def apply_action(self, a):
		assert (np.isfinite(a).all())
		for n, j in enumerate(self.ordered_joints):
			j.set_motor_torque(self.power * j.power_coef * float(np.clip(a[n], -1, +1)))

	def calc_state(self):
		j = np.array([j.current_relative_position() for j in self.ordered_joints], dtype=np.float32).flatten()
		# even elements [0::2] position, scaled to -1..+1 between limits
		# odd elements  [1::2] angular speed, scaled to show -1..+1
		self.joint_speeds = j[1::2]
		self.joints_at_limit = np.count_nonzero(np.abs(j[0::2]) > 0.99)

		body_pose = self.robot_body.pose()
		parts_xyz = np.array([p.pose().xyz() for p in self.parts.values()]).flatten()
		self.body_xyz = (
		parts_xyz[0::3].mean(), parts_xyz[1::3].mean(), body_pose.xyz()[2])  # torso z is more informative than mean z
		self.body_rpy = body_pose.rpy()
		z = self.body_xyz[2]
		if self.initial_z == None:
			self.initial_z = z
		r, p, yaw = self.body_rpy
		self.walk_target_theta = np.arctan2(self.walk_target_y - self.body_xyz[1],
											self.walk_target_x - self.body_xyz[0])
		self.walk_target_dist = np.linalg.norm(
			[self.walk_target_y - self.body_xyz[1], self.walk_target_x - self.body_xyz[0]])
		angle_to_target = self.walk_target_theta - yaw

		rot_speed = np.array(
			[[np.cos(-yaw), -np.sin(-yaw), 0],
			 [np.sin(-yaw), np.cos(-yaw), 0],
			 [		0,			 0, 1]]
		)
		vx, vy, vz = np.dot(rot_speed, self.robot_body.speed())  # rotate speed back to body point of view

		more = np.array([ z-self.initial_z,
			np.sin(angle_to_target), np.cos(angle_to_target),
			0.3* vx , 0.3* vy , 0.3* vz ,  # 0.3 is just scaling typical speed into -1..+1, no physical sense here
			r, p], dtype=np.float32)
		return np.clip( np.concatenate([more] + [j] + [self.feet_contact]), -5, +5)

class Draco(WalkerBase2):
	self_collision = False
	foot_list = ["lAnkle", "rAnkle"]

	def __init__(self):

		base_pos = [0, 0, 1.1]
		base_ori = [0, 0, 0, 1]
		WalkerBase2.__init__(self,  PROJECT_PATH+"/RobotModel/Robot/Draco/DracoFixed.urdf", 'Torso', action_dim=10, obs_dim=30, base_pos=base_pos, base_ori=base_ori, power=1)

	def robot_specific_reset(self, bullet_client):
		WalkerBase2.robot_specific_reset(self, bullet_client)
		self.motor_names = ["lHipYaw", "lHipRoll", "lHipPitch", "lKnee", "lAnkle"]
		self.motor_power  = [10, 10, 150, 150, 1.]
		self.motor_names = ["rHipYaw", "rHipRoll", "rHipPitch", "rKnee", "rAnkle"]
		self.motor_power  = [10, 10, 150, 150, 1.]

		alpha = -np.pi / 5.
		beta = np.pi / 4.
		self.jdict['lHipPitch'].reset_current_position(alpha, 0)
		self.jdict['rHipPitch'].reset_current_position(alpha, 0)
		self.jdict['lKnee'].reset_current_position(beta-alpha, 0)
		self.jdict['rKnee'].reset_current_position(beta-alpha, 0)
		self.jdict['lAnkle'].reset_current_position(np.pi/2. - beta, 0)
		self.jdict['rAnkle'].reset_current_position(np.pi/2. - beta, 0)

		self.motors = [self.jdict[n] for n in self.motor_names]
		if self.random_yaw:
			position = [0,0,0]
			orientation = [0,0,0]
			yaw = self.np_random.uniform(low=-3.14, high=3.14)
			if self.random_lean and self.np_random.randint(2)==0:
				cpose.set_xyz(0, 0, 1.4)
				if self.np_random.randint(2)==0:
					pitch = np.pi/2
					position = [0, 0, 0.45]
				else:
					pitch = np.pi*3/2
					position = [0, 0, 0.25]
				roll = 0
				orientation = [roll, pitch, yaw]
			else:
				position = [0, 0, 1.4]
				orientation = [0, 0, yaw]  # just face random direction, but stay straight otherwise
			self.robot_body.reset_position(position)
			self.robot_body.reset_orientation(orientation)
		self.initial_z = 0.8

	random_yaw = False
	random_lean = False

	def apply_action(self, a):
		assert( np.isfinite(a).all() )
		force_gain = 1
		for i, m, power in zip(range(17), self.motors, self.motor_power):
			m.set_motor_torque(float(force_gain * power * self.power * np.clip(a[i], -1, +1)))

	def alive_bonus(self, z, pitch):
		return +2 if z > 0.85 else -1   # 2 here because 17 joints produce a lot of electricity cost just from policy noise, living must be better than dying

class DracoEnv2(WalkerBaseBulletEnv):
	def __init__(self, robot=Draco(), render=False):
		self.robot = robot
		WalkerBaseBulletEnv.__init__(self, self.robot, render)
		self.electricity_cost  = 4.25*WalkerBaseBulletEnv.electricity_cost
		self.stall_torque_cost = 4.25*WalkerBaseBulletEnv.stall_torque_cost
