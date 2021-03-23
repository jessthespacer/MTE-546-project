import numpy as np


class var_store:

	def __init__(self):

		#global state vars

		self.state_rows = 3
		self.state_cols = 1

		#ekf variables
		#T: Front wheel drive torque [N-m]		
		self.Tk_m1 = ""
		#r: Wheel radius [m]
		self.r = ""
		self.mu_i = ""
		self.m = ""
		self.g = 9.81
		self.Lr = 1
		self.L = 1
		self.delta_t = ""
		self.Q = ""
		






