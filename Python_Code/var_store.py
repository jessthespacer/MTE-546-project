import numpy as np

class var_store:
	#values subject to change after receiving simulation data

	def __init__(self):

		#global state vars
		self.state_rows = 3
		self.state_cols = 1
		#ekf variables
		#T: Front wheel drive torque [N-m]		
		self.Tk_m1 = 2
		self.Q = np.ones((3,3))
		self.R = np.ones((3,3))
		self.Xk = np.random.rand(3,1)
###########################################################################################
		#loaded variables -> data being loaded in from csv(probably pandas)
		self.csv = ""
###########################################################################################
		#control and state variables -> designated as methods or vars otherwise below
		#rolling friction force
		self.f_f = 5	
		self.f_r = 5
###########################################################################################
		#Model constants
		#!Values subject to change
		#r: Wheel radius [m]
		self.r = 0.5
		#Axle coefficient of friction for wheel 𝑖; 𝜇𝑓 for front wheel, 𝜇𝑟 for rear wheel
		self.mu_i = 0.5
		#Vehicle mass [kg]
		self.m = 10
		#gravitational force
		self.g = 9.81
		#Distance between rear wheel axis and COG [m]
		self.L_r = 1
		#Length between rear and front wheel axes [m]
		self.L = 1
		#Frontal vehicle area [m2]
		self.A = 5
		#difference in time
		self.dt = 0.05
		#Coefficient of drag
		self.C_d = 0.81
###########################################################################################


'''
store all global variables - y 
create and run jacobian matrixes
verify all matrixes and run again
insert assertion for EKF_pred -> use main helper function to run all
have a way of iterating over time//dealing with continued vars
get kaps data and run with ekf
'''








