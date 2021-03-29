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
		self.case_name = ""
		self.csv = ""
###########################################################################################
		#control and state vars
		#mostly methods
###########################################################################################
		#Model constants
		#!Values subject to change
		#r: Wheel radius [m]
		#Distance between rear wheel axis and COG [m]
		self.L_r = 1
		#Length between rear and front wheel axes [m]
		self.L = 1
		#difference in time
		self.dt = 0.05
		
###########################################################################################
	#get data from csv
	def get_data():
		return None


'''
store all global variables - y 
create and run jacobian matrixes
verify all matrixes and run again
insert assertion for EKF_pred -> use main helper function to run all
have a way of iterating over time//dealing with continued vars
get kaps data and run with ekf
'''








