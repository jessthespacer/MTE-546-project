import numpy as np

class var_store:
	#values subject to change after receiving simulation data
	def __init__(self):
		#global state vars
		self.Q = np.array([[0.5, 0, 0], [0, 0.1, 0], [0, 0, 0.1]])
		self.R = np.array([[0.5, 0, 0], [0, 0.1, 0], [0, 0, 0.1]])
		self.del_k_m2 = 1
		#store_vals
		self.Pk_val = None
		self.Kk_val = None
###########################################################################################
		#loaded variables -> data being loaded in from csv(probably pandas)
		self.case_name = ""
		self.csv = ""
###########################################################################################
		#control and state vars
		#mostly methods
###########################################################################################
		#Model constants
		#Distance between rear wheel axis and COG [m]
		self.L_r = 0.75
		#Length between rear and front wheel axes [m]
		self.L = 1.5	
		
###########################################################################################










