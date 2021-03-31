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
		self.del_k_m1 = 2
		self.del_k_m2 = 1
###########################################################################################
		#loaded variables -> data being loaded in from csv(probably pandas)
		self.case_name = ""
		self.csv = ""
		#value returned from rows based on counter value
		self.counter = 0
###########################################################################################
		#control and state vars
		#mostly methods
###########################################################################################
		#Model constants
		#Distance between rear wheel axis and COG [m]
		self.L_r = 1
		#Length between rear and front wheel axes [m]
		self.L = 1
		
		
###########################################################################################
class data_process(var_store):
	def __init__(self):
		super().__init__()


	def get_data(self):
		return None

	def process_data(self):
		return None

	def save_clean_data(self):
		return None

	def analyze_data_Main(self):
		#main loop
		pass

	def visualized_analyzed(self):
		pass

	def save_analyzed(self):
		pass










