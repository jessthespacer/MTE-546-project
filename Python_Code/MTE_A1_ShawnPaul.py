import numpy as np  
from var_store import var_store
import math as m

class EKF_Corr(EKF_Pred):
	
	def __init__(self):
		super().__init__()
		
		
	def Yk(self):
		pass
		#I don't think we need this?

	def h(self):
		#is this the sensor model?
		pass


	def Pk_corr(self):
		val = np.matmul(self.Kk_corr(),self.Hk())
		identity = np.ones(np.size(val))
		return np.matmul((identity - val),self.Pk(Pk_m1))


	def Kk_corr(self):
		val =  np.matmul(self.Pk(Pk_m1),np.transpose(self.Hk()))
		val_2 = np.matmul(np.matmul(self.Hk(),self.Pk()),np.transpose(self.Hk())) + self.R
		return np.matmul(val,np.inv(val_2))

	def Hk(self):
		#3x3 matrix
		#should be close to identity

		pass

	def Xk_pred_corr(self):
		pass


	
