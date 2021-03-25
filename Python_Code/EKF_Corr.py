import numpy as np  
from var_store import var_store
import math as m

class EKF_Corr(EKF_Pred):
	
	def __init__(self):
		super().__init__()
		
		
	def Yk(self):
		#Need to get from sensor data

	def h(self):
		#is this the sensor model?
		identity = np.ones(Xk.size)
		return np.matmul(identity,self.Xk)


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
	#main function
	def Xk_pred_corr(self):
		val = self.Xk_pred(Xk_pred_corr_k_m1,Uk_m1)
		val_2 = self.h(self.Xk) - self.h(self.Xk_pred(Xk_pred_corr_k_m1,Uk_m1))
		return val + np.matmul(self.Kf_corr(),val_2)

	
