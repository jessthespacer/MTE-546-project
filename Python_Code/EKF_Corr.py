import numpy as np  
from var_store import var_store
import math as m
from EKF_pred import EKF_pred 

class EKF_corr(EKF_pred):
	
	def __init__(self):
		super().__init__()
	
		
	def Yk(self):
		#Need to get from sensor data
		pass

	def h(self,val):
		#is this the sensor model?
		identity = np.ones(val.size)
		return val

	def Pk_corr(self):
		val = np.matmul(self.Kk_corr(),self.Hk())
		identity = np.ones(np.size(val))
		return np.matmul((identity - val),self.Pk(Pk_m1))


	def Kk_corr(self,Pk_m1):
		val =  np.matmul(self.Pk(Pk_m1),np.transpose(self.Hk()))
		val_2 = np.matmul(np.matmul(self.Hk(),self.Pk(Pk_m1)),np.transpose(self.Hk())) + self.R
		return np.matmul(val,np.linalg.inv(val_2))

	def Hk(self):
		#should be just Fk
		#3x3 matrix
		#should be close to identity		
		return np.ones((3,3))
		
	#main function
	def Xk_pred_corr(self,Xk_pred_corr_k_m1,Uk_m1,Pk_m1):
		val = self.Xk_pred(Xk_pred_corr_k_m1,Uk_m1)	
		val_2 = self.h(self.Xk) - self.h(val)
		print(self.h(self.Xk))
		return val + np.matmul(self.Kk_corr(Pk_m1),val_2)


if __name__ == '__main__':
	EKF_corr = EKF_corr()
	#2 random matrixes of certain size
	Xk_pred_corr_k_m1 = np.random.rand(3,1)
	Uk_m1 = np.random.rand(2,1)
	Pk_m1 = np.random.rand(3,3)
	EKF_corr.Xk_pred_corr(Xk_pred_corr_k_m1,Uk_m1,Pk_m1)




'''
April 4th -> come up with airfcraft family set.
airfoil selection and wing design

April 4th -> Also come up with airfoil selction and wing design:
April 8th-> engine selection and placement
April 18th -> finish peromance analysis & sketch of final design

'''