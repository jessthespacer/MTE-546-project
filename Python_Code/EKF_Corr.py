import numpy as np  
from var_store import var_store
import math as m
from EKF_pred import EKF_pred 

class EKF_corr(EKF_pred):
	
	def __init__(self):
		super().__init__()
	
	#!sensor data	
	def Yk(self):
		#Need to get from sensor data
		#column on sensor data
		#'Time	Steering angle	Throttle	forward speed	yaw rotation	acceleration'
		return np.random.rand(3,1)

	#!sensor model
	def h(self,Xk):
		return Xk

	#!State covariance
	def Pk_corr(self,Pk_m1,Xk,Uk):
		#Xk and Uk are actually predm1 and m1
		#written as Xk,Uk for simplicity
		val = np.matmul(self.Kk_corr(Pk_m1,Xk,Uk),self.Hk(Xk))
		identity = np.ones(np.size(val))
		return np.matmul((identity - val),self.Pk(Pk_m1))

	#!kalman matrix gain
	def Kk_corr(self,Pk_m1,Xk,Uk):
		#Xk and Uk are actually predm1 and m1
		#written as Xk,Uk for simplicity		
		val =  np.matmul(self.Pk(Pk_m1,Xk,Uk),np.transpose(self.Hk(Xk)))
		val_2 = np.matmul(np.matmul(self.Hk(Xk),self.Pk(Pk_m1,Xk,Uk)),np.transpose(self.Hk(Xk))) + self.R
		return np.matmul(val,np.linalg.inv(val_2))

	#!jacobian of h(x)
	def Hk(self,Xk):		
		return np.random.rand(3,3)
		
	#!main function
	def Xk_pred_corr(self,Xk_pred_corr_k_m1,Uk_m1,Pk_m1):
		assert Xk_pred_corr_k_m1.shape == (3,1), 'incorr Xk_pred_corr_k_m1 shape'
		assert Uk_m1.shape == (2,1), 'incorr Uk_m1 shape'
		val = self.Xk_pred(Xk_pred_corr_k_m1,Uk_m1)	
		val_2 = self.Yk() - self.h(Xk_pred_corr_k_m1)		
		ret_val =  val + np.matmul(self.Kk_corr(Pk_m1,Xk_pred_corr_k_m1,Uk_m1),val_2)
		assert ret_val.shape == (3,1), 'corrected pred is incorr shape'
		return ret_val


if __name__ == '__main__':
	EKF_corr = EKF_corr()
	#2 random matrixes of certain size
	Xk_pred_corr_k_m1 = np.random.rand(3,1)
	Uk_m1 = np.random.rand(2,1)
	Pk_m1 = np.random.rand(3,3)
	v = EKF_corr.Xk_pred_corr(Xk_pred_corr_k_m1,Uk_m1,Pk_m1)
	print(v)



#TODO->
'''
- CHECK EACH INDIVIDUAL METHOD
- HAVE ASSERTIONS FOR INPUT
- COMPLETE JACOBINAS
- FINSIH READING AND WRITING CODE FOR EKF DATA
- Have several iterations
- GENERATE SIMPLE PLOT DATA
'''