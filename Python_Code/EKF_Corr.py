import numpy as np  
from var_store import var_store
import math as m
from EKF_pred import EKF_pred 

class EKF_corr(EKF_pred):
	
	def __init__(self):
		super().__init__()
	
	#!sensor data	
	def Yk(self):
		#set this at the start of each iteration
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

def main_loop():
	E_C = EKF_corr()
	#random vars just for start
	Xk_pred_corr_k_m1 = np.random.rand(3,1)
	print(Xk_pred_corr_k_m1)
	Pk_m1 = np.random.rand(3,3)
	#print(Xk_pred_corr_k_m1)
	#SET INITIAL VARS
	for i in range(9):
		#below variable will come from simulation controller	
		Uk_m1 = np.random.rand(2,1)
		print("Uk_m1")
		print(Uk_m1)
		Xk_pred_corr = E_C.Xk_pred_corr(Xk_pred_corr_k_m1,Uk_m1,Pk_m1)
		Xk_pred_corr_k_m1 = Xk_pred_corr
		Pk_m1 = E_C.Pk_val
		print("Xk_pred_corr_k_m1")
		print(Xk_pred_corr_k_m1)


if __name__ == '__main__':
	#seeing how the operations happen
	main_loop()
	
