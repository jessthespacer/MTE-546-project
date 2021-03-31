import numpy as np  
from var_store import var_store
import math as m
from EKF_pred import EKF_pred 

class EKF_corr(EKF_pred):
	
	def __init__(self):
		super().__init__()
	
	#sensor data	
	def Yk(self):
		#Need to get from sensor data
		#column on sensor data
		#'Time	Steering angle	Throttle	forward speed	yaw rotation	acceleration'
		return np.random.rand(3,1)

	#sensor model
	def h(self,Xk):
		return Xk

	#State covariance
	def Pk_corr(self,Pk_m1,Xk,Uk):
		#Xk and Uk are actually predm1 and m1
		#written as Xk,Uk for simplicity
		val = np.matmul(self.Kk_corr(Pk_m1,Xk,Uk),self.Hk(Xk,Uk))
		identity = np.ones(val.shape)
		return np.matmul(identity - val,self.Pk(Pk_m1,Xk,Uk))

	#kalman matrix gain
	def Kk_corr(self,Pk_m1,Xk,Uk):
		#Xk and Uk are actually predm1 and m1
		#written as Xk,Uk for simplicity		
		val =  np.matmul(self.Pk(Pk_m1,Xk,Uk),np.transpose(self.Hk(Xk,Uk)))
		val_2 = np.matmul(np.matmul(self.Hk(Xk,Uk),self.Pk(Pk_m1,Xk,Uk)),np.transpose(self.Hk(Xk,Uk))) + self.R
		self.Kk_val = np.matmul(val,np.linalg.inv(val_2))
		return self.Kk_val

	#jacobian of h(x)
	def Hk(self,Xk,Uk):		
		return self.Fk(Xk,Uk)
		
	#main function
	def Xk_pred_corr(self,Xk_pred_corr_k_m1,Uk_m1,Pk_m1):
		assert Xk_pred_corr_k_m1.shape == (3,1), 'incorr Xk_pred_corr_k_m1 shape'
		assert Uk_m1.shape == (2,1), 'incorr Uk_m1 shape'
		val = self.Xk_pred(Xk_pred_corr_k_m1,Uk_m1)	
		val_2 = self.Yk() - self.h(Xk_pred_corr_k_m1)		
		ret_val =  val + np.matmul(self.Kk_corr(Pk_m1,Xk_pred_corr_k_m1,Uk_m1),val_2)
		assert ret_val.shape == (3,1), 'corrected pred is incorr shape'
		return ret_val

#plot matrix vlaues over time
def plot_mat():
	pass




def main_loop():
	
	import matplotlib.pyplot as plt
	state_hist = []
	Pk_hist = []
	K_hist = []
	Xk_pred_corr_k_m1 = np.random.rand(3,1)
	Pk_m1 = np.random.rand(3,3)
	for i in range(8):
		Uk_m1 = np.random.rand(2,1)
		Pk_m1 = EKF_corr.Pk_corr(Pk_m1,Xk_pred_corr_k_m1,Uk_m1)
		Kk = EKF_corr
		Xk_pred_corr_k_m1 = EKF_corr.Xk_pred_corr(Xk_pred_corr_k_m1,Uk_m1,Pk_m1)
		#append to history side
		state_hist.append(Xk_pred_corr_k_m1)
		Pk_hist.append(Pk_m1)
		K_hist.append(EKF_corr.Kk_val)

	#Plot state
	state_hist_1 = np.transpose(np.array(state_hist)[:,0,:]).flatten()
	state_hist_2 = np.transpose(np.array(state_hist)[:,1,:]).flatten()
	state_hist_3 = np.transpose(np.array(state_hist)[:,2,:]).flatten()	
	time = np.array([i for i in range(len(state_hist))])
	plt.plot(time,state_hist_1,label = 'v')
	plt.plot(time,state_hist_2,label = 'thetha')
	plt.plot(time,state_hist_3,label = 'w')
	plt.title('State History')
	plt.xlabel('time step')
	# Set the y axis label of the current axis.
	plt.ylabel('State Values')
	plt.legend()
	plt.show()


		
		

if __name__ == '__main__':
	EKF_corr = EKF_corr()
	main_loop()



#TODO->
'''
- CHECK EACH INDIVIDUAL METHOD
- HAVE ASSERTIONS FOR INPUT
- COMPLETE JACOBINAS
- FINSIH READING AND WRITING CODE FOR EKF DATA
- Have several iterations
- GENERATE SIMPLE PLOT DATA
'''