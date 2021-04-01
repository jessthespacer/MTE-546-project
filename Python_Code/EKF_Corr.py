import warnings
warnings.simplefilter(action='ignore', category=Warning)
import pandas as pd
import numpy as np  
from var_store import var_store
import math as m
from EKF_pred import EKF_pred 
import matplotlib.pyplot as plt

class EKF_corr(EKF_pred):
	
	def __init__(self):
		super().__init__()
	
	#sensor data	
	def Yk(self,val):
		#need to return sensor data
		return val

	#sensor model
	def h(self,Xk):
		return Xk

	#State covariance
	def Pk_corr(self,Pk_m1,Xk,Uk):
		#Xk and Uk are actually predm1 and m1
		#written as Xk,Uk for simplicity
		identity = np.eye(3)
		self.Pk_corr_val = (identity - (self.Kk_corr(Pk_m1,Xk,Uk)\
			@self.Hk(Xk,Uk)))@self.Pk(Pk_m1,Xk,Uk)
		return self.Pk_corr_val

	#kalman matrix gain
	def Kk_corr(self,Pk_m1,Xk,Uk):
		#Xk and Uk are actually predm1 and m1
		#written as Xk,Uk for simplicity		
		self.Kk_corr_val = self.Pk(Pk_m1,Xk,Uk)@np.transpose(self.Hk(Xk,Uk))\
			@np.linalg.inv(((self.Hk(Xk,Uk)@self.Pk(Pk_m1,Xk,Uk))\
				@np.transpose(self.Hk(Xk,Uk)))+ self.R)
		return self.Kk_corr_val

	#jacobian of h(x)
	def Hk(self,Xk,Uk):		
		return self.Fk(Xk,Uk)

##################	#########	#########	#########	#########	#########	#########			
def read_csv(fil_path):
	#Time	Steering angle	Throttle	forward speed	yaw angle	acceleration	yaw angular rotation
	df = pd.read_csv(fil_path)
	columns = df.columns
	df_Uk = df[['Throttle','Steering angle']]
	df_Yk = df[['forward speed','yaw angle','yaw angular rotation']]
	#mulptiply rotation anngle and convert to radians
	df_Uk['Steering angle'] = df_Uk['Steering angle']*70*m.pi/180
	df_Yk[['yaw angle','yaw angular rotation']] = df_Yk[['yaw angle','yaw angular rotation']]*m.pi/180
	Uk = df_Uk.to_numpy()
	Yk = df_Yk.to_numpy()
	return Uk,Yk




def plot_state(state_hist,Yk):
	#Plot state
	state_hist_1 = np.transpose(np.array(state_hist)[:,0,:]).flatten()
	state_hist_2 = np.transpose(np.array(state_hist)[:,1,:]).flatten()
	state_hist_3 = np.transpose(np.array(state_hist)[:,2,:]).flatten()	
	Yk_hist_1 = Yk[:,0]
	Yk_hist_2 = Yk[:,1]
	Yk_hist_3 = Yk[:,2]

	time = np.array([i for i in range(len(state_hist))])
	print(len(time),len(state_hist_1))
	plt.plot(time,state_hist_1,label = 'v')
	plt.plot(time,state_hist_2,label = 'theta')
	plt.plot(time,state_hist_3,label = 'w')
	print(len(time),len(Yk_hist_1))
	
	plt.plot(time,Yk_hist_1,label = 'v groundtruth')
	plt.plot(time,Yk_hist_2,label = 'theta groundtruth')
	plt.plot(time,Yk_hist_3,label = 'w groundtruth')
	plt.title('State History')
	plt.xlabel('time step')
	# Set the y axis label of the current axis.
	plt.ylabel('State Values')
	plt.legend()
	plt.show()
	


def main_loop(Yk,Uk):
	state_hist = []
	Pk_hist = []
	K_hist = []
	Xk_pred_corr_k_m1 = np.array([Yk[0]]).transpose()
	state_hist.append(Xk_pred_corr_k_m1)
	Pk_m1 = np.random.rand(3,3)
	for i in range(1,len(Uk)):
		#set carla data
		EKF_corr.dt()
		Uk_m1 = np.transpose(np.array([Uk[i-1]]))
		#EKF_corr.Yk()		
		#make predicitions
		Xk_pred = EKF_corr.Xk_pred(Xk_pred_corr_k_m1,Uk_m1)
		Yk_pred = EKF_corr.h(Xk_pred_corr_k_m1)
		Pk = EKF_corr.Pk(Pk_m1,Xk_pred_corr_k_m1 ,Uk_m1)
		Hk = EKF_corr.Hk(Xk_pred_corr_k_m1,Uk_m1)
		#correct with sensor data		
		Yk_curr_iter = np.array([Yk[i-1]]).transpose()
		K = EKF_corr.Kk_corr(Pk,Xk_pred_corr_k_m1,Uk_m1)
		Xk_pred_corr = Xk_pred + K@(Yk_curr_iter - Yk_pred) 
		Pk = (np.eye(3) - K@Hk)@Pk
		#set curr values as previous
		Xk_pred_corr_k_m1 = Xk_pred_corr
		Pk_m1 = Pk
		#print(Xk_pred_corr_k_m1)
		state_hist.append(Xk_pred_corr_k_m1)

	return state_hist

	
	






		
		

if __name__ == '__main__':
	EKF_corr = EKF_corr()
	path = r"C:\Users\shawn paul\Desktop\MTE-546-project\cases\MVPcasefinalwithnoise.csv"
	Uk,Yk = read_csv(path)
	EKF_state_hist = main_loop(Yk,Uk)
	#get true states
	path = r"C:\Users\shawn paul\Desktop\MTE-546-project\cases\MVPcasefinal.csv"
	Uk_real,Yk_real = read_csv(path)
	plot_state(EKF_state_hist,Yk_real)
	
	
	



