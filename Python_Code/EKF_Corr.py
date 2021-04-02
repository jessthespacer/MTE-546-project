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

def compute_errors(predictions,targets):
	return np.sqrt((predictions-targets)**2)

def plot_state(state_hist,Yk):
	#Plot state
	state_hist_1 = np.transpose(np.array(state_hist)[:,0,:]).flatten()
	state_hist_2 = np.transpose(np.array(state_hist)[:,1,:]).flatten()
	state_hist_3 = np.transpose(np.array(state_hist)[:,2,:]).flatten()	
	Yk_hist_1 = Yk[:,0]
	Yk_hist_2 = Yk[:,1]
	Yk_hist_3 = Yk[:,2]

	time = np.array([i for i in range(len(state_hist))])
	rmse_v = compute_errors(state_hist_1,Yk_hist_1)
	rmse_theta = compute_errors(state_hist_2,Yk_hist_2)
	rmse_w = compute_errors(state_hist_3,Yk_hist_3)

	var_list = ['v','theta','w']
	for val in var_list:
		if val == 'v':
			plt.plot(time,state_hist_1,label = 'V EKF')
			plt.plot(time,Yk_hist_1,label = 'V Groundtruth')
			
			#get error change on seperate plot

		elif val == 'theta':
			plt.plot(time,state_hist_2,label = 'Theta EKF')
			plt.plot(time,Yk_hist_2,label = 'Theta Groundtruth')

			#get error change on seperate plot
		else:
			plt.plot(time,state_hist_3,label = 'W')
			plt.plot(time,Yk_hist_3,label = 'W Groundtruth')
			#get error change on seperate plot

		plt.title('State History')
		plt.xlabel('Time')
		# Set the y axis label of the current axis.
		plt.ylabel('State Values Over Time')
		plt.legend()
		#plt.show()
		fn = './plots/' + val + '_' + 'GTvsEKF'
		plt.savefig(fn)
		plt.clf()

	#get error change on
	for val in var_list:
		if val == 'v':
			print(len(time),len(rmse_v))
			plt.plot(time,rmse_v,label = 'RMSE V', color="blue")
		elif val == 'theta':
			plt.plot(time,rmse_theta,label = 'RMSE Theta', color="green")	
		else:
			plt.plot(time,rmse_w,label = 'RMSE W', color="red")
		

		plt.title('Error History')
		plt.xlabel('Time')
		# Set the y axis label of the current axis.
		plt.ylabel('Error Values Over Time')
		plt.legend()
		#plt.show()
		fn = './plots/' + val + '_' + 'RMSE'
		plt.savefig(fn)
		plt.clf()


	


def plot_P(P):
	P = np.array(P)
	P_11 = P[:,0,0]
	P_12 = P[:,0,1]
	P_13 = P[:,0,2]
	P_21 = P[:,1,0]
	P_22 = P[:,1,1]
	P_23 = P[:,1,2]
	P_31 = P[:,2,0]
	P_32 = P[:,2,1]
	P_33 = P[:,2,2]
	time = np.array([i for i in range(len(P))])
	plt.plot(time,P_11,label = 'P_11')
	plt.plot(time,P_12,label = 'P_12')
	plt.plot(time,P_13,label = 'P_13')
	plt.plot(time,P_21,label = 'P_21')
	plt.plot(time,P_22,label = 'P_22')
	plt.plot(time,P_23,label = 'P_23')
	plt.plot(time,P_31,label = 'P_31')
	plt.plot(time,P_32,label = 'P_32')
	plt.plot(time,P_33,label = 'P_33')
	plt.title('P Matrix History')
	plt.xlabel('Time')
	# Set the y axis label of the current axis.
	plt.ylabel('P Values Over Time')
	plt.legend()
	#plt.show()
	fn = './plots/' + 'P_Matrix_vs_Time'
	plt.savefig(fn)
	plt.clf()
	
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
		Pk_hist.append(Pk_m1)
		state_hist.append(Xk_pred_corr_k_m1)
	
	return state_hist,Pk_hist	

	
	



if __name__ == '__main__':
	EKF_corr = EKF_corr()
	path = r"C:\Users\shawn paul\Desktop\MTE-546-project\cases\MVPcasefinalwithnoise.csv"
	Uk,Yk = read_csv(path)
	EKF_state_hist,Pk_hist = main_loop(Yk,Uk)
	#get true states
	plot_P(Pk_hist)

	path = r"C:\Users\shawn paul\Desktop\MTE-546-project\cases\MVPcasefinal.csv"
	Uk_real,Yk_real = read_csv(path)
	plot_state(EKF_state_hist,Yk_real)

	
	



