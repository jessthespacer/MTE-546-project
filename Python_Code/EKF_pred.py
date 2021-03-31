import numpy as np  
from var_store import var_store
import math as m
from math import tan,cos


class EKF_pred(var_store):

	def __init__(self):
		super().__init__()

	def dt(self):
		return 0.05

	#!jacobian of f
	def Fk(self,Xk_pred_corr_k_m1,Uk_m1):
		#values
		T = Uk_m1[0,:][0]
		del_k_m1 = Uk_m1[1,:][0]
		#state constants
		dt = self.dt()
		b = self.b()		
		c1 = self.c1(T)
		L = self.L
		L_r = self.L_r
		#vars from input		
		del_dot_m1	= self.del_dot_m1(del_k_m1)	
		#print('del_dot_m1',del_dot_m1)
		row_1 = np.array([dt*(-b + c1) + 1,0,0])
		row_2 = np.array([L*dt*tan(del_k_m1)/(L**2 + L_r**2*tan(del_k_m1)**2),1,0])
		row_3 = np.array([L*del_dot_m1*dt*(L**2 - L_r**2*tan(del_k_m1)**2)/((L**2 + L_r**2*tan(del_k_m1)**2)**2*cos(del_k_m1)**2),0,1])
		ret_val =  np.stack([row_1,row_2,row_3])
		assert ret_val.shape == (3,3), 'weird jacobian size'
		return ret_val

	#State covariance
	def Pk(self,Pk_m1,Xk,Uk):
		Fk = self.Fk(Xk,Uk)
		self.Pk_val = np.matmul(np.matmul(Fk,Pk_m1),np.transpose(Fk)) + self.Q
		return self.Pk_val


	#Steering angle rate [rad/s]
	def del_dot_m1(self,del_k_m1):
		del_dot_m1 = (del_k_m1 - self.del_k_m2)/self.dt()
		#print('self.del_k_m2 ',type(self.del_k_m2))
		return del_dot_m1

	#Deceleration caused by drag [m/s2]
	def ad(self,v):
		return self.b()*v

	#Acceleration caused by engine [m/s2]
	def af(self,T,v):
		return self.c1(T)*v + self.c2(T)

	#Specific power model parameter 1 [m/s2]
	def c1(self,T):
		return -0.8304*(T**2)+1.6639*T

	#Specific power model parameter 1 [m/s2]
	def c2(self,T):
		return -0.0323*(T**2)+0.4115*T

	#direction of motion for COG
	def Bk_m1(self,del_k_m1):
		return m.atan((self.L_r/self.L)*m.tan(del_k_m1))

	#friction coefficient
	def b(self,s=1):
		return 0.365056*(s**(-1))

	#!State prediction
	def Xk_pred(self,Xk_pred_corr_k_m1,Uk_m1):
		#ceck if inut makes sense
		assert Xk_pred_corr_k_m1.shape == (3,1), 'incorr Xk_pred_corr_k_m1 shape'
		assert Uk_m1.shape == (2,1), 'incorr Uk_m1 shape'
		#get all values from rows
		vk_m1 = Xk_pred_corr_k_m1[0,:]
		thetha_k_m1 = Xk_pred_corr_k_m1[1,:]
		w_k_m1 = Xk_pred_corr_k_m1[2,:]
		T_k_m1 = Uk_m1[0,:]
		del_k_m1 = Uk_m1[1,:]
		#check if sterring ange makes sense
		#remove assertion when using real data
		#assert del_k_m1 < m.pi/2 and del_k_m1 > -1 *m.pi/2, 'steering agle out of bounds'
		#calculate current iteration
		vk = vk_m1 + (self.af(T_k_m1,vk_m1) - self.ad(vk_m1))*self.dt()
		#calc_2
		thetha_k = thetha_k_m1 + \
				((self.L*m.tan(del_k_m1))/(self.L**2 + (self.L_r**2)*(m.tan(del_k_m1)**2)))\
						*vk_m1*self.dt()
		#calc_3
		w_k = w_k_m1 + ((self.L*(self.L**2 - (self.L_r**2)*\
			(m.tan(del_k_m1)**2)))/(((self.L**2 + (self.L_r**2)*(m.tan(del_k_m1)**2))**2)\
				*(m.cos(del_k_m1)**2)))\
						*vk_m1*self.del_dot_m1(del_k_m1)*self.dt()
		#save value for next iteration
		self.del_k_m2 = del_k_m1[0]		
		#insert assertion on shape type
		ret_val = np.array([vk,thetha_k,w_k])
		assert ret_val.shape == (3,1), 'incorr Xk_pred shape'
		return ret_val







if __name__ == '__main__':
	pass
	

