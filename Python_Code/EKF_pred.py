import numpy as np  
from var_store import var_store
import math as m

class EKF_pred(var_store):

	def __init__(self):
		super().__init__()

	#!Throttle setting, 0<T<1
	def Tk(self):
		#subject to change if time dependant
		#get from csv data
		Tk = self.Tk_m1	
		return Tk_m1

	#!State covariance
	def Pk(self,Pk_m1):
		Fk = self.Fk()
		return np.matmul(np.matmul(Fk,Pk_m1),np.transpose(Fk)) + self.Q

	#!Steering angle rate [rad/s]
	def del_dot_m1(self,del_k_m1,del_k_m2):
		return (del_k_m1 - del_k_m2)/self.dt
	#!
	def ad(self,v):
		return self.b()*v
	#!
	def af(self,T,v):
		return self.c1(T)*v + self.c2(T)

	#!Specific power model parameter 1 [m/s2]
	def c1(self,T):
		return -0.8304*(T**2)+1.6639*T

	#!Specific power model parameter 1 [m/s2]
	def c2(self,T):
		return -0.0323*T**2+0.4115*T

	#direction of motion for COG
	def Bk_m1(self,del_k_m1):
		return m.atan((self.L_r/self.L)*m.tan(del_k_m1))


	#!jacobian of f
	def Fk(self):
		return np.random.rand(3,3)

	#!friction coefficient
	def b(self,s=1):
		return 0.365056*(s**(-1))



	#!State prediction
	def Xk_pred(self,Xk_pred_corr_k_m1,Uk_m1):
		assert Xk_pred_corr_k_m1.shape == (3,1), 'incorr Xk_pred_corr_k_m1 shape'
		assert Uk_m1.shape == (2,1), 'incorr Uk_m1 shape'
		vk_m1 = Xk_pred_corr_k_m1[0,:]
		thetha_k_m1 = Xk_pred_corr_k_m1[1,:]
		w_k_m1 = Xk_pred_corr_k_m1[2,:]
		T_k_m1 = Uk_m1[0,:]
		del_k_m1 = Uk_m1[1,:]
		del_k_m2 = del_k_m1
		vk = vk_m1 + (self.af(T_k_m1,vk_m1) - self.ad(vk_m1))*self.dt
		thetha_k = thetha_k_m1 + (self.L*m.tan(del_k_m1))/(self.L**2 + (self.L_r**2)*(m.tan(del_k_m1)**2))*vk_m1*self.dt
		w_k = w_k_m1 + ((self.L*(self.L**2 - (self.L_r**2)*(m.tan(del_k_m1)**2)))/((((self.L_r**2)*(m.tan(del_k_m1)**2))**2)*(m.cos(del_k_m1)**2)))*vk_m1*self.del_dot_m1(del_k_m1,self.del_k_m2)*self.dt
		self.del_k_m2 = del_k_m1
		#insert assertion on shape type
		ret_val = np.array([vk,thetha_k,w_k])
		assert ret_val.shape == (3,1), 'incorr Xk_pred shape'
		return ret_val



	'''
	def F_f(self):
		return self.Tk()/self.r 

	def fi(self,vk_m1,del_k_m1,beta_k_m1):
		return self.mu_i*self.m*self.g*np.sign(vk_m1*m.cos(del_k_m1-beta_k_m1))
	'''


if __name__ == '__main__':
	E = EKF_pred()
	Xk_pred_corr_k_m1 = np.ones((3,1))
	Uk_m1 = np.ones((2,1))
	v = E.Xk_pred(Xk_pred_corr_k_m1,Uk_m1)

