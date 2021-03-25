import numpy as np  
from var_store import var_store
import math as m

class EKF_pred(var_store):

	def __init__(self):
		super().__init__()


	def Tk(self):
		#subject to change if time dependant	
		return self.Tk_m1

	def Pk(self,Pk_m1):
		Fk = self.Fk()
		return np.matmul(np.matmul(Fk,Pk_m1),np.transpose(Fk)) + self.Q

	#Steering angle rate [rad/s]
	def del_dot_m1(self,del_k_m1,del_k_m2):
		return (del_k_m1 - del_k_m2)/self.dt

	def Bk_m1(self,del_k_m1):
		return m.atan((self.L_r/self.L)*m.tan(del_k_m1))

	def F_f(self):
		return self.Tk()/self.r 

	def fi(self,vk_m1,del_k_m1,beta_k_m1):
		return self.mu_i*self.m*self.g*np.sign(vk_m1*m.cos(del_k_m1-beta_k_m1))

	def Fk(self):
		#jacobian of f
		#3x3 matrix
		#insert jacobian from pde file
		#returns identity for now
		return np.random.rand(3,3)



	
	def Xk_pred(self,Xk_pred_corr_k_m1,Uk_m1):
		vk_m1 = Xk_pred_corr_k_m1[0,:]
		thetha_k_m1 = Xk_pred_corr_k_m1[1,:]
		w_k_m1 = Xk_pred_corr_k_m1[2,:]
		T_k_m1 = Uk_m1[0,:]
		del_k_m1 = Uk_m1[1,:]
		del_k_m2 = del_k_m1
		vk = vk_m1 + (((self.F_f() - self.f_f)*m.cos(del_k_m1 - self.Bk_m1(del_k_m1)) - self.f_r*m.cos(self.Bk_m1(del_k_m1)) -0.5*self.A*self.C_d*vk_m1*vk_m1)*self.dt)/self.m
		thetha_k = thetha_k_m1 + (self.L*m.tan(del_k_m1))/(self.L**2 + (self.L_r**2)*(m.tan(del_k_m1)**2))*vk_m1*self.dt
		w_k = w_k_m1 + ((self.L*(self.L**2 - (self.L_r**2)*(m.tan(del_k_m1)**2)))/((((self.L_r**2)*(m.tan(del_k_m1)**2))**2)*(m.cos(del_k_m1)**2)))*vk_m1*self.del_dot_m1(del_k_m1,del_k_m2)*self.dt
		#insert assertion on shape type
		return np.array([vk,thetha_k,w_k])


