import numpy as np  
from var_store import var_store
import math as m

class EKF_system:

	def __init__(var_store):
		super.__init__()

	def Fk(self):
		pass

	def Pk(self,Pk_m1):
		return np.matmul(np.matmul(self.Fk,Pk_m1),np.transpose(Fk)) + self.Q

	def del_dot_m1(self,del_k_m1,del_k_m2):
		return (del_k_m1 - del_k_m2)/self.delta_t

	def Bk_m1(self,del_k_m1):
		return m.atan((self.Lr/self.L)*m.tan(del_k_m1))

	def Ff(self):
		return self.Tk_m1/self.r 

	def fi(self,vk_m1,del_k_m1,beta_k_m1):
		return self.mu_i*self.m*self.g*np.sign(vk_m1*cos(del_k_m1-beta_k_m1))

	def pred_step(self):
		pass
######################################################################################

	def P_corr(self):
		pass

	def Kk_corr(self):
		pass

	def Hk(self):
		pass

	def Xk_pred_corr(self):
		pass


	def Yk(self):
		pass

