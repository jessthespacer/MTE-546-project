import sympy
from sympy import sin, cos, Matrix, tan
from sympy import *

F_f,f_f,del_k_m1,Bk_m1,\
f_r,A,C_d,vk,dt,m,thetha,w,L,L_r,del_dot_m1= symbols("F_f \
	f_f del_k_m1 Bk_m1 f_r A C_d vk dt m \
	thetha w L L_r del_dot_m1" )

val_1 = Matrix([vk + (((F_f - f_f)*cos(del_k_m1 - Bk_m1) - f_r*cos(Bk_m1) -\
	0.5*A*C_d*vk*vk))*(1/m)*dt])
val_2 = Matrix([thetha + (L*tan(del_k_m1))/(L**2 + \
 	(L_r**2)*(tan(del_k_m1)**2))*vk*dt])
val_3 = Matrix([w + ((L*(L**2 - (L_r\
	**2)*(tan(del_k_m1)**2)))/((((L_r**2)*(tan(del_k_m1)**2))**2)*(\
	cos(del_k_m1)**2)))*vk*del_dot_m1*dt

	])
v_1_1 = val_1.jacobian(Matrix([vk]))
v_1_2 = val_1.jacobian(Matrix([thetha]))
v_1_3 = val_1.jacobian(Matrix([w]))

v_2_1 = val_2.jacobian(Matrix([vk]))
v_2_2 = val_2.jacobian(Matrix([thetha]))
v_2_3 = val_2.jacobian(Matrix([w]))


v_3_1 = val_3.jacobian(Matrix([vk]))
v_3_2 = val_3.jacobian(Matrix([thetha]))
v_3_3 = val_3.jacobian(Matrix([w]))

#print out first jacobian over here
print('jacobian')
print(v_1_1,':',v_2_1,':',v_3_1 )
print(v_1_2,':',v_2_2,':',v_3_2 )
print(v_1_3,':',v_2_3,':',v_3_3 )

#Output-> 
'''
Matrix([[-1.0*A*C_d*dt*vk/m + 1]]) : Matrix([[L*dt*tan(del_k_m1)/(L**2 + L_r**2*tan(del_k_m1)**2)]]) : Matrix([[L*del_dot_m1*dt*(L**2 - L_r**2*tan(del_k_m1)**2)/(L_r**4*cos(del_k_m1)**2*tan(del_k_m1)**4)]])
Matrix([[0]]) : Matrix([[1]]) : Matrix([[0]])
Matrix([[0]]) : Matrix([[0]]) : Matrix([[1]])
'''