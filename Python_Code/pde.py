import sympy
from sympy import sin, cos, Matrix, tan
from sympy import *

del_k_m1,Bk_m1, vk,dt,thetha,w,L,L_r,del_dot_m1,b,c1,c2,vk_m1= symbols("del_k_m1 Bk_m1 vk dt thetha w L L_r del_dot_m1 b c1 c2 vk_m1" )

val_1 = Matrix([vk + (c1*vk + c2 - b*vk)*dt])
val_2 = Matrix([thetha + ((L*tan(del_k_m1))/(L**2 + (L_r**2)*(tan(del_k_m1)**2)))*vk*dt])
	
val_3 = Matrix([w + ((L*(L**2 - (L_r**2)*(tan(del_k_m1)**2)))/(((L**2 + (L_r**2)*(tan(del_k_m1)**2))**2)*(cos(del_k_m1)**2)))*vk*del_dot_m1*dt])
	
v_1_1 = val_1.jacobian(Matrix([vk]))
v_1_2 = val_1.jacobian(Matrix([thetha]))
v_1_3 = val_1.jacobian(Matrix([w]))
print('v_1_1,v_1_2,v_1_3')
print(v_1_1,v_1_2,v_1_3)

v_2_1 = val_2.jacobian(Matrix([vk]))
v_2_2 = val_2.jacobian(Matrix([thetha]))
v_2_3 = val_2.jacobian(Matrix([w]))
print('v_2_1,v_2_2,v_2_3')
print(v_2_1,v_2_2,v_2_3)

v_3_1 = val_3.jacobian(Matrix([vk]))
v_3_2 = val_3.jacobian(Matrix([thetha]))
v_3_3 = val_3.jacobian(Matrix([w]))
print('v_3_1,v_3_2,v_3_3')
print(v_3_1,v_3_2,v_3_3)

