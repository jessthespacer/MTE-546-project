clear all; close all;

%%
syms t b L Lr v theta omega d11 d12 d21 d22 real;
syms delta(t) T(t);

% d11 = -0.8304;
% d12 = 1.6639;
% d21 = -0.0323;
% d22 = 0.4115;
% b = 0.365056;

c1 = d11*T^2 + d12*T;
c2 = d21*T^2 + d22*T;

aF = c1*v + c2;
aD = b*v;

%%
vdot = aF - aD;
thetadot = tan(delta)*v/sqrt(L^2 + Lr^2*tan(delta)^2);

X = [v; theta; omega];

Xdot = [vdot; ...
        thetadot; ...
        simplify(diff(thetadot, t))];

%%
Fk = jacobian(Xdot, X);