close all; clear all;

%% Vehicle design parameters and constants
params.A = 2.1;
params.Cd = 0.35;
params.L = 2.7;
params.Lr = params.L/2;
params.m = 1300;
params.mu = 0.1;
params.r = 0.3175;

params.g = 9.81;

%% Simulation setup
X0 = [30; 0];
U = @(t) [0; deg2rad(10)];

%% Simulation loop
odefun = @(t, y) f(y, U(t), params);
[t, y] = ode45(odefun, [0 15], X0);

%% Plot results
plot(t, y(:, 1));
ylabel('Velocity [m/s]');
yyaxis right;
plot(t, rad2deg(y(:, 2)));
ylabel('Orientation [deg]');
ax = gca;
ax.YAxis(1).Color = 'k';
ax.YAxis(2).Color = 'k';
xlabel('Time [s]');
title('System response');
legend({'Velocity', 'Orientation'});
grid on;
grid minor;

%% Function definitions
function Xdot = f(X, U, params)
    A = params.A;
    Cd = params.Cd;
    L = params.L;
    Lr = params.Lr;
    m = params.m;
    mu = params.mu;
    r = params.r;
    g = params.g;
    
    v = X(1);
    theta = X(2);
    T = U(1);
    delta = U(2);
    beta = atan(Lr/L*tan(delta));
    
    Ff = T/r;
    dir = sign(v*cos(delta - beta));
    ff = dir*mu*m*g;
    fr = dir*mu*m*g;
    
    vdot = ((Ff - ff)*cos(delta - beta) - fr*cos(beta) - 0.5*A*Cd*v^2)/m;
    thetadot = v*(L*tan(delta))/(L^2 + Lr^2*tan(delta)^2);
    
    Xdot = [vdot; thetadot];
end