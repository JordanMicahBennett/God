%% Author ~ Christopher Lu
%% Adaptor ~ Jordan Micah Bennett
%% Adaptor ~ Adaptation Designation ~ Thought Curvature Abstraction : "Causal Neural Perturbation Curvature ( Causal Neural Manifold ( Causal Neural Atom ) )"
%% Adaptor ~ Adaptation Intent : The encodement of curvature of MESOSCALE/MACROSCALE abstraction EXPRESSION, in the Belmanian regime. Therein, I shall derive strictly non-intemperate particle-particle interaction sequences.


%% Defined Constants

NP = 1000; % Number of Particles
xsize = 1024;
ysize = 512 ;
denr = 1000;
time = 0.0 ;
step = 0.1 ; % Step Size
endt = 1001; % End Time
Mach = 2.0 ;


%% [Jordan_Post_Scriptum] I take the facing arrays of numbers of allocation as symbolization qua "Christopher Lu's MESOSCALE" fabric, particularly the input signal sequence of particle properties.
%% Allocating Arrays
cen = zeros(2,NP); % X and Y Particle Center
vel = zeros(2,NP); % X and Y Particle Velocity
acc = zeros(2,NP); % X and Y Particle Acceleration
phi = zeros(NP ); % Volume Fraction
max = zeros(NP ); % ANN Max Force Prediction
tau = zeros(NP ); % ANN Relaxation Time Prediction
lvlset = zeros(xsize, ysize); % Level Set Field
volfrc = zeros(xsize, ysize); % Volume Fraction Field
%% Seeding and Initializing Level Set Field
for i = 1:NP
 cen(1, i) = int( randn()*(ysize)+ysize/2) ); % x
 cen(2, i) = int( randn()*(ysize)+ysize/2) ); % y
 for x = -1:1
 for y = -1:1
 lvlset( cen(1,i)+x , cen(2,i)+y ) = 1;
 end
 end
end
%% Calculates Local Volume Fraction Field
for i = 1:xsize
 for j = 1:ysize
 volfrc(i,j) = sum(lvlset( i-30:i+30 , j-30:j+30 ))/(60^2);
 end
end
%% Sets Volume Fraction to Particle
for i = 1:NP
 phi(i) = volfrac( cen(1,i) , cen(2,i) );
end
tau(i) = ARTIFICIAL_NEURAL_NETWORK_TAU(Mach, denr, phi(i), time); % Already Trained ANN
max(i) = ARTIFICIAL_NEURAL_NETWORK_MAX(Mach, denr, phi(i), time);
%% Main Loop: ADVECTION SCHEME
while time < endtime

 for i = 1 : NP
 % frc(1, i) = MRA_ANN(Mach, phi(i), time) % without lifting
 frc(1, i) = max(i)*exp(-time/tau(i));
 frc(2, i) = max(i)*exp(-time/tau(i))*rand(.1);
 acc(1,i) = frc(1,i) / mass; % mass = denr * den_f * vol
 acc(2,i) = frc(2,i) / mass;
 vel(1,i) = vel(1,i) + acc(1,i) * step;
 vel(2,i) = vel(2,i) + acc(2,i) * step;
 cen(1,i) = cen(1,i) + vel(1,i) * step;
 cen(2,i) = cen(2,i) + vel(2,i) * step;
 end
 time = time + step;
 if (mod(time, 10) == 0)
 LEVELSET_IMAGE_PROCESS(cen, lvlset);
 end
end