%% ROT.m  
% 
%       Returns the 3x3 rotation matrix given and axis of ratation and an angle (radian)
%
%% Input: 
%
%       vec: an array of shape 3x1, representing the axis of rotation
%
%       theta: a float representing the angle of rotation in radian
%
%% Output:
%
%       R:  3x3 rotation matrix
%

function R = Rot(vec, theta)

    % Applying Rodrigues formula
    vec = vec/norm(vec);
    R = eye(3) + sin(theta)*skew(vec) + (1-cos(theta))*skew(vec)^2;

end