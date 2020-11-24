%% skew.m  
% 
%       Returns the 3x3 skew matrix given a 3x1 vector
%
%% Input: 
%
%       v: 3x1 vector
%
%% Output:
%
%       A: 3x3 skew matrix
%

function A = skew(v)
    v_size = size(v);
    if v_size(1) ~= 3 || v_size(2) ~= 1
        error('Invalid input vector size, input vector size should be 3x1')
    end
    A = [[0, -v(3), v(2)],
            [v(3), 0, -v(1)],
            [-v(2), v(1), 0]];
    
end
