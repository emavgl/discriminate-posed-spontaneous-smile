
function [x,fval,exitflag,output] = pgDampeGaussNewton( fun, x0, options, varargin )
%function [x,fval,exitflag,output] = pgDampedGaussNewton( fun, x0, options, varargin )

% constants used by the algorithm
delta     = 1e-04;                   % initial damping parameter
DELTA_INC = 1e+01;                   % factor for increasing delta
DELTA_DEC = 1e-02;                   % factor for decreasing delta
DELTA_MIN = 1e-20;                   % very small (< eps)!
DELTA_MAX = 1e+20;                   % very large!

exitflag = nan;                      % final status of algorithm
msg = '';                            % output status message
fcount = 0;                          % number of cost function evaluations

x = x0;
nx = numel( x0 );                    % number of unknowns
Id = speye( nx );                    % used for damping Hessian matrix
%g  = zeros( nx, 1 );                 % gradient vector
%H  = zeros( nx,nx );                 % Hessian matrix

warning('off', 'MATLAB:nearlySingularMatrix');     % OK! Damping fixes Hessian
options.OutputFcn( [], [], 'init' );

for iter = 0:options.MaxIter;
        
    % (0) evaluate function and compute gradient and Hessian
    [fval,g,H] = fun( x, varargin{:} ); 
    fcount = fcount + 1;
    
    % (1) call user output function
    optvals = struct('funccount',fcount,'fval',fval,'gradient',g,'iteration',iter,'lambda',delta);
    stop = options.OutputFcn( x, optvals, options.Display );
    if (stop), msg = 'terminated by output function'; exitflag = -1; break; end
        
    % (2) damping loop: repeat solving for dx until f(x-dx) < f(x) or converged
    %[V,D] = eig( H );                      % dx = inv(H+dI)g = V inv(D+dI) V'g
    %D = diag(D);
    %g = V'*g;
    
    while true
        %dx = (H + delta*Id) \ g;
        dx = pinv(H + delta*Id) * g;
        %dx = V * diag(1./(D+delta)) * g;
        newx = x - dx;        

        newfval = fun( newx, varargin{:} ); 
        fcount = fcount + 1;
        if (newfval < fval), break, end
            
        delta = delta * DELTA_INC;
        if (delta > DELTA_MAX)
            global err_no_descent; err_no_descent = true;
            msg = 'Could not find descent direction!'; exitflag = 1; break
        end
    end
    if ~isnan(exitflag), break, end     % break again if delta is too high
    
    % update x, decrease delta
    x = newx;
    delta = max( delta * DELTA_DEC, DELTA_MIN );
       
    % convergence test
    if ( fval-newfval < options.TolFun )
        msg = 'Change in function value is less than TolFun.'; exitflag = 3; break
    end 
end
warning('on', 'MATLAB:nearlySingularMatrix');

if (iter == options.MaxIter), msg = 'Reached maximum number of iterations.'; exitflag = 0; end

output = struct('iterations',iter,'funcCount',fcount,'algorithm','pgDampedNewton','message',msg);
options.OutputFcn( x, optvals, 'done' );

% ------------------------------------------------------------------------------
