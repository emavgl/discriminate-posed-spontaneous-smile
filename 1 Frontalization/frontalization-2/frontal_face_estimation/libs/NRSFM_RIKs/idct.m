function x = idct( y )
%function x = idct( y )
%
% 1D inverse DCT of the columns of y, as defined by MATLAB's idct() function
% (use this function in case the signal processing toolbox is not available)

[N,d] = size(y);

% time and frequency indices
[kk,nn] = meshgrid( 2*(1:N)-1, ((1:d)-1)*pi/(2*N) );

% set up weights
w = sqrt(2/N) * ones(N,1);
w(1) = sqrt(1/N);

% compute idct
x = diag(w) * y * cos( nn .* kk );

x = x(1:d,:)';
