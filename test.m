% load matrix Y and Z here
% Y and Z should be n by k matrix, n -- dimension, k -- number of datapoint
% parameter containing in opt:
%                 record: 0 -- no print, 1 -- print (default 1)
%                 mxitr: maximal iteration (default 20)
%                 sigma: parameter in MMD (default 1)
%                 scale: whether perform scaling regularization (defaule 1)

Y = load("Y.mat");
Y = double(Y.data);
Z = load("Z.mat");
Z = double(Z.data);
opt.mxitr = 20;
DIST = OMMD(Y,Z,opt);
save("DIST.mat","DIST");