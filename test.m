% load matrix Y and Z here
% Y and Z should be n by k matrix, n -- dimension, k -- number of datapoint
% parameter containing in opt:
%                 record: 0 -- no print, 1 -- print (default 1)
%                 mxitr: maximal iteration (default 20)
%                 sigma: parameter in MMD (default 1)
%                 scale: whether perform scaling regularization (defaule 1)
DIST = zeros(10,1);
for i=1:10
Y = load("xxx.mat");
Y = double(Y);
Z = load("xxx.mat");
Z = double(Z);
opt.mxitr = 20;
DIST(i) = OMMD(Y,Z,opt);
end
save("xxx.mat","DIST");