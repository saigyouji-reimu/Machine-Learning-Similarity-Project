function DIST = OMMD2(Y,Z,opts)
%-------------------------------------------------------------------------
% Evaluate orthogonal version of MMD
%
% Input:
%           Y --- n1 by k matrix
%           Z --- n2 by k matrix, n2<n1
%        opts --- option structure with fields:
%                 record: 0 -- no print, 1 -- print
%                 mxitr: maximal iteration
%                 sigma: parameter in MMD
%                 scale: whether perform scaling regularization
% Output:
%        DIST --- the distance between measure
%
% -------------------------------------

if ~isfield(opts, 'record');        opts.record = 1;  end
if ~isfield(opts, 'mxitr');         opts.mxitr = 20;  end
if ~isfield(opts, 'sigma');         opts.sigma = 1;  end
if ~isfield(opts, 'scale');         opts.scale = 0;  end
sigma = opts.sigma;
opts.xtol = 1e-12;
opts.gtol = 1e-12;
opts.ftol = 1e-12;
[n1,~] = size(Y);
[n2,~] = size(Z);
if(opts.scale)
    Y = Y./dismean(Y);
    Z = Z./dismean(Z);
end
F2 = MMDobj(Z,sigma);
F_fin = F2;
[u,s,v] = svd(Y*Z');
X = v*eye(n2,n1)*u';
clear s v u;
[X, out]= OptStiefelGBB(X', @MMD_orth, opts, Y, Z, sigma,F_fin);
[DIST,~] = MMD_orth(X,Y,Z,sigma,F_fin);
end

function [F_fin,G_fin] = MMD_orth(X,Y,Z,sigma,F_fin)
[n,k] = size(Y);
Y_new = X'*Y;
Y_norm = dot(Y_new,Y_new);
Z_norm = dot(Z,Z);
YZ_dist = Y_new'*Z;
dist = Y_norm'+Z_norm-2*YZ_dist;
MMDist = exp(-dist./sigma^2);
F = sum(sum(MMDist));
Z_tilde = MMDist*Z';
MMD_sum = sum(MMDist,2);
G = (Y_new.*MMD_sum'-Z_tilde')*Y';


distY = Y_norm'+Y_norm-2*(Y_new'*Y_new);
MMDistY = exp(-distY./sigma^2);
MMD_sumY = sum(MMDistY,2);
Y_tilde = MMDistY*Y_new';
GY = (Y_new.*MMD_sumY'-Y_tilde')*Y';

F_fin = -2*F/k^2+F_fin+sum(sum(MMDistY))/k^2;
G_fin = 4*(G'-GY')/k^2/sigma^2;
end

function F = MMDobj(Y, sigma)
[n,k] = size(Y);
Y_norm = dot(Y,Y);
dist = Y_norm+Y_norm'-2*(Y'*Y);
F = sum(sum(exp(-dist/sigma^2)))/k^2;
end

function M = dismean(Y)
Y_norm = dot(Y,Y);
dist = real(sqrt(Y_norm+Y_norm'-2*(Y'*Y)));
M = mean(mean(dist));
end
