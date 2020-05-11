function DIST = OMMD(Y,Z,opts)
%--------------------------------------------------------------------------
% Evaluate orthogonal version of MMD
%
% Input:
%         Y,Z --- n by k matrix
%        opts --- option structure with fields:
%                 record: 0 -- no print, 1 -- print
%                 mxitr: maximal iteration
%                 sigma: parameter in MMD
%                 scale: whether perform scaling regularization
% Output:
%        DIST --- the distance between measure
%
% -------------------------------------------------------------------------

if ~isfield(opts, 'record');        opts.record = 1;  end
if ~isfield(opts, 'mxitr');         opts.mxitr = 20;  end
if ~isfield(opts, 'sigma');         opts.sigma = 1;  end
if ~isfield(opts, 'scale');         opts.scale = 0;  end
sigma = opts.sigma;
opts.xtol = 1e-12;
opts.gtol = 1e-12;
opts.ftol = 1e-12;
if(opts.scale)
    Y = Y./dismean(Y);
    Z = Z./dismean(Z);
end
sigma = (dismean(Y)+dismean(Z))/2;
F1 = MMDobj(Y,sigma);
F2 = MMDobj(Z,sigma);
F_fin = F1+F2;
[u,s,v] = svd(Y*Z');
X = v*u';
clear s v u;
[X, out]= OptStiefelGBB(X, @MMD_orth, opts, Y, Z, sigma,F_fin);
[DIST,~] = MMD_orth(X,Y,Z,sigma,F_fin);
end

function [F_fin,G_fin] = MMD_orth(X,Y,Z,sigma,F_fin)
[n,k] = size(Y);
Y_new = X*Y;
Y_norm = dot(Y_new,Y_new);
Z_norm = dot(Z,Z);
YZ_dist = Y_new'*Z;
dist = Y_norm'+Z_norm-2*YZ_dist;
MMDist = exp(-dist./sigma^2);
F = sum(sum(MMDist));
Z_tilde = MMDist*Z';
MMD_sum = sum(MMDist,2);
G = (Y_new.*MMD_sum'-Z_tilde')*Y';
F_fin = -2*F/k^2+F_fin;
G_fin = 2*G/k^2/sigma^2*2;
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
