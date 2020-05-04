# Generated with SMOP  0.41
from libsmop import *
# ommd.m

    
@function
def OMMD(Y=None,Z=None,opts=None,*args,**kwargs):
    varargin = OMMD.varargin
    nargin = OMMD.nargin

    #-------------------------------------------------------------------------
# Evaluate orthogonal version of MMD
    
    # Input:
#         Y,Z --- n by k matrix
#        opts --- option structure with fields:
#                 record: 0 -- no print, 1 -- print
#                 mxitr: maximal iteration
#                 sigma: parameter in MMD
#                 scale: whether perform scaling regularization
# Output:
#        DIST --- the distance between measure
    
    # -------------------------------------
    
    if logical_not(isfield(opts,'record')):
        opts.record = copy(1)
# ommd.m:17
    
    if logical_not(isfield(opts,'mxitr')):
        opts.mxitr = copy(20)
# ommd.m:18
    
    if logical_not(isfield(opts,'sigma')):
        opts.sigma = copy(1)
# ommd.m:19
    
    if logical_not(isfield(opts,'scale')):
        opts.scale = copy(1)
# ommd.m:20
    
    sigma=opts.sigma
# ommd.m:21
    opts.xtol = copy(1e-12)
# ommd.m:22
    opts.gtol = copy(1e-12)
# ommd.m:23
    opts.ftol = copy(1e-12)
# ommd.m:24
    if (opts.scale):
        Y=Y / dismean(Y)
# ommd.m:26
        Z=Z / dismean(Z)
# ommd.m:27
    
    F1=MMDobj(Y,sigma)
# ommd.m:29
    F2=MMDobj(Z,sigma)
# ommd.m:30
    F_fin=F1 + F2
# ommd.m:31
    u,s,v=svd(dot(Y,Z.T),nargout=3)
# ommd.m:32
    X=dot(v,u.T)
# ommd.m:33
    clear('s','v','u')
    X,out=OptStiefelGBB(X,MMD_orth,opts,Y,Z,sigma,F_fin,nargout=2)
# ommd.m:35
    DIST,__=MMD_orth(X,Y,Z,sigma,F_fin,nargout=2)
# ommd.m:36
    return DIST
    
if __name__ == '__main__':
    pass
    
    
@function
def MMD_orth(X=None,Y=None,Z=None,sigma=None,F_fin=None,*args,**kwargs):
    varargin = MMD_orth.varargin
    nargin = MMD_orth.nargin

    n,k=size(Y,nargout=2)
# ommd.m:40
    Y_new=dot(X,Y)
# ommd.m:41
    Y_norm=dot(Y_new,Y_new)
# ommd.m:42
    Z_norm=dot(Z,Z)
# ommd.m:43
    YZ_dist=dot(Y_new.T,Z)
# ommd.m:44
    dist=Y_norm.T + Z_norm - dot(2,YZ_dist)
# ommd.m:45
    MMDist=exp(- dist / sigma ** 2)
# ommd.m:46
    F=sum(sum(MMDist))
# ommd.m:47
    Z_tilde=dot(MMDist,Z.T)
# ommd.m:48
    MMD_sum=sum(MMDist,2)
# ommd.m:49
    G=dot((multiply(Y_new,MMD_sum.T) - Z_tilde.T),Y.T)
# ommd.m:50
    F_fin=dot(- 2,F) / k ** 2 + F_fin
# ommd.m:51
    G_fin=dot(dot(2,G) / k ** 2 / sigma ** 2,2)
# ommd.m:52
    return F_fin,G_fin
    
if __name__ == '__main__':
    pass
    
    
@function
def MMDobj(Y=None,sigma=None,*args,**kwargs):
    varargin = MMDobj.varargin
    nargin = MMDobj.nargin

    n,k=size(Y,nargout=2)
# ommd.m:56
    Y_norm=dot(Y,Y)
# ommd.m:57
    dist=Y_norm + Y_norm.T - dot(2,(dot(Y.T,Y)))
# ommd.m:58
    F=sum(sum(exp(- dist / sigma ** 2))) / k ** 2
# ommd.m:59
    return F
    
if __name__ == '__main__':
    pass
    
    
@function
def dismean(Y=None,*args,**kwargs):
    varargin = dismean.varargin
    nargin = dismean.nargin

    Y_norm=dot(Y,Y)
# ommd.m:63
    dist=real(sqrt(Y_norm + Y_norm.T - dot(2,(dot(Y.T,Y)))))
# ommd.m:64
    M=mean(mean(dist))
# ommd.m:65
    return M
    
if __name__ == '__main__':
    pass
    