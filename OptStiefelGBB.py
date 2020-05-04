# Generated with SMOP  0.41
from libsmop import *
# .\OptStiefelGBB.m

    
@function
def OptStiefelGBB(X=None,fun=None,opts=None,varargin=None,*args,**kwargs):
    varargin = OptStiefelGBB.varargin
    nargin = OptStiefelGBB.nargin

    #-------------------------------------------------------------------------
# curvilinear search algorithm for optimization on Stiefel manifold
    
    #   min F(X), S.t., X'*X = I_k, where X \in R^{n,k}
    
    #   H = [G, X]*[X -G]'
#   U = 0.5*tau*[G, X];    V = [X -G]
#   X(tau) = X - 2*U * inv( I + V'*U ) * V'*X
    
    #   -------------------------------------
#   U = -[G,X];  V = [X -G];  VU = V'*U;
#   X(tau) = X - tau*U * inv( I + 0.5*tau*VU ) * V'*X
    
    
    # Input:
#           X --- n by k matrix such that X'*X = I
#         fun --- objective function and its gradient:
#                 [F, G] = fun(X,  data1, data2)
#                 F, G are the objective function value and gradient, repectively
#                 data1, data2 are addtional data, and can be more
#                 Calling syntax:
#                   [X, out]= OptStiefelGBB(X0, @fun, opts, data1, data2);
    
    #        opts --- option structure with fields:
#                 record = 0, no print out
#                 mxitr       max number of iterations
#                 xtol        stop control for ||X_k - X_{k-1}||
#                 gtol        stop control for the projected gradient
#                 ftol        stop control for |F_k - F_{k-1}|/(1+|F_{k-1}|)
#                             usually, max{xtol, gtol} > ftol
#   
# Output:
#           X --- solution
#         Out --- output information
    
    # -------------------------------------
# For example, consider the eigenvalue problem F(X) = -0.5*Tr(X'*A*X);
    
    # function demo
# 
# function [F, G] = fun(X,  A)
#   G = -(A*X);
#   F = 0.5*sum(dot(G,X,1));
# end
# 
# n = 1000; k = 6;
# A = randn(n); A = A'*A;
# opts.record = 0; #
# opts.mxitr  = 1000;
# opts.xtol = 1e-5;
# opts.gtol = 1e-5;
# opts.ftol = 1e-8;
# 
# X0 = randn(n,k);    X0 = orth(X0);
# tic; [X, out]= OptStiefelGBB(X0, @fun, opts, A); tsolve = toc;
# out.fval = -2*out.fval; # convert the function value to the sum of eigenvalues
# fprintf('\nOptM: obj: #7.6e, itr: #d, nfe: #d, cpu: #f, norm(XT*X-I): #3.2e \n', ...
#             out.fval, out.itr, out.nfe, tsolve, norm(X'*X - eye(k), 'fro') );
# 
# end
#-------------------------------------------------------------------------
    
    # Size information
    if isempty(X):
        error('input X is an empty matrix')
    else:
        n,k=size(X,nargout=2)
# .\OptStiefelGBB.m:69
    
    if nargin < 2:
        error('[X, out]= OptStiefelGBB(X0, @fun, opts)')
    
    if nargin < 3:
        opts=[]
# .\OptStiefelGBB.m:73
    
    if logical_not(isfield(opts,'X0')):
        opts.X0 = copy([])
# .\OptStiefelGBB.m:75
    
    if logical_not(isfield(opts,'xtol')):
        opts.xtol = copy(1e-06)
# .\OptStiefelGBB.m:76
    
    if logical_not(isfield(opts,'gtol')):
        opts.gtol = copy(1e-06)
# .\OptStiefelGBB.m:77
    
    if logical_not(isfield(opts,'ftol')):
        opts.ftol = copy(1e-12)
# .\OptStiefelGBB.m:78
    
    # parameters for control the linear approximation in line search,
    if logical_not(isfield(opts,'tau')):
        opts.tau = copy(0.001)
# .\OptStiefelGBB.m:81
    
    if logical_not(isfield(opts,'rhols')):
        opts.rhols = copy(0.0001)
# .\OptStiefelGBB.m:82
    
    if logical_not(isfield(opts,'eta')):
        opts.eta = copy(0.1)
# .\OptStiefelGBB.m:83
    
    if logical_not(isfield(opts,'retr')):
        opts.retr = copy(0)
# .\OptStiefelGBB.m:84
    
    if logical_not(isfield(opts,'gamma')):
        opts.gamma = copy(0.85)
# .\OptStiefelGBB.m:85
    
    if logical_not(isfield(opts,'STPEPS')):
        opts.STPEPS = copy(1e-10)
# .\OptStiefelGBB.m:86
    
    if logical_not(isfield(opts,'nt')):
        opts.nt = copy(5)
# .\OptStiefelGBB.m:87
    
    if logical_not(isfield(opts,'mxitr')):
        opts.mxitr = copy(1000)
# .\OptStiefelGBB.m:88
    
    if logical_not(isfield(opts,'record')):
        opts.record = copy(0)
# .\OptStiefelGBB.m:89
    
    if logical_not(isfield(opts,'tiny')):
        opts.tiny = copy(1e-13)
# .\OptStiefelGBB.m:90
    
    #-------------------------------------------------------------------------------
# copy parameters
    xtol=opts.xtol
# .\OptStiefelGBB.m:94
    gtol=opts.gtol
# .\OptStiefelGBB.m:95
    ftol=opts.ftol
# .\OptStiefelGBB.m:96
    rhols=opts.rhols
# .\OptStiefelGBB.m:97
    STPEPS=opts.STPEPS
# .\OptStiefelGBB.m:98
    eta=opts.eta
# .\OptStiefelGBB.m:99
    gamma=opts.gamma
# .\OptStiefelGBB.m:100
    retr=opts.retr
# .\OptStiefelGBB.m:101
    record=opts.record
# .\OptStiefelGBB.m:102
    nt=opts.nt
# .\OptStiefelGBB.m:103
    crit=ones(nt,3)
# .\OptStiefelGBB.m:104
    tiny=opts.tiny
# .\OptStiefelGBB.m:105
    #-------------------------------------------------------------------------------
    
    # Initial function value and gradient
# prepare for iterations
    F,G=feval(fun,X,varargin[arange()],nargout=2)
# .\OptStiefelGBB.m:110
    out.nfe = copy(1)
# .\OptStiefelGBB.m:110
    GX=dot(G.T,X)
# .\OptStiefelGBB.m:111
    if retr == 1:
        invH=copy(true)
# .\OptStiefelGBB.m:114
        if k < n / 2:
            invH=copy(false)
# .\OptStiefelGBB.m:114
            eye2k=eye(dot(2,k))
# .\OptStiefelGBB.m:114
        if invH:
            GXT=dot(G,X.T)
# .\OptStiefelGBB.m:116
            H=dot(0.5,(GXT - GXT.T))
# .\OptStiefelGBB.m:116
            RX=dot(H,X)
# .\OptStiefelGBB.m:116
        else:
            U=concat([G,X])
# .\OptStiefelGBB.m:118
            V=concat([X,- G])
# .\OptStiefelGBB.m:118
            VU=dot(V.T,U)
# .\OptStiefelGBB.m:118
            #VX = VU(:,k+1:end); #VX = V'*X;
            VX=dot(V.T,X)
# .\OptStiefelGBB.m:121
    
    dtX=G - dot(X,GX)
# .\OptStiefelGBB.m:124
    nrmG=norm(dtX,'fro')
# .\OptStiefelGBB.m:124
    Q=1
# .\OptStiefelGBB.m:126
    Cval=copy(F)
# .\OptStiefelGBB.m:126
    tau=opts.tau
# .\OptStiefelGBB.m:126
    # Print iteration header if debug == 1
    if (opts.record == 1):
        fid=1
# .\OptStiefelGBB.m:130
        fprintf(fid,'----------- Gradient Method with Line search ----------- \n')
        fprintf(fid,'%4s %8s %8s %10s %10s\n','Iter','tau','F(X)','nrmG','XDiff')
    
    # main iteration
    for itr in arange(1,opts.mxitr).reshape(-1):
        XP=copy(X)
# .\OptStiefelGBB.m:138
        FP=copy(F)
# .\OptStiefelGBB.m:138
        GP=copy(G)
# .\OptStiefelGBB.m:138
        dtXP=copy(dtX)
# .\OptStiefelGBB.m:138
        nls=1
# .\OptStiefelGBB.m:141
        deriv=dot(rhols,nrmG ** 2)
# .\OptStiefelGBB.m:141
        while 1:

            # calculate G, F,
            if retr == 1:
                if invH:
                    X,infX=linsolve(eye(n) + dot(tau,H),XP - dot(tau,RX),nargout=2)
# .\OptStiefelGBB.m:146
                else:
                    aa,infR=linsolve(eye2k + dot((dot(0.5,tau)),VU),VX,nargout=2)
# .\OptStiefelGBB.m:148
                    X=XP - dot(U,(dot(tau,aa)))
# .\OptStiefelGBB.m:149
            else:
                X,RR=myQR(XP - dot(tau,dtX),k,nargout=2)
# .\OptStiefelGBB.m:152
            if norm(dot(X.T,X) - eye(k),'fro') > tiny:
                X=myQR(X,k)
# .\OptStiefelGBB.m:155
            F,G=feval(fun,X,varargin[arange()],nargout=2)
# .\OptStiefelGBB.m:157
            out.nfe = copy(out.nfe + 1)
# .\OptStiefelGBB.m:158
            if le(F,Cval - dot(tau,deriv)) or ge(nls,5):
                break
            tau=dot(eta,tau)
# .\OptStiefelGBB.m:163
            nls=nls + 1
# .\OptStiefelGBB.m:163

        GX=dot(G.T,X)
# .\OptStiefelGBB.m:166
        if retr == 1:
            if invH:
                GXT=dot(G,X.T)
# .\OptStiefelGBB.m:169
                H=dot(0.5,(GXT - GXT.T))
# .\OptStiefelGBB.m:169
                RX=dot(H,X)
# .\OptStiefelGBB.m:169
            else:
                U=concat([G,X])
# .\OptStiefelGBB.m:171
                V=concat([X,- G])
# .\OptStiefelGBB.m:171
                VU=dot(V.T,U)
# .\OptStiefelGBB.m:171
                #VX = VU(:,k+1:end); # VX = V'*X;
                VX=dot(V.T,X)
# .\OptStiefelGBB.m:174
        dtX=G - dot(X,GX)
# .\OptStiefelGBB.m:177
        nrmG=norm(dtX,'fro')
# .\OptStiefelGBB.m:177
        S=X - XP
# .\OptStiefelGBB.m:178
        XDiff=norm(S,'fro') / sqrt(n)
# .\OptStiefelGBB.m:178
        tau=opts.tau
# .\OptStiefelGBB.m:179
        FDiff=abs(FP - F) / (abs(FP) + 1)
# .\OptStiefelGBB.m:179
        Y=dtX - dtXP
# .\OptStiefelGBB.m:182
        SY=abs(iprod(S,Y))
# .\OptStiefelGBB.m:182
        if mod(itr,2) == 0:
            tau=(norm(S,'fro') ** 2) / SY
# .\OptStiefelGBB.m:183
        else:
            tau=SY / (norm(Y,'fro') ** 2)
# .\OptStiefelGBB.m:184
        tau=max(min(tau,1e+20),1e-20)
# .\OptStiefelGBB.m:185
        if ge(record,1):
            fprintf('%4d  %3.2e  %4.3e  %3.2e  %3.2e  %3.2e  %2d\n',itr,tau,F,nrmG,XDiff,FDiff,nls)
            #    itr, tau, F, nrmG, XDiff, alpha1, alpha2);
        crit[itr,arange()]=concat([nrmG,XDiff,FDiff])
# .\OptStiefelGBB.m:194
        mcrit=mean(crit(arange(itr - min(nt,itr) + 1,itr),arange()),1)
# .\OptStiefelGBB.m:195
        #if (XDiff < xtol || nrmG < gtol ) || FDiff < ftol
    #if ( XDiff < xtol && FDiff < ftol ) || nrmG < gtol
    #if ( XDiff < xtol || FDiff < ftol ) || nrmG < gtol
    #if any(mcrit < [gtol, xtol, ftol])
        if (XDiff < xtol and FDiff < ftol) or nrmG < gtol or all(mcrit(arange(2,3)) < dot(10,concat([xtol,ftol]))):
            out.msg = copy('converge')
# .\OptStiefelGBB.m:202
            break
        Qp=copy(Q)
# .\OptStiefelGBB.m:206
        Q=dot(gamma,Qp) + 1
# .\OptStiefelGBB.m:206
        Cval=(dot(dot(gamma,Qp),Cval) + F) / Q
# .\OptStiefelGBB.m:206
    
    if ge(itr,opts.mxitr):
        out.msg = copy('exceed max iteration')
# .\OptStiefelGBB.m:210
    
    out.feasi = copy(norm(dot(X.T,X) - eye(k),'fro'))
# .\OptStiefelGBB.m:213
    if out.feasi > 1e-13:
        #X = MGramSchmidt(X);
        X=myQR(X,k)
# .\OptStiefelGBB.m:216
        F,G=feval(fun,X,varargin[arange()],nargout=2)
# .\OptStiefelGBB.m:217
        out.nfe = copy(out.nfe + 1)
# .\OptStiefelGBB.m:218
        out.feasi = copy(norm(dot(X.T,X) - eye(k),'fro'))
# .\OptStiefelGBB.m:219
    
    out.nrmG = copy(nrmG)
# .\OptStiefelGBB.m:222
    out.fval = copy(F)
# .\OptStiefelGBB.m:223
    out.itr = copy(itr)
# .\OptStiefelGBB.m:224
    return X,out
    
if __name__ == '__main__':
    pass
    
    
@function
def iprod(x=None,y=None,*args,**kwargs):
    varargin = iprod.varargin
    nargin = iprod.nargin

    #a = real(sum(sum(x.*y)));
    a=real(sum(sum(multiply(conj(x),y))))
# .\OptStiefelGBB.m:229
    return a
    
if __name__ == '__main__':
    pass
    
    
@function
def myQR(XX=None,k=None,*args,**kwargs):
    varargin = myQR.varargin
    nargin = myQR.nargin

    Q,RR=qr(XX,0,nargout=2)
# .\OptStiefelGBB.m:235
    diagRR=sign(diag(RR))
# .\OptStiefelGBB.m:236
    ndr=diagRR < 0
# .\OptStiefelGBB.m:236
    if nnz(ndr) > 0:
        Q=dot(Q,spdiags(diagRR,0,k,k))
# .\OptStiefelGBB.m:238
    
    return Q,RR
    
if __name__ == '__main__':
    pass
    