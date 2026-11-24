from pathlib import Path
import json, platform
import numpy as np
import scipy
import mpmath as mp
from scipy.integrate import quad
from scipy.special import j0, j1, jn_zeros, jv, yv, eval_legendre, lpmv
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import pandas as pd

ROOT=Path(__file__).resolve().parent
FIG=ROOT/'figures'; FIG.mkdir(exist_ok=True)
plt.rcParams.update({'font.family':'DejaVu Sans','font.size':10,'axes.grid':True,'grid.alpha':.25,'savefig.dpi':190})
N=10
zeros=jn_zeros(0,N)
def basis(name,k,t):
    t=np.asarray(t)
    if name=='Bessel': return j0(zeros[k]*np.sqrt(t))/abs(j1(zeros[k]))
    if name=='Legendre': return np.sqrt(2*k+1)*eval_legendre(k,2*t-1)
    return np.ones_like(t) if k==0 else np.sqrt(2)*np.cos(k*np.pi*t)
functions=[('exp',lambda t:np.exp(t),r'$f(t)=e^t$',[]),
           ('corner',lambda t:np.abs(t-.4),r'$f(t)=|t-0.4|$',[.4]),
           ('layer',lambda t:(1-t)*np.exp(-8*t),r'$f(t)=(1-t)e^{-8t}$',[])]
names=['Bessel','Legendre','Cosine']
def integrate(fun,points,eps=2e-12):
    return quad(fun,0,1,points=points,epsabs=eps,epsrel=eps,limit=500)[0]
rows=[]; coefs={}; max_error_delta=0.; max_gram=0.
x=np.linspace(0,1,1601)
fig,axes=plt.subplots(1,3,figsize=(12,3.4))
for ax,(key,f,title,pts) in zip(axes,functions):
    ax.plot(x,f(x),'k',lw=2,label='f')
    for name in names:
        c=np.array([integrate(lambda t:float(f(t)*basis(name,k,t)),pts) for k in range(N)])
        def approx(t): return sum(c[k]*basis(name,k,t) for k in range(N))
        if key=='exp' and name=='Legendre':
            mp.mp.dps=60
            cm=[mp.sqrt(2*k+1)*mp.quad(lambda t:mp.exp(t)*mp.legendre(k,2*t-1),[0,1]) for k in range(N)]
            err2=float((mp.exp(2)-1)/2-sum(v*v for v in cm))
        else:
            err2=quad(lambda t:float((f(t)-approx(t))**2),0,1,points=pts,epsabs=1e-28,epsrel=2e-8,limit=500)[0]
            err2_check=quad(lambda t:float((f(t)-approx(t))**2),0,1,points=pts,epsabs=1e-29,epsrel=2e-9,limit=500)[0]
            max_error_delta=max(max_error_delta,abs(err2-err2_check))
        norm2=integrate(lambda t:float(f(t)**2),pts)
        rows.append(dict(function=key,basis=name,N=N,E2=err2,E=np.sqrt(err2),relative_E=np.sqrt(err2/norm2),parseval_residual=abs(err2-(norm2-c@c))))
        coefs[key+'_'+name]=c
        ax.plot(x,approx(x),lw=1.1,label=name)
    ax.set(title=title,xlabel='t'); ax.legend(fontsize=8)
fig.tight_layout(); fig.savefig(FIG/'approximations.png'); plt.close(fig)
for name in names:
    for k in range(N):
        for l in range(k,N):
            value=integrate(lambda t:float(basis(name,k,t)*basis(name,l,t)),[])
            max_gram=max(max_gram,abs(value-(k==l)))
df=pd.DataFrame(rows); df.to_csv(ROOT/'errors.csv',index=False,float_format='%.16e')
pd.DataFrame(coefs).to_csv(ROOT/'coefficients.csv',index_label='k',float_format='%.16e')
fig,ax=plt.subplots(figsize=(6.7,3.4))
for j,name in enumerate(names):
    ax.bar(np.arange(3)+(j-1)*.24,df[df.basis==name].E2,.24,label=name)
ax.set(yscale='log',xticks=np.arange(3),xticklabels=[r'$e^t$',r'$|t-0.4|$',r'$(1-t)e^{-8t}$'],ylabel=r'$E_N^2$'); ax.legend(); fig.tight_layout(); fig.savefig(FIG/'errors.png'); plt.close(fig)
fig,axes=plt.subplots(1,2,figsize=(9,3.1))
x=np.linspace(.04,20,1200)
for nu in [0,1]:
    axes[0].plot(x,jv(nu,x),label=fr'$J_{nu}$')
    axes[1].plot(x,yv(nu,x),label=fr'$Y_{nu}$')
axes[0].set(xlabel='x',ylim=(-.7,1.1)); axes[1].set(xlabel='x',ylim=(-2,1))
for ax in axes: ax.legend()
fig.tight_layout(); fig.savefig(FIG/'bessel.png'); plt.close(fig)
x=np.linspace(-1,1,1001)
fig,axes=plt.subplots(2,3,figsize=(10.5,5.3))
for n in range(1,6): axes.flat[0].plot(x,eval_legendre(n,x),label=fr'$P_{n}$')
axes.flat[0].set_title('m = 0')
for m in range(1,6):
    ax=axes.flat[m]
    for n in range(m,6): ax.plot(x,lpmv(m,n,x),label=fr'$P_{n}^{m}$')
    ax.set_title(f'm = {m}')
for ax in axes.flat: ax.set_xlabel('x'); ax.legend(fontsize=7,ncol=2)
fig.tight_layout(); fig.savefig(FIG/'legendre.png'); plt.close(fig)
meta={'python':platform.python_version(),'numpy':np.__version__,'scipy':scipy.__version__,'N':N,'max_gram_deviation':max_gram,'max_E2_change_tighter_quadrature':max_error_delta,'max_parseval_residual':float(df.parseval_residual.max()),'bessel_zeros':zeros.tolist()}
(ROOT/'validation.json').write_text(json.dumps(meta,indent=2),encoding='utf-8')
print(df.to_string(index=False)); print(json.dumps(meta,indent=2))
