"""
	
    Implementation of Deep Active Inference for
    General Artificial Intelligence
    
    (c) Kai Ueltzhoeffer, 2017

"""

# Imports
import cPickle
import timeit
import scipy

import matplotlib.pyplot as plt

import numpy
import scipy

import theano
import theano.tensor as T
from theano.ifelse import ifelse

from theano import pprint as pp

from theano.sandbox.rng_mrg import MRG_RandomStreams as RandomStreams  

# Parameters
n_s = 20 # States

n_o = 1 # Sensory Input
n_oh = 1 # Rewards (OPTIONAL!)
n_oa = 1 # Proprioception (OPTIONAL!)

n_run_steps = 30
n_proc = 1
n_perturbations = 1000# 20000

n_steps = 1000000

sig_min_obs = 0.001
sig_min_states = 0.001

learning_rate = 1e-3

eps = 1e-3

init_sig_obs = -3.0
init_sig_states = -3.0

# Initialize RNG
ii32 = numpy.iinfo(numpy.int32)
theano_rng = RandomStreams(numpy.random.randint(ii32.max)) # ADD RANDOM SEED!  

# Helper Functions and Classes

def initweight(shape1, shape2, minval =-0.05, maxval = 0.05):
    val = numpy.random.rand(
        shape1, shape2
    )
    
    val = minval + (maxval - minval)*val
    
    #val = numpy.random.randn(shape1, shape2) / numpy.sqrt(shape1) # "Xavier" initialization
    
    return val.astype(theano.config.floatX) 
    
def initconst(shape1, shape2, val = 0.0):
    val = val*numpy.ones(
        (shape1,shape2),
        dtype=theano.config.floatX
    )
    
    return val.astype(theano.config.floatX)
    
def initortho(shape1, shape2):
        x = numpy.random.normal(0.0, 0.01, (shape1, shape2))
        xo = scipy.linalg.orth(x)
        
        return xo.astype(theano.config.floatX)

#ADAM Optimizer, following Kingma & Ba (2015)
class Adam(object):

    def __init__(self, grads, p, b1, b2, alpha, epsilon = 10e-8):
    
        #self.L = L
        self.p = p
        self.b1 = b1
        self.b2 = b2
        self.alpha = alpha
        
        self.t = theano.shared( value = numpy.cast[theano.config.floatX](1.0))
        self.t_next = self.t + 1
        
        #self.g = T.grad(cost = theano.gradient.grad_clip(L,-1.0,1.0), wrt = p)
        self.g = grads.astype(dtype = theano.config.floatX) #T.grad(cost = L, wrt = p)
        self.m = theano.shared( value=numpy.zeros_like(
                p.get_value(),
                dtype=theano.config.floatX
            ),
            name='m',
            borrow=True,
            broadcastable=self.p.broadcastable
        )
        self.m_next = self.b1*self.m + (1 - self.b1)*self.g
        self.v = theano.shared( value=numpy.zeros_like(
                p.get_value(),
                dtype=theano.config.floatX
            ),
            name='v',
            borrow=True,
            broadcastable=self.p.broadcastable
        )
        self.v_next = b2*self.v + (1 - self.b2)*self.g*self.g
        self.m_ub = self.m/(1-b1**self.t)
        self.v_ub = self.v/(1-b2**self.t)
        self.update = self.p - alpha*self.m_ub/(T.sqrt(self.v_ub) + epsilon)
        
        self.updates = [(self.t, self.t_next),
                        (self.m, self.m_next),
                        (self.v, self.v_next),
                        (self.p, self.update)]
                             
def softmax(X):
    eX = T.exp(X - X.max(axis=1, keepdims = True))
    prob = eX / eX.sum(axis=1, keepdims=True)
    return prob  
       
def Cat_sample(pi, num_sample=None):

    z = theano_rng.multinomial(n=1, pvals = pi, dtype=pi.dtype)
    
    return z
    
def CatNLL(y, pi):

    nll = - T.sum(y * T.log(T.clip(pi,1e-16,1)),axis = 0)

    return nll
    
    
def KLCatCat(pi1, pi2):
 
    kl = T.sum(pi1 * ( T.log(T.clip(pi1,1e-16,1)) - T.log(T.clip(pi2,1e-16,1)) ),axis = 0);

    return kl

def GaussianNLL(y, mu, sig):

    nll = 0.5 * T.sum(T.sqr(y - mu) / sig**2 + 2 * T.log(sig) +
                      T.log(2 * numpy.pi), axis=1)
    return nll
    
def KLGaussianStdGaussian(mu, sig):

    kl = T.sum(0.5 * (-2 * T.log(sig) + mu**2 + sig**2 - 1), axis=1)

    return kl

def KLGaussianGaussian(mu1, sig1, mu2, sig2):
   
    kl = T.sum(0.5 * (2 * T.log(sig2) - 2 * T.log(sig1) +
                   (sig1**2 + (mu1 - mu2)**2) /
                   sig2**2 - 1), axis=1)

    return kl
       
#########################################
#
# Parameters
#
#########################################

# Generate List of Parameters to Optimize
params = []

# Parameters of approximate posterior

# Q(s_t | s_t-1, o_t, oh_t, oa_t)

Wq_hst_ot = theano.shared(
    value=initweight(n_s, n_o).reshape(1,n_s,n_o),
    name='Wq_hst_ot',
    borrow=True,
    broadcastable=(True, False, False)
    
)

params.append(Wq_hst_ot)

Wq_hst_oht = theano.shared(
    value=initweight(n_s, n_oh).reshape(1,n_s,n_oh),
    name='Wq_hst_oht',
    borrow=True,
    broadcastable=(True, False, False)
)

params.append(Wq_hst_oht)

Wq_hst_oat = theano.shared(
    value=initweight(n_s, n_oa).reshape(1,n_s,n_oa),
    name='Wq_hst_oat',
    borrow=True,
    broadcastable=(True, False, False)
)

params.append(Wq_hst_oat)

Wq_hst_stm1 = theano.shared(
    value=initweight(n_s, n_s).reshape(1,n_s,n_s),
    name='Wq_hst_stm1',
    borrow=True,
    broadcastable=(True, False, False)
)

params.append(Wq_hst_stm1)

bq_hst = theano.shared(
    value=initconst(n_s, 1).reshape(1,n_s,1),
    name='bq_hst',
    borrow=True,
    broadcastable=(True,False,True)
)

params.append(bq_hst)

Wq_hst2_hst = theano.shared(
    value=initweight(n_s, n_s).reshape(1, n_s, n_s),
    name='Wq_hst2_hst',
    borrow=True,
    broadcastable=(True, False, False)
)

params.append(Wq_hst2_hst)

bq_hst2 = theano.shared(
    value=initconst(n_s, 1).reshape(1,n_s,1),
    name='bq_hst2',
    borrow=True,
    broadcastable=(True,False,True)
)

params.append(bq_hst2)

Wq_stmu_hst2 = theano.shared(
    value=initweight(n_s, n_s).reshape(1,n_s,n_s),
    name='Wq_stmu_hst2',
    borrow=True,
    broadcastable=(True, False, False)
)

params.append(Wq_stmu_hst2)

bq_stmu = theano.shared(
    value=initconst(n_s, 1).reshape(1,n_s,1),
    name='bq_stmu',
    borrow=True,
    broadcastable=(True,False,True)
)

params.append(bq_stmu)

Wq_stsig_hst2 = theano.shared(
    value=initweight(n_s, n_s).reshape(1,n_s,n_s),
    name='Wq_stsig_hst2',
    borrow=True,
    broadcastable=(True, False, False)
)

params.append(Wq_stsig_hst2)

bq_stsig = theano.shared(
    value=initconst(n_s,1,init_sig_states).reshape(1,n_s,1),
    name='bq_stsig',
    borrow=True,
    broadcastable=(True,False,True)
)

params.append(bq_stsig)

# Define Parameters for Likelihood Function

# p( s_t | s_t-1 )

Wl_stmu_stm1 = theano.shared(
    value=initweight(n_s, n_s).reshape(1,n_s,n_s),#Wq_stmu_stm1.get_value(),#initortho(n_s, n_s),
    name='Wl_stmu_stm1',
    borrow=True,
    broadcastable=(True, False, False)
)

params.append(Wl_stmu_stm1)

bl_stmu = theano.shared(
    value=initconst(n_s, 1).reshape(1,n_s,1),
    name='bl_stmu',
    borrow=True,
    broadcastable=(True,False,True)
)

params.append(bl_stmu)

Wl_stsig_stm1 = theano.shared(
    value=initweight(n_s, n_s).reshape(1,n_s,n_s),#Wq_stsig_stm1.get_value(),#initweight(n_s, n_s),
    name='Wl_stsig_stm1',
    borrow=True,
    broadcastable=(True, False, False)
)

params.append(Wl_stsig_stm1)

bl_stsig = theano.shared(
    value=initconst(n_s, 1,init_sig_states).reshape(1,n_s,1),
    name='bl_stsig',
    borrow=True,
    broadcastable=(True,False,True)
)

params.append(bl_stsig)

Wl_ost_st = theano.shared(
    value=initweight(n_s, n_s).reshape(1,n_s,n_s),
    name='Wl_ost_st',
    borrow=True,
    broadcastable=(True, False, False)
)

params.append(Wl_ost_st)

bl_ost = theano.shared(
    value=initconst(n_s, 1).reshape(1,n_s,1),
    name='bl_ost',
    borrow=True,
    broadcastable=(True,False,True)
)

params.append(bl_ost)

# p( o_t | s_t )

Wl_otmu_st = theano.shared(
    value=initweight(n_o, n_s).reshape(1,n_o,n_s),
    name='Wl_otmu_st',
    borrow=True,
    broadcastable=(True, False, False)
)

params.append(Wl_otmu_st)

bl_otmu = theano.shared(
    value=initconst(n_o, 1).reshape(1,n_o,1),
    name='bl_otmu',
    borrow=True,
    broadcastable=(True,False,True)
)

params.append(bl_otmu)

Wl_otsig_st = theano.shared(
    value=initweight(n_o, n_s).reshape(1,n_o,n_s),
    name='Wl_otsig_st',
    borrow=True,
    broadcastable=(True, False, False)
)

params.append(Wl_otsig_st)

bl_otsig = theano.shared(
    value=initconst(n_o,1,init_sig_obs).reshape(1,n_o,1),
    name='bl_otsig',
    borrow=True,
    broadcastable=(True,False,True)
)

params.append(bl_otsig)

# p( oh_t | s_t )

Wl_ohtmu_st = theano.shared(
    value=initweight(n_oh, n_s).reshape(1,n_oh,n_s),
    name='Wl_ohtmu_st',
    borrow=True,
    broadcastable=(True, False, False)
)

params.append(Wl_ohtmu_st)

bl_ohtmu = theano.shared(
    value=initconst(n_oh, 1).reshape(1,n_oh,1),
    name='bl_ohtmu',
    borrow=True,
    broadcastable=(True,False,True)
)

params.append(bl_ohtmu)

Wl_ohtsig_st = theano.shared(
    value=initweight(n_oh, n_s).reshape(1,n_oh,n_s),
    name='Wl_ohtsig_st',
    borrow=True,
    broadcastable=(True, False, False)
)

params.append(Wl_ohtsig_st)

bl_ohtsig = theano.shared(
    value=initconst(n_oh, 1,init_sig_obs).reshape(1,n_oh,1),
    name='bl_ohtsig',
    borrow=True,
    broadcastable=(True,False,True)
)

params.append(bl_ohtsig)

# p( oa_t | s_t, a_t )

Wl_oatmu_st = theano.shared(
    value=initweight(n_oa, n_s).reshape(1,n_oa,n_s),
    name='Wl_oatmu_st',
    borrow=True,
    broadcastable=(True, False, False)
)

params.append(Wl_oatmu_st)

bl_oatmu = theano.shared(
    value=initconst(n_oa, 1).reshape(1,n_oa,1),
    name='bl_oatmu',
    borrow=True,
    broadcastable=(True,False,True)
)

params.append(bl_oatmu)

Wl_oatsig_st = theano.shared(
    value=initweight(n_oa, n_s).reshape(1,n_oa,n_s),
    name='Wl_oatsig_st',
    borrow=True,
    broadcastable=(True, False, False)
)

params.append(Wl_oatsig_st)

bl_oatsig = theano.shared(
    value=initconst(n_oa, 1,init_sig_obs).reshape(1,n_oa,1),
    name='bl_oatsig',
    borrow=True,
    broadcastable=(True,False,True)
)

params.append(bl_oatsig)

####################################################################
#
# Function to Create randomly perturbed version of parameters
#
####################################################################

def randomize_parameters(params, epsilon):
    
    r_params = []
    r_epsilons = []
    
    for param in params:
        epsilon_half = theano_rng.normal((n_perturbations/2,param.shape[1],param.shape[2]), dtype = theano.config.floatX)
        r_epsilon = T.concatenate( [epsilon_half, -1.0*epsilon_half], axis = 0 )
        r_param = param + r_epsilon*epsilon
        r_params.append(r_param)
        r_epsilons.append(r_epsilon)
        
    return r_params, r_epsilons
    
####################################################################
#
# Create randomly perturbed version of parameters
#
####################################################################
    
[r_params, r_epsilons] = randomize_parameters( params, eps )

[r_Wq_hst_ot, r_Wq_hst_oht, r_Wq_hst_oat, r_Wq_hst_stm1, r_bq_hst,\
r_Wq_hst2_hst, r_bq_hst2,\
r_Wq_stmu_hst2, r_bq_stmu,\
r_Wq_stsig_hst2, r_bq_stsig,\
r_Wl_stmu_stm1, r_bl_stmu,\
r_Wl_stsig_stm1, r_bl_stsig,\
r_Wl_ost_st, r_bl_ost,\
r_Wl_otmu_st, r_bl_otmu,\
r_Wl_otsig_st, r_bl_otsig,\
r_Wl_ohtmu_st, r_bl_ohtmu,\
r_Wl_ohtsig_st, r_bl_ohtsig,\
r_Wl_oatmu_st, r_bl_oatmu,\
r_Wl_oatsig_st, r_bl_oatsig] = r_params

###################################################################
#  
# Define Variational Free Energy for Simulated Run
#
###################################################################

def inner_fn(t, stm1, oat, ot, oht, pos, vt,\
r_Wq_hst_ot, r_Wq_hst_oht, r_Wq_hst_oat, r_Wq_hst_stm1, r_bq_hst,\
r_Wq_hst2_hst, r_bq_hst2,\
r_Wq_stmu_hst2, r_bq_stmu,\
r_Wq_stsig_hst2, r_bq_stsig,\
r_Wl_stmu_stm1, r_bl_stmu,\
r_Wl_stsig_stm1, r_bl_stsig,\
r_Wl_ost_st, r_bl_ost,\
r_Wl_otmu_st, r_bl_otmu,\
r_Wl_otsig_st, r_bl_otsig,\
r_Wl_ohtmu_st, r_bl_ohtmu,\
r_Wl_ohtsig_st, r_bl_ohtsig,\
r_Wl_oatmu_st, r_bl_oatmu,\
r_Wl_oatsig_st, r_bl_oatsig\
):
   
    hst =  T.nnet.relu( T.batched_tensordot(r_Wq_hst_stm1,T.reshape(stm1,(n_perturbations,n_s,n_proc)),axes=[[2],[1]]) + T.batched_tensordot(r_Wq_hst_ot,T.reshape(ot,(n_perturbations,n_o,n_proc)),axes=[[2],[1]]) + T.batched_tensordot(r_Wq_hst_oht,T.reshape(oht,(n_perturbations,n_oh,n_proc)),axes=[[2],[1]]) + T.batched_tensordot(r_Wq_hst_oat,T.reshape(oat,(n_perturbations,n_oa,n_proc)),axes=[[2],[1]]) + r_bq_hst )
    hst2 =  T.nnet.relu( T.batched_tensordot(r_Wq_hst2_hst,T.reshape(hst,(n_perturbations,n_s,n_proc)),axes=[[2],[1]]) + r_bq_hst2 )

    stmu =  T.tanh( T.batched_tensordot(r_Wq_stmu_hst2,T.reshape(hst2,(n_perturbations,n_s,n_proc)),axes=[[2],[1]]) + r_bq_stmu )
    stsig = T.nnet.softplus( T.batched_tensordot(r_Wq_stsig_hst2,T.reshape(hst2,(n_perturbations,n_s,n_proc)),axes=[[2],[1]]) + r_bq_stsig ) + sig_min_states
    
    # Rescale representation to fit within linear response of the tanh-nonlinearity
    stmu = T.set_subtensor(stmu[:,0,:],0.1*ot[:,0,:]).reshape((n_perturbations,n_s,n_proc))
    stsig = T.set_subtensor(stsig[:,0,:],0.01).reshape((n_perturbations,n_s,n_proc))
    
    st = stmu + theano_rng.normal((n_perturbations,n_s,n_proc))*stsig
    
    ost = T.nnet.relu( T.batched_tensordot(r_Wl_ost_st,T.reshape(st,(n_perturbations,n_s,n_proc)),axes=[[2],[1]]) + r_bl_ost )
    
    otmu = T.batched_tensordot(r_Wl_otmu_st, T.reshape(ost,(n_perturbations,n_s,n_proc)),axes=[[2],[1]]) + r_bl_otmu
    otsig = T.nnet.softplus(T.batched_tensordot(r_Wl_otsig_st, T.reshape(ost,(n_perturbations,n_s,n_proc)),axes=[[2],[1]]) + r_bl_otsig) + sig_min_obs
    
    ohtmu = T.batched_tensordot(r_Wl_ohtmu_st, T.reshape(ost,(n_perturbations,n_s,n_proc)),axes=[[2],[1]]) + r_bl_ohtmu
    ohtsig = T.nnet.softplus( T.batched_tensordot(r_Wl_ohtsig_st, T.reshape(ost,(n_perturbations,n_s,n_proc)),axes=[[2],[1]]) + r_bl_ohtsig ) + sig_min_obs
    
    oatmu = T.batched_tensordot(r_Wl_oatmu_st, T.reshape(ost,(n_perturbations,n_s,n_proc)),axes=[[2],[1]]) + r_bl_oatmu
    oatsig = T.nnet.softplus( T.batched_tensordot(r_Wl_oatsig_st, T.reshape(ost,(n_perturbations,n_s,n_proc)),axes=[[2],[1]]) + r_bl_oatsig ) + sig_min_obs
    
    p_ot  = GaussianNLL(ot, otmu, otsig)
    p_oht = GaussianNLL(oht, ohtmu, ohtsig)    
    p_oat = GaussianNLL(oat, oatmu, oatsig)
    
    prior_stmu = T.tanh( T.batched_tensordot(r_Wl_stmu_stm1, T.reshape(stm1,(n_perturbations,n_s,n_proc)),axes=[[2],[1]]) + r_bl_stmu )
    prior_stsig = T.nnet.softplus( T.batched_tensordot(r_Wl_stsig_stm1, T.reshape(stm1,(n_perturbations,n_s,n_proc)),axes=[[2],[1]]) + r_bl_stsig ) + sig_min_states
    
    prior_stmu = ifelse(T.lt(t,20),prior_stmu, T.set_subtensor(prior_stmu[:,0,:],0.1))
    prior_stsig = ifelse(T.lt(t,20),prior_stsig, T.set_subtensor(prior_stsig[:,0,:],0.01))    
   
    KL_st = KLGaussianGaussian(stmu, stsig, prior_stmu, prior_stsig)
    
    FEt =  KL_st + p_ot + p_oht + p_oat
    
    oat_new = T.tanh(0.0*oat + st[:,1,:].reshape((n_perturbations,n_oa,n_proc)))
    
    action_force = oat_new
    force = (-T.pow(1+5*T.sqr(pos),-0.5)-T.sqr(pos)*T.pow(1 + 5*T.sqr(pos),-1.5)-T.pow(pos,4)/16.0)*T.nnet.sigmoid(100*pos) + (-2*pos - 1)*T.nnet.sigmoid(-100*pos) - 0.25*vt
    vt_new = vt + 0.05*force + 0.03*action_force# + 0.005*theano_rng.normal((n_o,n_proc))#0.03*T.tanh(oat_new)# + 0.005*theano_rng.normal((n_o,n_proc))
    pos_new = pos + vt_new     
    
    ot_new = pos_new + theano_rng.normal((n_perturbations,n_o,n_proc))*0.01
    
    oht_new = T.exp(-T.sqr(pos_new-1.0)/2.0/0.3/0.3)
    
    return st, oat_new, ot_new, oht_new, pos_new, vt_new, FEt, KL_st, hst, hst2, stmu, stsig, force, p_ot, p_oht, p_oat

if n_proc == 1:
    s_t0 = theano.shared(name = 's_t0', value = numpy.zeros( (n_perturbations,n_s, n_proc) ).astype( dtype = theano.config.floatX ), borrow = True, broadcastable=(False, False, True))
else:
    s_t0 = theano.shared(name = 's_t0', value = numpy.zeros( (n_perturbations,n_s, n_proc) ).astype( dtype = theano.config.floatX ), borrow = True)

a_t0 = theano.shared(name = 'a_t0', value = numpy.zeros( (n_perturbations,n_oa, n_proc) ).astype( dtype = theano.config.floatX ), borrow = True )
o_t0 = theano.shared(name = 'o_t0', value = numpy.zeros( (n_perturbations,n_o, n_proc) ).astype( dtype = theano.config.floatX ), borrow = True )
oh_t0 = theano.shared(name = 'oh_t0', value = numpy.zeros( (n_perturbations,n_oh, n_proc) ).astype( dtype = theano.config.floatX ), borrow = True )
pos_t0 = theano.shared(name = 'pos_t0', value = -0.5*numpy.ones( (n_perturbations,n_o, n_proc) ).astype( dtype = theano.config.floatX ), borrow = True )
v_t0 = theano.shared(name = 'v_t0', value = numpy.zeros( (n_perturbations,n_o, n_proc) ).astype( dtype = theano.config.floatX ), borrow = True )

   
((states_th, oat_th, ot_th, oht_th, pos_th, vt_th, FEt_th, KL_st_th, hst_th, hst2_th, stmu_th, stsig_th, force_th, p_ot_th, p_oht_th, p_oat_th), fe_updates) =\
                     theano.scan(fn=inner_fn,
                     sequences = [numpy.arange(n_run_steps).astype(dtype = theano.config.floatX)],
                     outputs_info=[s_t0, a_t0, o_t0, oh_t0, pos_t0, v_t0, None, None, None, None, None, None, None, None, None, None],
                     non_sequences = [r_Wq_hst_ot, r_Wq_hst_oht, r_Wq_hst_oat, r_Wq_hst_stm1, r_bq_hst,\
                     r_Wq_hst2_hst, r_bq_hst2,\
                     r_Wq_stmu_hst2, r_bq_stmu,\
                     r_Wq_stsig_hst2, r_bq_stsig,\
                     r_Wl_stmu_stm1, r_bl_stmu,\
                     r_Wl_stsig_stm1, r_bl_stsig,\
                     r_Wl_ost_st, r_bl_ost,\
                     r_Wl_otmu_st, r_bl_otmu,\
                     r_Wl_otsig_st, r_bl_otsig,\
                     r_Wl_ohtmu_st, r_bl_ohtmu,\
                     r_Wl_ohtsig_st, r_bl_ohtsig,\
                     r_Wl_oatmu_st, r_bl_oatmu,\
                     r_Wl_oatsig_st, r_bl_oatsig]
                     )
                     
FE_mean = FEt_th.mean()
KL_st_mean = KL_st_th.mean()
ot_mean = p_ot_th.mean()
oht_mean = p_oht_th.mean()
oat_mean = p_oat_th.mean()

FE_mean_perturbations = FEt_th.mean(axis = 0).mean(axis = 1)
FE_std_perturbations = FEt_th.mean(axis = 0).std(axis = 1)
FE_mean_perturbations_std = FE_mean_perturbations.std(axis = 0)

FE_rank = n_perturbations - T.argsort( T.argsort(FE_mean_perturbations) )

FE_rank_score = T.clip( numpy.log(0.5*n_perturbations+1) - T.log(FE_rank) , 0.0, 10000.0).astype(dtype = theano.config.floatX)

FE_rank_score_normalized = FE_rank_score/FE_rank_score.sum() - 1.0/n_perturbations

run_agent_scan = theano.function(inputs = [], outputs = [states_th, oat_th, ot_th, oht_th, FEt_th, KL_st_th, hst_th, hst2_th, stmu_th, stsig_th, force_th, pos_th], allow_input_downcast = True, on_unused_input='ignore')

#######################################################
#
# Test agent and plot some properties of
# the environment
#
#######################################################

states, actions, observations, rewards, FEs, KLs, hsts, hst2s, stmus, stsigs, forces, positions = run_agent_scan()
'''
print 'Sizes :'
print states.shape, actions.shape, observations.shape, rewards.shape#, state_noise.shape

print 'Ranges:'
print 'States:'
print states.min(), states.max()
print 'Actions:'
print actions.min(), actions.max()
print 'Observations:'
print observations.min(), observations.max()
print 'Rewards:'
print rewards.min(), rewards.max()

# Some diagnostic output!
print 'Actions: Mean: %f Std: %f Min: %f Max: %f' % (actions.mean(), actions.std(), actions.min(), actions.max())
#print 'stmus: Mean: %f Std: %f Min: %f Max: %f' % (stmus.mean(), stmus.std(), stmus.min(), stmus.max())
#print 'stsigs: Mean: %f Std: %f Min: %f Max: %f' % (stsigs.mean(), stsigs.std(), stsigs.min(), stsigs.max())
print 'Actions:'
print actions
'''
'''
plt.figure(1)
plt.subplot(1,3,1)
plt.title('Observations')
for j in range(1):
    plt.plot(observations[:,0,0,j].squeeze())
plt.subplot(1,3,2)
plt.title('Rewards')
for j in range(1):
    plt.plot(rewards[:,0,0,j].squeeze())
plt.subplot(1,3,3)
plt.title('F(x)')
for j in range(1):
    plt.plot(positions[:-1,0,0,j].squeeze(), forces[1:,0,0,j].squeeze(),"o")
plt.show()
'''

########################################################
#
# Test free energy calculation
#
########################################################

free_energy = theano.function([], [FE_mean, FEt_th, FE_mean_perturbations, r_epsilons[0], FE_std_perturbations, FE_mean_perturbations_std], allow_input_downcast = True, on_unused_input='ignore')

[free_energy_mean, free_energy_coords, free_energy_mean_perturbations, epsilons_perturbation, oFE_std_perturbations, oFE_mean_perturbations_std] = free_energy()

print 'Free Energy'
print free_energy_mean
print 'SHAPE:'
print free_energy_coords.shape
print 'SHAPE PERTURBATIONS:'
print free_energy_mean_perturbations.shape
print 'SHAPE EPSILONS_PERTURBATION'
print epsilons_perturbation.shape
print 'STD OF FE_means:'
print oFE_mean_perturbations_std
print 'INDIVIDUAL STDS from %f to %f...' % (oFE_std_perturbations.min(), oFE_std_perturbations.max())

#########################################################
#
# Define Parameter Updates
#
#########################################################

# Create List of Updates
updates = []

for i in range(len(params)):
    print 'Creating updates for parameter %d...' % i
    
    print 'Calculating derivative'
    #delta = T.tensordot(FE_mean_perturbations,r_epsilons[i],axes = [[0],[0]])/(eps*n_perturbations)
    #delta_rank = T.tensordot(FE_rank_score_normalized,r_epsilons[i],axes = [[0],[0]])/eps/n_perturbations
    delta_backprop = T.grad(cost = FE_mean, wrt = params[i])
    
    # FOR CHECKING STABILITY: USE HALF OF THE SAMPLES EACH AND COMPARE GRADIENTS
    delta_h1 = T.tensordot(FE_mean_perturbations[0::2],r_epsilons[i][0::2,:,:],axes = [[0],[0]])/(eps*0.5*n_perturbations)
    delta_h2 = T.tensordot(FE_mean_perturbations[1::2],r_epsilons[i][1::2,:,:],axes = [[0],[0]])/(eps*0.5*n_perturbations)
    
    if i == 0:
        deltas_h1 = delta_h1.flatten()
    else:
        deltas_h1 = T.concatenate([deltas_h1, delta_h1.flatten()], axis = 0 )
    
    if i == 0:
        deltas_h2 = delta_h2.flatten()
    else:
        deltas_h2 = T.concatenate([deltas_h2, delta_h2.flatten()], axis = 0 )
    
    # USE ADAM OPTIMIZER
    #p_adam = Adam(delta, params[i], 0.9, 0.999, learning_rate, epsilon = 10e-6)
    #p_adam = Adam(delta_rank, params[i], 0.9, 0.999, learning_rate, epsilon = 10e-6)
    print 'Creating Adam object'
    p_adam = Adam(delta_backprop, params[i], 0.9, 0.999, learning_rate, epsilon = 10e-6)
    updates = updates + p_adam.updates
    
    # USE SIMPLE GRADIENT DESCENT
    #update = (params[i], params[i] - learning_rate*delta)
    #update = (params[i], params[i] - learning_rate*delta_rank)
    #update = (params[i], params[i] - learning_rate*delta_backprop)
    #updates.append(update)
    
grad_corr = T.dot(deltas_h1, deltas_h2)/(deltas_h1.norm(2)*deltas_h2.norm(2))
        
# Define Training Function
train = theano.function(
        inputs=[],
        outputs=[FE_mean, FE_mean_perturbations, KL_st_mean, ot_mean, oht_mean, oat_mean, grad_corr, deltas_h1, deltas_h2, FE_std_perturbations, FE_mean_perturbations_std], 
        updates=updates,
        on_unused_input='ignore',
        allow_input_downcast = True
    )

########################################################################
#
# Run Optimization
#
########################################################################

[FE_min, oFE_mean_perturbations, oKL_st_mean, oot_mean, ooht_mean, ooat_mean, ograd_corr, odeltas_h1, odeltas_h2, oFE_std_perturbations, oFE_mean_perturbations_std] = train()

print 'Initial FEs:'
print [FE_min, oKL_st_mean, oot_mean, ooht_mean, ooat_mean]

numpy.savetxt('initial_deltas_h1.txt',odeltas_h1)
numpy.savetxt('initial_deltas_h2.txt',odeltas_h2)

# Optimization Loop
for i in range(n_steps):
    
    #print 'Constraint weight:'
    #print constraint_weight.get_value()
    
    # Take the time for each loop
    start_time = timeit.default_timer()
    
    print 'Iteration: %d' % i    
    
    # Perform stochastic gradient descent using ADAM updates
    print 'Descending on Free Energy...'    
    [oFE_mean, oFE_mean_perturbations, oKL_st_mean, oot_mean, ooht_mean, ooat_mean, ograd_corr, odeltas_h1, odeltas_h2, oFE_std_perturbations, oFE_mean_perturbations_std] = train()
    
    print 'Free Energies:'
    print [oFE_mean, oKL_st_mean, oot_mean, ooht_mean, ooat_mean]
       
    print 'Correlation between gradients: %f' % ograd_corr   
    print 'Norms of Gradients: %f vs. %f' % (numpy.linalg.norm(odeltas_h1), numpy.linalg.norm(odeltas_h2))
    
    print 'STD OF FE_means:'
    print oFE_mean_perturbations_std
    print 'INDIVIDUAL STDS from %f to %f...' % (oFE_std_perturbations.min(), oFE_std_perturbations.max())
       
    if i == 0:
        with open("log_evAI_mountaincar_backprop.txt", "w") as myfile:
            myfile.write("%f %f %f\n" % (oFE_mean, FE_min, ograd_corr))
    else:
        with open("log_evAI_mountaincar_backprop.txt", "a") as myfile:
            myfile.write("%f %f %f\n" % (oFE_mean, FE_min, ograd_corr))
    
    # Stop time
    end_time = timeit.default_timer()
    print 'Time for iteration: %f' % (end_time - start_time)
    
    # Save current parameters every nth loop
    if i % 100 == 0:
        with open('evAI_mountaincar_backprop.pkl', 'w') as f:
            cPickle.dump(params, f)
            
    # Save best parameters
    if oFE_mean < FE_min:
        FE_min = oFE_mean
        with open('evAI_mountaincar_backprop_best.pkl', 'w') as f:
            cPickle.dump(params, f)
        

# Save final parameters
with open('evAI_mountaincar_backprop.pkl', 'w') as f:
    cPickle.dump(params, f)


