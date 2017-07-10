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
base_name = 'deepAI_demo' # Name for Saves and Logfile
learning_rate = 1e-3 # Learning Rate

n_run_steps = 30 # No. of Timesteps to Simulate
n_proc = 1 # No. of Processes to Simulate for each Sample from the Population Density
n_perturbations = 100000 # No. of Samples from Population Density per Iteration

n_o = 1 # Sensory Input encoding Position
n_oh = 1 # Nonlinearly Transformed Channel (OPTIONAL!)
n_oa = 1 # Proprioception (OPTIONAL!)

# Minimum Value of Standard-Deviations, to prevent Division-by-Zero
sig_min_obs = 1e-6
sig_min_states = 1e-6
sig_min_perturbations = 1e-6
sig_min_action = 1e-6

init_sig_obs = -3.0
init_sig_states = -3.0
init_sig_perturbations = -3.0
init_sig_action = -3.0

# Max. Number of Optimization Steps
n_steps = 1000000

# Initialize RNG
ii32 = numpy.iinfo(numpy.int32)
theano_rng = RandomStreams(numpy.random.randint(ii32.max)) # ADD RANDOM SEED!  

# Helper Functions and Classes

# Uniform Weight Distribution
def initweight(shape1, shape2, minval =-0.05, maxval = 0.05):
    val = numpy.random.rand(
        shape1, shape2
    )
    
    val = minval + (maxval - minval)*val    
    
    return val.astype(theano.config.floatX) 
    
# Constant Weight
def initconst(shape1, shape2, val = 0.0):
    val = val*numpy.ones(
        (shape1,shape2),
        dtype=theano.config.floatX
    )
    
    return val.astype(theano.config.floatX)   

#ADAM Optimizer, following Kingma & Ba (2015), c.f. https://arxiv.org/abs/1412.6980
class Adam(object):

    def __init__(self, grads, p, b1, b2, alpha, epsilon = 10e-8):
    
        #self.L = L
        self.p = p
        self.b1 = b1
        self.b2 = b2
        self.alpha = alpha
        
        self.t = theano.shared( value = numpy.cast[theano.config.floatX](1.0))
        self.t_next = self.t + 1
        
        self.g = grads.astype(dtype = theano.config.floatX)
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
                             
def GaussianNLL(y, mu, sig):

    nll = 0.5 * T.sum(T.sqr(y - mu) / sig**2 + 2 * T.log(sig) +
                      T.log(2 * numpy.pi), axis=1)
    return nll
    
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

Wl_ost2_ost = theano.shared(
    value=initweight(n_s, n_s).reshape(1,n_s,n_s),
    name='Wl_ost2_ost',
    borrow=True,
    broadcastable=(True, False, False)
)

params.append(Wl_ost2_ost)

bl_ost2 = theano.shared(
    value=initconst(n_s, 1).reshape(1,n_s,1),
    name='bl_ost2',
    borrow=True,
    broadcastable=(True,False,True)
)

params.append(bl_ost2)

Wl_ost3_ost2 = theano.shared(
    value=initweight(n_s, n_s).reshape(1,n_s,n_s),
    name='Wl_ost3_ost2',
    borrow=True,
    broadcastable=(True, False, False)
)

params.append(Wl_ost3_ost2)

bl_ost3 = theano.shared(
    value=initconst(n_s, 1).reshape(1,n_s,1),
    name='bl_ost3',
    borrow=True,
    broadcastable=(True,False,True)
)

params.append(bl_ost3)

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

# Action Function

Wa_atmu_st = theano.shared(
    value=initweight(n_oa, n_s).reshape(1,n_oa,n_s),
    name='Wa_atmu_st',
    borrow=True,
    broadcastable=(True, False, False)
)

params.append(Wa_atmu_st)

ba_atmu = theano.shared(
    value=initconst(n_oa, 1).reshape(1,n_oa,1),
    name='ba_atmu',
    borrow=True,
    broadcastable=(True,False,True)
)

params.append(ba_atmu)

Wa_atsig_st = theano.shared(
    value=initweight(n_oa, n_s).reshape(1,n_oa,n_s),
    name='Wa_atsig_st',
    borrow=True,
    broadcastable=(True, False, False)
)

params.append(Wa_atsig_st)

ba_atsig = theano.shared(
    value=initconst(n_oa, 1,init_sig_action).reshape(1,n_oa,1),
    name='ba_atsig',
    borrow=True,
    broadcastable=(True,False,True)
)

params.append(ba_atsig)

####################################################################
#
# Function to Create randomly perturbed version of parameters
#
####################################################################

def initialize_sigmas(params, init_sig_perturbations):
    
    sigmas = []
    
    for param in params:
        sigma = theano.shared(name = 'sigma_' + param.name, value = init_sig_perturbations*numpy.ones( param.get_value().shape ).astype( dtype = theano.config.floatX ), borrow = True, broadcastable=param.broadcastable )
        sigmas.append(sigma)
        
    return sigmas
    
def randomize_parameters(params, sigmas, sig_min_perturbations):
    
    r_params = []
    r_epsilons = []
    
    for i in range(len(params)):
        epsilon_half = theano_rng.normal((n_perturbations/2,params[i].shape[1],params[i].shape[2]), dtype = theano.config.floatX)
        r_epsilon = T.concatenate( [epsilon_half, -1.0*epsilon_half], axis = 0 )
        r_param = params[i] + r_epsilon*(T.nnet.softplus( sigmas[i] ) + sig_min_perturbations)
        r_params.append(r_param)
        r_epsilons.append(r_epsilon)
        
    return r_params, r_epsilons
    
####################################################################
#
# Create randomly perturbed version of parameters
#
####################################################################
    
sigmas = initialize_sigmas(params, init_sig_perturbations)
    
[r_params, r_epsilons] = randomize_parameters( params, sigmas, sig_min_perturbations )

[r_Wq_hst_ot, r_Wq_hst_oht, r_Wq_hst_oat, r_Wq_hst_stm1, r_bq_hst,\
r_Wq_hst2_hst, r_bq_hst2,\
r_Wq_stmu_hst2, r_bq_stmu,\
r_Wq_stsig_hst2, r_bq_stsig,\
r_Wl_stmu_stm1, r_bl_stmu,\
r_Wl_stsig_stm1, r_bl_stsig,\
r_Wl_ost_st, r_bl_ost,\
r_Wl_ost2_ost, r_bl_ost2,\
r_Wl_ost3_ost2, r_bl_ost3,\
r_Wl_otmu_st, r_bl_otmu,\
r_Wl_otsig_st, r_bl_otsig,\
r_Wl_ohtmu_st, r_bl_ohtmu,\
r_Wl_ohtsig_st, r_bl_ohtsig,\
r_Wl_oatmu_st, r_bl_oatmu,\
r_Wl_oatsig_st, r_bl_oatsig,\
r_Wa_atmu_st, r_ba_atmu,\
r_Wa_atsig_st, r_ba_atsig] = r_params

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
r_Wl_ost2_ost, r_bl_ost2,\
r_Wl_ost3_ost2, r_bl_ost3,\
r_Wl_otmu_st, r_bl_otmu,\
r_Wl_otsig_st, r_bl_otsig,\
r_Wl_ohtmu_st, r_bl_ohtmu,\
r_Wl_ohtsig_st, r_bl_ohtsig,\
r_Wl_oatmu_st, r_bl_oatmu,\
r_Wl_oatsig_st, r_bl_oatsig,\
r_Wa_atmu_st, r_ba_atmu,\
r_Wa_atsig_st, r_ba_atsig\
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
    ost2 = T.nnet.relu( T.batched_tensordot(r_Wl_ost2_ost,T.reshape(ost,(n_perturbations,n_s,n_proc)),axes=[[2],[1]]) + r_bl_ost2 )
    ost3 = T.nnet.relu( T.batched_tensordot(r_Wl_ost3_ost2,T.reshape(ost2,(n_perturbations,n_s,n_proc)),axes=[[2],[1]]) + r_bl_ost3 )
    
    otmu = T.batched_tensordot(r_Wl_otmu_st, T.reshape(ost3,(n_perturbations,n_s,n_proc)),axes=[[2],[1]]) + r_bl_otmu
    otsig = T.nnet.softplus(T.batched_tensordot(r_Wl_otsig_st, T.reshape(ost3,(n_perturbations,n_s,n_proc)),axes=[[2],[1]]) + r_bl_otsig) + sig_min_obs
    
    ohtmu = T.batched_tensordot(r_Wl_ohtmu_st, T.reshape(ost3,(n_perturbations,n_s,n_proc)),axes=[[2],[1]]) + r_bl_ohtmu
    ohtsig = T.nnet.softplus( T.batched_tensordot(r_Wl_ohtsig_st, T.reshape(ost3,(n_perturbations,n_s,n_proc)),axes=[[2],[1]]) + r_bl_ohtsig ) + sig_min_obs
    
    oatmu = T.batched_tensordot(r_Wl_oatmu_st, T.reshape(ost3,(n_perturbations,n_s,n_proc)),axes=[[2],[1]]) + r_bl_oatmu
    oatsig = T.nnet.softplus( T.batched_tensordot(r_Wl_oatsig_st, T.reshape(ost3,(n_perturbations,n_s,n_proc)),axes=[[2],[1]]) + r_bl_oatsig ) + sig_min_obs
    
    p_ot  = GaussianNLL(ot, otmu, otsig)
    p_oht = GaussianNLL(oht, ohtmu, ohtsig)    
    p_oat = GaussianNLL(oat, oatmu, oatsig)
    
    prior_stmu = T.tanh( T.batched_tensordot(r_Wl_stmu_stm1, T.reshape(stm1,(n_perturbations,n_s,n_proc)),axes=[[2],[1]]) + r_bl_stmu )
    prior_stsig = T.nnet.softplus( T.batched_tensordot(r_Wl_stsig_stm1, T.reshape(stm1,(n_perturbations,n_s,n_proc)),axes=[[2],[1]]) + r_bl_stsig ) + sig_min_states
    
    prior_stmu = ifelse(T.lt(t,20),prior_stmu, T.set_subtensor(prior_stmu[:,0,:],0.1))
    prior_stsig = ifelse(T.lt(t,20),prior_stsig, T.set_subtensor(prior_stsig[:,0,:],0.01))    
   
    KL_st = KLGaussianGaussian(stmu, stsig, prior_stmu, prior_stsig)
    
    FEt =  KL_st + p_ot + p_oht + p_oat
    
    oat_mu = T.batched_tensordot(r_Wa_atmu_st, T.reshape(st,(n_perturbations,n_s,n_proc)),axes=[[2],[1]]) + ba_atmu
    oat_sig = T.nnet.softplus( T.batched_tensordot(r_Wa_atsig_st, T.reshape(st,(n_perturbations,n_s,n_proc)),axes=[[2],[1]]) + ba_atsig ) + sig_min_action
    
    oat_new = 0.0*oat + oat_mu + theano_rng.normal((n_perturbations,n_oa,n_proc))*oat_sig
    
    action_force = T.tanh( oat_new )
    force = T.switch(T.lt(pos,0.0),-2*pos - 1,-T.pow(1+5*T.sqr(pos),-0.5)-T.sqr(pos)*T.pow(1 + 5*T.sqr(pos),-1.5)-T.pow(pos,4)/16.0) - 0.25*vt
    vt_new = vt + 0.05*force + 0.03*action_force
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
                     r_Wl_ost2_ost, r_bl_ost2,\
                     r_Wl_ost3_ost2, r_bl_ost3,\
                     r_Wl_otmu_st, r_bl_otmu,\
                     r_Wl_otsig_st, r_bl_otsig,\
                     r_Wl_ohtmu_st, r_bl_ohtmu,\
                     r_Wl_ohtsig_st, r_bl_ohtsig,\
                     r_Wl_oatmu_st, r_bl_oatmu,\
                     r_Wl_oatsig_st, r_bl_oatsig,\
                     r_Wa_atmu_st, r_ba_atmu,\
                     r_Wa_atsig_st, r_ba_atsig]
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
    normalization = T.nnet.softplus( sigmas[i] ) + sig_min_perturbations
    delta = T.tensordot(FE_mean_perturbations,r_epsilons[i],axes = [[0],[0]])/normalization/n_perturbations
    #delta_rank = T.tensordot(FE_rank_score_normalized,r_epsilons[i],axes = [[0],[0]])/eps/n_perturbations
    #delta_backprop = T.grad(cost = FE_mean, wrt = params[i])
    
    # FOR CHECKING STABILITY: USE HALF OF THE SAMPLES EACH AND COMPARE GRADIENTS
    delta_h1 = T.tensordot(FE_mean_perturbations[0::2],r_epsilons[i][0::2,:,:],axes = [[0],[0]])/normalization/(0.5*n_perturbations)
    delta_h2 = T.tensordot(FE_mean_perturbations[1::2],r_epsilons[i][1::2,:,:],axes = [[0],[0]])/normalization/(0.5*n_perturbations)
    
    if i == 0:
        deltas_h1 = delta_h1.flatten()
    else:
        deltas_h1 = T.concatenate([deltas_h1, delta_h1.flatten()], axis = 0 )
    
    if i == 0:
        deltas_h2 = delta_h2.flatten()
    else:
        deltas_h2 = T.concatenate([deltas_h2, delta_h2.flatten()], axis = 0 )
    
    # USE ADAM OPTIMIZER
    p_adam = Adam(delta, params[i], 0.9, 0.999, learning_rate, epsilon = 10e-6)
    #p_adam = Adam(delta_rank, params[i], 0.9, 0.999, learning_rate, epsilon = 10e-6)
    #p_adam = Adam(delta_backprop, params[i], 0.9, 0.999, learning_rate, epsilon = 10e-6)
    updates = updates + p_adam.updates
    
    # USE SIMPLE GRADIENT DESCENT
    #update = (params[i], params[i] - learning_rate*delta)
    #update = (params[i], params[i] - learning_rate*delta_rank)
    #update = (params[i], params[i] - learning_rate*delta_backprop)
    #updates.append(update)
    
grad_corr = T.dot(deltas_h1, deltas_h2)/(deltas_h1.norm(2)*deltas_h2.norm(2))
   
for i in range(len(sigmas)):
    
    print 'Creating updates for std dev of parameter %d...' % i
    
    print 'Calculating derivative'
    normalization = T.nnet.softplus( sigmas[i] ) + sig_min_perturbations
    outer_der = (r_epsilons[i]*r_epsilons[i]-1.0)/normalization
    inner_der = T.exp(sigmas[i])/(1.0 + T.exp(sigmas[i]))
    delta_sigma = T.tensordot(FE_mean_perturbations,outer_der*inner_der,axes = [[0],[0]])/n_perturbations
 
    delta_h1_sigma = T.tensordot(FE_mean_perturbations[0::2],outer_der[0::2,:,:]*inner_der,axes = [[0],[0]])/(0.5*n_perturbations)
    delta_h2_sigma = T.tensordot(FE_mean_perturbations[1::2],outer_der[1::2,:,:]*inner_der,axes = [[0],[0]])/(0.5*n_perturbations)
    
    if i == 0:
        deltas_h1_sigma = delta_h1_sigma.flatten()
    else:
        deltas_h1_sigma = T.concatenate([deltas_h1_sigma, delta_h1_sigma.flatten()], axis = 0 )
    
    if i == 0:
        deltas_h2_sigma = delta_h2_sigma.flatten()
    else:
        deltas_h2_sigma = T.concatenate([deltas_h2_sigma, delta_h2_sigma.flatten()], axis = 0 )

    # USE ADAM OPTIMIZER
    p_adam = Adam(delta_sigma, sigmas[i], 0.9, 0.999, learning_rate, epsilon = 10e-6)
    #p_adam = Adam(delta_rank, params[i], 0.9, 0.999, learning_rate, epsilon = 10e-6)
    #p_adam = Adam(delta_backprop, sigmas[i], 0.9, 0.999, learning_rate, epsilon = 10e-6)
    updates = updates + p_adam.updates
    
    # USE SIMPLE GRADIENT DESCENT
    #update = (params[i], params[i] - learning_rate*delta)
    #update = (params[i], params[i] - learning_rate*delta_rank)
    #update = (params[i], params[i] - learning_rate*delta_backprop)
    #updates.append(update)
    
grad_corr_sigma = T.dot(deltas_h1_sigma, deltas_h2_sigma)/(deltas_h1_sigma.norm(2)*deltas_h2_sigma.norm(2))
   
# Define Training Function
train = theano.function(
        inputs=[],
        outputs=[FE_mean, FE_mean_perturbations, KL_st_mean, ot_mean, oht_mean, oat_mean, grad_corr, grad_corr_sigma, deltas_h1, deltas_h2, deltas_h1_sigma, deltas_h2_sigma, FE_std_perturbations, FE_mean_perturbations_std], 
        updates=updates,
        on_unused_input='ignore',
        allow_input_downcast = True
    )

########################################################################
#
# Run Optimization
#
########################################################################

[FE_min, oFE_mean_perturbations, oKL_st_mean, oot_mean, ooht_mean, ooat_mean, ograd_corr, ograd_corr_sigma, odeltas_h1, odeltas_h2, odeltas_h1_sigma, odeltas_h2_sigma, oFE_std_perturbations, oFE_mean_perturbations_std] = train()

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
    [oFE_mean, oFE_mean_perturbations, oKL_st_mean, oot_mean, ooht_mean, ooat_mean, ograd_corr, ograd_corr_sigma, odeltas_h1, odeltas_h2, odeltas_h1_sigma, odeltas_h2_sigma, oFE_std_perturbations, oFE_mean_perturbations_std] = train()
    
    print 'Free Energies:'
    print [oFE_mean, oKL_st_mean, oot_mean, ooht_mean, ooat_mean]
       
    print 'Correlation between gradients: %f' % ograd_corr   
    print 'Norms of Gradients: %f vs. %f' % (numpy.linalg.norm(odeltas_h1), numpy.linalg.norm(odeltas_h2))
    
    print 'Correlation between gradients for std devs: %f' % ograd_corr_sigma 
    print 'Norms of Gradients: %f vs. %f' % (numpy.linalg.norm(odeltas_h1_sigma), numpy.linalg.norm(odeltas_h2_sigma))
    
    print 'STD OF FE_means:'
    print oFE_mean_perturbations_std
    print 'INDIVIDUAL STDS from %f to %f...' % (oFE_std_perturbations.min(), oFE_std_perturbations.max())
       
    if i == 0:
        with open('log_' + base_name + '.txt', "w") as myfile:
            myfile.write("%f %f %f\n" % (oFE_mean, FE_min, ograd_corr))
    else:
        with open('log_' + base_name + '.txt', "a") as myfile:
            myfile.write("%f %f %f\n" % (oFE_mean, FE_min, ograd_corr))
    
    # Stop time
    end_time = timeit.default_timer()
    print 'Time for iteration: %f' % (end_time - start_time)
    
    # Save current parameters every nth loop
    if i % 100 == 0:
        with open(base_name + '_%d.pkl' % i, 'w') as f:
            cPickle.dump([params,sigmas], f)
        with open(base_name + '_current.pkl', 'w') as f:
            cPickle.dump([params,sigmas], f)
            
    # Save best parameters
    if oFE_mean < FE_min:
        FE_min = oFE_mean
        with open(base_name + '_best.pkl', 'w') as f:
            cPickle.dump([params,sigmas], f)
        

# Save final parameters
with open(base_name + '_final.pkl', 'w') as f:
    cPickle.dump([params,sigmas], f)


