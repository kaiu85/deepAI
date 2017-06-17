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

from theano import pprint as pp

from theano.sandbox.rng_mrg import MRG_RandomStreams as RandomStreams  

# Parameters
n_s = 100 # States

n_o = 1 # Sensory Input
n_oh = 1 # Rewards (OPTIONAL!)
n_oa = 1 # Proprioception (OPTIONAL!)

n_output_states = 1 # How many state variables can have influences on the environment

n_run_steps = 100
n_proc = 1000

n_steps = 1000000

sig_min_obs = 0.01
sig_min_states = 1e-6
sig_min_action = 1e-6

n_sample_steps = 100
n_samples = 1000

learning_rate = 0.0001

init_sig_obs = -3.0
init_sig_states = -3.0
init_sig_action = 1.0


# Initialize RNG
ii32 = numpy.iinfo(numpy.int32)
theano_rng = RandomStreams(numpy.random.randint(ii32.max)) # ADD RANDOM SEED!  

# Helper Functions and Classes

def initweight(shape1, shape2, minval =-0.1, maxval = 0.1):
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

    def __init__(self, L, p, b1, b2, alpha, epsilon = 10e-8):
    
        self.L = L
        self.p = p
        self.b1 = b1
        self.b2 = b2
        self.alpha = alpha
        
        self.t = theano.shared( value = numpy.cast[theano.config.floatX](1.0))
        self.t_next = self.t + 1
        
        #self.g = T.grad(cost = theano.gradient.grad_clip(L,-1.0,1.0), wrt = p)
        self.g = T.grad(cost = L, wrt = p)
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
                        
#ADAM Optimizer, following Kingma & Ba (2015)
class NegAdam(object):

    def __init__(self, L, p, b1, b2, alpha, epsilon = 10e-8):
    
        self.L = L
        self.p = p
        self.b1 = b1
        self.b2 = b2
        self.alpha = alpha
        
        self.t = theano.shared( value = numpy.cast[theano.config.floatX](1.0))
        self.t_next = self.t + 1
        
        #self.g = T.grad(cost = theano.gradient.grad_clip(L,-1.0,1.0), wrt = p)
        self.g = T.grad(cost = L, wrt = p)
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
        self.update = self.p + alpha*self.m_ub/(T.sqrt(self.v_ub) + epsilon)
        
        self.updates = [(self.t, self.t_next),
                        (self.m, self.m_next),
                        (self.v, self.v_next),
                        (self.p, self.update)]
                        
def softmax(X):
    eX = T.exp(X - X.max(axis=0, keepdims = True))
    prob = eX / eX.sum(axis=0, keepdims=True)
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
                      T.log(2 * numpy.pi), axis=0)
    return nll
    
def KLGaussianStdGaussian(mu, sig):

    kl = T.sum(0.5 * (-2 * T.log(sig) + mu**2 + sig**2 - 1), axis=0)

    return kl

def KLGaussianGaussian(mu1, sig1, mu2, sig2):
   
    kl = T.sum(0.5 * (2 * T.log(sig2) - 2 * T.log(sig1) +
                   (sig1**2 + (mu1 - mu2)**2) /
                   sig2**2 - 1), axis=0)

    return kl
       
#########################################
#
# Parameters
#
#########################################

# Load learned Agent
with open('fullAI_mountaincar_passive4_best.pkl', 'r') as f:#open('fullAI_mountaincar_state_prior_scaled_explicit_prior8passive2_best.pkl', 'r') as f:
    params = cPickle.load(f)

[Wq_hst_ot, Wq_hst_oht, Wq_hst_oat, Wq_hst_stm1, bq_hst,\
Wq_hst2_hst, bq_hst2,\
Wq_stmu_hst2, bq_stmu,\
Wq_stsig_hst2, bq_stsig,\
Wl_stmu_stm1, bl_stmu,\
Wl_stsig_stm1, bl_stsig,\
Wl_ost_st, bl_ost,\
Wl_otmu_st, bl_otmu,\
Wl_otsig_st, bl_otsig,\
Wl_ohtmu_st, bl_ohtmu,\
Wl_ohtsig_st, bl_ohtsig,\
Wl_oatmu_st, bl_oatmu,\
Wl_oatsig_st, bl_oatsig,\
] = params

###################################################################
#  
# Define Variational Free Energy for Simulated Run
#
###################################################################

###################################################################
#  
# Define Variational Free Energy for Simulated Run
#
###################################################################

def inner_fn(t, stm1, oat, ot, oht, pos, vt):
   
    hst =  T.tanh( T.dot(Wq_hst_stm1,stm1) + T.dot(Wq_hst_ot,ot) + T.dot(Wq_hst_oht,oht) + T.dot(Wq_hst_oat,oat) + bq_hst )
    hst2 =  T.tanh( T.dot(Wq_hst2_hst,hst) + bq_hst2 )

    stmu =  T.tanh( T.dot(Wq_stmu_hst2,hst2) + bq_stmu )
    stsig = T.nnet.softplus( T.dot(Wq_stsig_hst2,hst2) + bq_stsig ) + sig_min_states
    
    # Rescale representation to fit within linear response of the tanh-nonlinearity
    #stmu = T.set_subtensor(stmu[0,:],stm1[0,:] + 0.01*oht[0,:])
    #stsig = T.set_subtensor(stsig[0,:],1e-5)
    
    st = stmu + theano_rng.normal((n_s,n_proc))*stsig
    
    ost = T.tanh( T.dot(Wl_ost_st,st) + bl_ost )
    
    otmu = T.dot(Wl_otmu_st, ost) + bl_otmu
    otsig = T.nnet.softplus(T.dot(Wl_otsig_st, ost) + bl_otsig) + sig_min_obs
    
    ohtmu = T.dot(Wl_ohtmu_st, ost) + bl_ohtmu
    ohtsig = T.nnet.softplus( T.dot(Wl_ohtsig_st, ost) + bl_ohtsig ) + sig_min_obs
    
    oatmu = T.dot(Wl_oatmu_st, ost) + bl_oatmu
    oatsig = T.nnet.softplus( T.dot(Wl_oatsig_st, ost) + bl_oatsig ) + sig_min_obs
    
    p_ot  = GaussianNLL(ot, otmu, otsig)
    p_oht = GaussianNLL(oht, ohtmu, ohtsig)    
    p_oat = GaussianNLL(oat, oatmu, oatsig)
    
    prior_stmu = T.tanh( T.dot(Wl_stmu_stm1, stm1) + bl_stmu )
    prior_stsig = T.nnet.softplus( T.dot(Wl_stsig_stm1, stm1) + bl_stsig ) + sig_min_states
    
    KL_st = KLGaussianGaussian(stmu, stsig, prior_stmu, prior_stsig)
    
    FEt =  KL_st + p_ot + p_oht + p_oat    
    
    oat_new = 0.99*oat + 0.5*theano_rng.normal((n_oa,n_proc))
    
    force = -0.1*pos - 0.05*vt 
    vt_new = vt + force/0.5 + 0.5*T.tanh(oat_new)
    pos_new = pos + vt_new     
      
    ot_new = pos_new + theano_rng.normal((n_o,n_proc))*0.01
    
    oht_new = vt_new
    
    return st, oat_new, ot_new, oht_new, pos_new, vt_new, FEt, KL_st, hst, hst2, stmu, stsig, force, p_ot, p_oht, p_oat, prior_stmu, prior_stsig
    
s_t0 = theano.shared(name = 's_t0', value = numpy.zeros( (n_s, n_proc) ).astype( dtype = theano.config.floatX ), borrow = True )
a_t0 = theano.shared(name = 'a_t0', value = numpy.zeros( (n_oa, n_proc) ).astype( dtype = theano.config.floatX ), borrow = True )
o_t0 = theano.shared(name = 'o_t0', value = numpy.zeros( (n_o, n_proc) ).astype( dtype = theano.config.floatX ), borrow = True )
oh_t0 = theano.shared(name = 'oh_t0', value = numpy.zeros( (n_oh, n_proc) ).astype( dtype = theano.config.floatX ), borrow = True )
pos_t0 = theano.shared(name = 'pos_t0', value = 5.0*numpy.random.randn( n_o, n_proc ).astype( dtype = theano.config.floatX ), borrow = True )
v_t0 = theano.shared(name = 'v_t0', value = 0.5*numpy.random.randn( n_o, n_proc ).astype( dtype = theano.config.floatX ), borrow = True )

observations_th = T.tensor3('observations_th')
rewards_th = T.tensor3('rewards_th')
action_observations_th = T.tensor3('actions_in_th')
state_noise_th = T.tensor3('state_noise_th')
action_noise_th = T.tensor3('action_noise_th')
    
((states_th, oat_th, ot_th, oht_th, post_th, vt_th, FEt_th, KL_st_th, hst_th, hst2_th, stmu_th, stsig_th, aht_th, p_ot_th, p_oht_th, p_oat_th, prior_stmu_th, prior_stsig_th), fe_updates) =\
                     theano.scan(fn=inner_fn,
                     sequences = [theano.shared(numpy.arange(n_run_steps).astype(dtype = theano.config.floatX))],
                     outputs_info=[s_t0, a_t0, o_t0, oh_t0, pos_t0, v_t0, None, None, None, None, None, None, None, None, None, None, None, None])
                     
last_stmus = stmu_th[-1,0,:].squeeze().dimshuffle('x',0)
last_stsigs = stsig_th[-1,0,:].squeeze().dimshuffle('x',0)

fixed_prior_cost = GaussianNLL(last_stmus, 0.3, 0.01)
                     
FE_mean = FEt_th.mean()# + fixed_prior_cost.mean()
KL_st_mean = KL_st_th.mean()
p_ot_mean = p_ot_th.mean()
p_oht_mean = p_oht_th.mean()
p_oat_mean = p_oat_th.mean()

run_agent_scan = theano.function(inputs = [], outputs = [states_th, oat_th, ot_th, oht_th, FEt_th, KL_st_th, hst_th, hst2_th, stmu_th, stsig_th, aht_th], allow_input_downcast = True, on_unused_input='ignore')

#######################################################
#
# Test agent and plot some properties of
# the environment
#
#######################################################

states, actions, observations, rewards, FEs, KLs, hsts, hst2s, stmus, stsigs, ahts = run_agent_scan()

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

print 'hsts:'
print hsts

print 'hst2s:'
print hst2s

print 'stmus:'
print stmus

print 'stsigs:'
print stsigs

print 'ahts:'
print ahts

print 'sts:'
print states

plt.figure(1)
plt.subplot(3,4,5)
plt.title('True trajectories')
for j in range(3):
    plt.plot(observations[:,0,j].squeeze())
plt.subplot(3,4,6)
plt.title('True rewards')
for j in range(3):
    plt.plot(rewards[:,0,j].squeeze())
plt.subplot(3,4,7)
plt.title('True actions')
for j in range(3):
    plt.plot(actions[:,0,j].squeeze())
plt.subplot(3,4,8)
plt.title('True states')
for j in range(10):
    plt.plot(states[:,0,j].squeeze())



########################################################
#
# Test free energy calculation
#
########################################################

free_energy = theano.function([], [FE_mean], allow_input_downcast = True, on_unused_input='ignore')

free_energy_sum = free_energy()

print 'Free Energy'
[oFE_mean] = free_energy_sum
print oFE_mean

#############################################################
#
# Define Sampling Function for Simulated Run
#
#############################################################

def inner_fn_sample(stm1):
    
    prior_stmu = T.tanh( T.dot(Wl_stmu_stm1, stm1) + bl_stmu )
    prior_stsig = T.nnet.softplus( T.dot(Wl_stsig_stm1, stm1) + bl_stsig ) + sig_min_states
    
    st = prior_stmu + theano_rng.normal((n_s,n_samples))*prior_stsig
    
    ost = T.tanh( T.dot(Wl_ost_st,st) + bl_ost )    
    
    otmu = T.dot(Wl_otmu_st, ost) + bl_otmu
    otsig = T.nnet.softplus(T.dot(Wl_otsig_st, ost) + bl_otsig) + sig_min_obs
    
    ohtmu = T.dot(Wl_ohtmu_st, ost) + bl_ohtmu
    ohtsig = T.nnet.softplus( T.dot(Wl_ohtsig_st, ost) + bl_ohtsig ) + sig_min_obs
    
    oatmu = T.dot(Wl_oatmu_st, ost) + bl_oatmu
    oatsig = T.nnet.softplus( T.dot(Wl_oatsig_st, ost) + bl_oatsig ) + sig_min_obs
    
    ot = otmu + theano_rng.normal((n_o,n_samples))*otsig
    oht = ohtmu + theano_rng.normal((n_oh,n_samples))*ohtsig   
    oat = oatmu + theano_rng.normal((n_oa,n_samples))*oatsig   
    
    return st, ohtmu, ohtsig, ot, oht, oat, prior_stmu, prior_stsig
    
# Define initial state and action
    
s_t0_sample = theano.shared(name = 's_t0', value = numpy.zeros( (n_s,n_samples) ).astype( dtype = theano.config.floatX ), borrow = True )
    

    
((states_sampled, reward_probabilities_mu_sampled, reward_probabilities_sig_sampled, observations_sampled, rewards_sampled, actions_observations_sampled, stmus_sampled, stsigs_sampled), updates_sampling) =\
                     theano.scan(fn=inner_fn_sample,
                     outputs_info=[s_t0_sample, None, None, None, None, None, None, None],
                     #sequences = [theano.shared(target_mu.reshape((n_run_steps,1)).astype(dtype = theano.config.floatX)), theano.shared(target_sig.reshape((n_run_steps,1)).astype(dtype = theano.config.floatX))],
                     n_steps = n_sample_steps)                     

########################################################
#
# Test Calculation of Combined Objective
#
########################################################

# Build Function
eval_Penalized_FE = theano.function([], [FE_mean, observations_sampled, rewards_sampled, actions_observations_sampled, states_sampled], allow_input_downcast = True, on_unused_input='ignore')

# Evaluate Function
print 'Penalized FE'
results_eval =  eval_Penalized_FE()
print results_eval

FE_min = results_eval[0]

observations = results_eval[1]
rewards = results_eval[2]
actions = results_eval[3]
states = results_eval[4]

print 'FE_min:'
print FE_min

plt.subplot(3,4,1)
plt.title('Sampled trajectories')
for j in range(3):
    plt.plot(observations[:,0,j].squeeze())
plt.subplot(3,4,2)
plt.title('Sampled rewards')
for j in range(3):
    plt.plot(rewards[:,0,j].squeeze())
plt.subplot(3,4,3)
plt.title('Sampled actions')
for j in range(3):
    plt.plot(actions[:,0,j].squeeze())
plt.subplot(3,4,4)
plt.title('Sampled states')
for j in range(3):
    plt.plot(states[:,0,j].squeeze())

###################################################################
#
# Define Sampling Function for Simulated Run with given Actions
#
###################################################################

def inner_fn_init_trajectory(st, ot, oht, oat, ot_given, stm1):
    
    hst =  T.tanh( T.dot(Wq_hst_stm1,stm1) + T.dot(Wq_hst_ot,ot_given) + T.dot(Wq_hst_oht,oht) + T.dot(Wq_hst_oat,oat) + bq_hst )
    hst2 =  T.tanh( T.dot(Wq_hst2_hst,hst) + bq_hst2 )

    stmu =  T.tanh( T.dot(Wq_stmu_hst2,hst2) + bq_stmu )
    stsig = T.nnet.softplus( T.dot(Wq_stsig_hst2,hst2) + bq_stsig ) + sig_min_states
    
    # Rescale representation to fit within linear response of the tanh-nonlinearity
    stmu = T.set_subtensor(stmu[0,:],0.1*ot[0,:])
    stsig = T.set_subtensor(stsig[0,:],0.01)
    
    st_new = stmu + theano_rng.normal((n_s,n_proc))*stsig
    
    ost = T.tanh( T.dot(Wl_ost_st,st_new) + bl_ost )
    
    otmu = T.dot(Wl_otmu_st, ost) + bl_otmu
    otsig = T.nnet.softplus(T.dot(Wl_otsig_st, ost) + bl_otsig) + sig_min_obs
    
    ohtmu = T.dot(Wl_ohtmu_st, ost) + bl_ohtmu
    ohtsig = T.nnet.softplus( T.dot(Wl_ohtsig_st, ost) + bl_ohtsig ) + sig_min_obs
    
    oatmu = T.dot(Wl_oatmu_st, ost) + bl_oatmu
    oatsig = T.nnet.softplus( T.dot(Wl_oatsig_st, ost) + bl_oatsig ) + sig_min_obs
    
    ot_new = otmu + theano_rng.normal((n_o,n_samples))*otsig
    oht_new = ohtmu + theano_rng.normal((n_oh,n_samples))*ohtsig   
    oat_new = oatmu + theano_rng.normal((n_oa,n_samples))*oatsig    
    
    #oat_new = oat_given  
    
    return st_new, ot_new, oht_new, oat_new

def init_trajectory(ot_given, stm1):
    
    st0_condition = theano.shared(name = 's_t0', value = numpy.random.randn( n_s,n_samples ).astype( dtype = theano.config.floatX ), borrow = True )
    ot0_condition = theano.shared(name = 's_t0', value = numpy.random.randn( n_o,n_samples ).astype( dtype = theano.config.floatX ), borrow = True )
    oht0_condition = theano.shared(name = 's_t0', value = numpy.random.randn( n_oh,n_samples ).astype( dtype = theano.config.floatX ), borrow = True )
    oat0_condition = theano.shared(name = 's_t0', value = numpy.random.randn( n_oa,n_samples ).astype( dtype = theano.config.floatX ), borrow = True )
    
    ((st, ot, oht, oat), _) = theano.scan(fn=inner_fn_condition, 
                                           outputs_info=[st0_condition, ot0_condition, oht0_condition, oat0_condition],
                                           non_sequences=[ot_given, stm1],
                                           n_steps=100)
    
    st = st[-1]
    ot = ot[-1]
    oht = oht[-1]
    oat = oat[-1]    
    
    return st, ot, oht, oat

def inner_fn_condition(st, ot, oht, oat, oat_given, stm1):
    
    hst =  T.tanh( T.dot(Wq_hst_stm1,stm1) + T.dot(Wq_hst_ot,ot) + T.dot(Wq_hst_oht,oht) + T.dot(Wq_hst_oat,oat_given) + bq_hst )
    hst2 =  T.tanh( T.dot(Wq_hst2_hst,hst) + bq_hst2 )

    stmu =  T.tanh( T.dot(Wq_stmu_hst2,hst2) + bq_stmu )
    stsig = T.nnet.softplus( T.dot(Wq_stsig_hst2,hst2) + bq_stsig ) + sig_min_states
    
    # Rescale representation to fit within linear response of the tanh-nonlinearity
    stmu = T.set_subtensor(stmu[0,:],0.1*ot[0,:])
    stsig = T.set_subtensor(stsig[0,:],0.01)
    
    st_new = stmu + theano_rng.normal((n_s,n_proc))*stsig
    
    ost = T.tanh( T.dot(Wl_ost_st,st_new) + bl_ost )
    
    otmu = T.dot(Wl_otmu_st, ost) + bl_otmu
    otsig = T.nnet.softplus(T.dot(Wl_otsig_st, ost) + bl_otsig) + sig_min_obs
    
    ohtmu = T.dot(Wl_ohtmu_st, ost) + bl_ohtmu
    ohtsig = T.nnet.softplus( T.dot(Wl_ohtsig_st, ost) + bl_ohtsig ) + sig_min_obs
    
    oatmu = T.dot(Wl_oatmu_st, ost) + bl_oatmu
    oatsig = T.nnet.softplus( T.dot(Wl_oatsig_st, ost) + bl_oatsig ) + sig_min_obs
    
    ot_new = otmu + theano_rng.normal((n_o,n_samples))*otsig
    oht_new = ohtmu + theano_rng.normal((n_oh,n_samples))*ohtsig   
    oat_new = oatmu + theano_rng.normal((n_oa,n_samples))*oatsig    
    
    #oat_new = oat_given  
    
    return st_new, ot_new, oht_new, oat_new

def inner_fn_sample_actions_given(oat_given, stm1):
    
    st0_condition = theano.shared(name = 's_t0', value = numpy.random.randn( n_s,n_samples ).astype( dtype = theano.config.floatX ), borrow = True )
    ot0_condition = theano.shared(name = 's_t0', value = numpy.random.randn( n_o,n_samples ).astype( dtype = theano.config.floatX ), borrow = True )
    oht0_condition = theano.shared(name = 's_t0', value = numpy.random.randn( n_oh,n_samples ).astype( dtype = theano.config.floatX ), borrow = True )
    oat0_condition = theano.shared(name = 's_t0', value = numpy.random.randn( n_oa,n_samples ).astype( dtype = theano.config.floatX ), borrow = True )
    
    ((st, ot, oht, oat), _) = theano.scan(fn=inner_fn_condition, 
                                           outputs_info=[st0_condition, ot0_condition, oht0_condition, oat0_condition],
                                           non_sequences=[oat_given, stm1],
                                           n_steps=10)
    
    st = st[-1]
    ot = ot[-1]
    oht = oht[-1]
    oat = oat[-1]    
    
    return st, ot, oht, oat
    
# Define initial state and action
    
     
s_t0_sample_actions_given = theano.shared(name = 's_t0', value = numpy.zeros( (n_s,n_samples) ).astype( dtype = theano.config.floatX ), borrow = True )
ots_given = theano.shared(name = 's_t0', value = -4.0*numpy.ones( (10, n_o,n_samples) ).astype( dtype = theano.config.floatX ), borrow = True )

((states_sampled_it, observations_sampled_it, rewards_sampled_it, actions_observations_sampled_it), _) =\
                     theano.scan(fn=init_trajectory,
                     outputs_info=[s_t0_sample, None, None, None],
                     sequences = [ots_given]
                     )  
                     
s_t0_it = states_sampled_it[-1]

oats_given = theano.shared(name = 's_t0', value = numpy.concatenate( (4.0*numpy.ones( (30, n_oa,n_samples) ), -4.0*numpy.ones( (30, n_oa,n_samples) ), 0.0*numpy.ones( (30, n_oa,n_samples) )), axis = 0).astype( dtype = theano.config.floatX ), borrow = True )
     
((states_sampled_ag, observations_sampled_ag, rewards_sampled_ag, actions_observations_sampled_ag), _) =\
                     theano.scan(fn=inner_fn_sample_actions_given,
                     outputs_info=[s_t0_it, None, None, None],
                     sequences = [oats_given]
                     )     
                     
                     
########################################################
#
# Test Calculation of Combined Objective
#
########################################################

# Build Function
sample_ag = theano.function([], [observations_sampled_ag, rewards_sampled_ag, actions_observations_sampled_ag, states_sampled_ag], allow_input_downcast = True, on_unused_input='ignore')

[observations_ag, rewards_ag, actions_ag, states_ag] =  sample_ag()


FE_min = results_eval[0]

observations = results_eval[1]
rewards = results_eval[2]
actions = results_eval[3]
states = results_eval[4]

print 'FE_min:'
print FE_min

plt.subplot(3,4,9)
plt.title('Sampled AG trajectories')
for j in range(10):
    plt.plot(observations_ag[:,0,j].squeeze())
plt.subplot(3,4,10)
plt.title('Sampled AG rewards')
for j in range(10):
    plt.plot(rewards_ag[:,0,j].squeeze())
plt.subplot(3,4,11)
plt.title('Sampled AG actions')
for j in range(10):
    plt.plot(actions_ag[:,0,j].squeeze())
plt.subplot(3,4,12)
plt.title('Sampled AG states')
for j in range(10):
    plt.plot(states_ag[:,0,j].squeeze())

plt.show()
