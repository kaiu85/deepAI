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

# For nicer text rendering, requires LateX
plt.rc('text', usetex=True)
plt.rc('font', family='serif') 

# Parameters (Should be identical to fitting script)
n_s = 20 # States

n_o = 1 # Sensory Input encoding Position
n_oh = 1 # Nonlinearly Transformed Channel (OPTIONAL!)
n_oa = 1 # Proprioception (OPTIONAL!)

n_run_steps = 30
n_proc = 10

n_steps = 1000000

sig_min_obs = 1e-6
sig_min_states = 1e-6
sig_min_action = 1e-6

n_sample_steps = 30
n_samples = 10

# Sampling Parameters 

# Constrained Sampling(_ag means "actions given")
n_iterations_ag = 5000 # How many iterations of MCMC to perform for 
                       # constrained sampling (100 is considerably faster)
shift_ag = 10 # Offset to shift action timecourse for constrained sampling

# Initialize RNG
ii32 = numpy.iinfo(numpy.int32)
theano_rng = RandomStreams(numpy.random.randint(ii32.max)) # ADD RANDOM SEED!  

# Helper Functions and Classes
def GaussianNLL(y, mu, sig):

    nll = 0.5 * T.sum(T.sqr(y - mu) / sig**2 + 2 * T.log(sig) +
                      T.log(2 * numpy.pi), axis=0)
    return nll   

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

def reduce_first_dim(x):
    
    ms = []
    for m in x:
        m = m.reshape( (m.shape[1], m.shape[2]) )
        ms.append(m)
    return ms

# Load learned Agent
with open('deepAI_demo_best.pkl', 'r') as f:
    [params, sigmas] = cPickle.load(f)

[Wq_hst_ot, Wq_hst_oht, Wq_hst_oat, Wq_hst_stm1, bq_hst,\
Wq_hst2_hst, bq_hst2,\
Wq_stmu_hst2, bq_stmu,\
Wq_stsig_hst2, bq_stsig,\
Wl_stmu_stm1, bl_stmu,\
Wl_stsig_stm1, bl_stsig,\
Wl_ost_st, bl_ost,\
Wl_ost2_ost, bl_ost2,\
Wl_ost3_ost2, bl_ost3,\
Wl_otmu_st, bl_otmu,\
Wl_otsig_st, bl_otsig,\
Wl_ohtmu_st, bl_ohtmu,\
Wl_ohtsig_st, bl_ohtsig,\
Wl_oatmu_st, bl_oatmu,\
Wl_oatsig_st, bl_oatsig,
Wa_atmu_st, ba_atmu,\
Wa_atsig_st, ba_atsig,\
] = reduce_first_dim(params)

###################################################################
#  
# Define Variational Free Energy for Simulated Run
#
###################################################################

# Build sampling function
def inner_fn(t, stm1, oat, ot, oht, pos, vt):
   
    hst =  T.nnet.relu( T.dot(Wq_hst_stm1,stm1) + T.dot(Wq_hst_ot,ot) + T.dot(Wq_hst_oht,oht) + T.dot(Wq_hst_oat,oat) + bq_hst )
    hst2 =  T.nnet.relu( T.dot(Wq_hst2_hst,hst) + bq_hst2 )

    stmu =  T.tanh( T.dot(Wq_stmu_hst2,hst2) + bq_stmu )
    stsig = T.nnet.softplus( T.dot(Wq_stsig_hst2,hst2) + bq_stsig ) + sig_min_states
    
    # Rescale representation to fit within linear response of the tanh-nonlinearity
    stmu = T.set_subtensor(stmu[0,:],0.1*ot[0,:])
    stsig = T.set_subtensor(stsig[0,:],0.01)
    
    st = stmu + theano_rng.normal((n_s,n_proc))*stsig
    
    ost = T.nnet.relu( T.dot(Wl_ost_st,st) + bl_ost )
    ost2 = T.nnet.relu( T.dot(Wl_ost2_ost,ost) + bl_ost2 )
    ost3 = T.nnet.relu( T.dot(Wl_ost3_ost2,ost2) + bl_ost3 )
    
    otmu = T.dot(Wl_otmu_st, ost3) + bl_otmu
    otsig = T.nnet.softplus(T.dot(Wl_otsig_st, ost3) + bl_otsig) + sig_min_obs
    
    ohtmu = T.dot(Wl_ohtmu_st, ost3) + bl_ohtmu
    ohtsig = T.nnet.softplus( T.dot(Wl_ohtsig_st, ost3) + bl_ohtsig ) + sig_min_obs
    
    oatmu = T.dot(Wl_oatmu_st, ost3) + bl_oatmu
    oatsig = T.nnet.softplus( T.dot(Wl_oatsig_st, ost3) + bl_oatsig ) + sig_min_obs
    
    p_ot  = GaussianNLL(ot, otmu, otsig)
    p_oht = GaussianNLL(oht, ohtmu, ohtsig)    
    p_oat = GaussianNLL(oat, oatmu, oatsig)
    
    prior_stmu = T.tanh( T.dot(Wl_stmu_stm1, stm1) + bl_stmu )
    prior_stsig = T.nnet.softplus( T.dot(Wl_stsig_stm1, stm1) + bl_stsig ) + sig_min_states
    
    prior_stmu = ifelse(T.lt(t,20),prior_stmu, T.set_subtensor(prior_stmu[0,:],0.1))
    prior_stsig = ifelse(T.lt(t,20),prior_stsig, T.set_subtensor(prior_stsig[0,:],0.01))    
   
    KL_st = KLGaussianGaussian(stmu, stsig, prior_stmu, prior_stsig)
    
    FEt =  KL_st + p_ot + p_oht + p_oat
    
    oat_mu = T.dot(Wa_atmu_st, st) + ba_atmu
    oat_sig = T.nnet.softplus( T.dot(Wa_atsig_st, st) + ba_atsig ) + sig_min_action
    
    oat_new = 0.0*oat + oat_mu + theano_rng.normal((n_oa,n_proc))*oat_sig
    
    action_force = T.tanh( oat_new )
    force = T.switch(T.lt(pos,0.0),-2*pos - 1,-T.pow(1+5*T.sqr(pos),-0.5)-T.sqr(pos)*T.pow(1 + 5*T.sqr(pos),-1.5)-T.pow(pos,4)/16.0) - 0.25*vt
    vt_new = vt + 0.05*force + 0.03*action_force
    pos_new = pos + vt_new     
    
    ot_new = pos_new + theano_rng.normal((n_o,n_samples))*0.01
    
    oht_new = T.exp(-T.sqr(pos_new-1.0)/2.0/0.3/0.3)
    
    return st, oat_new, ot_new, oht_new, pos_new, vt_new, FEt, KL_st, stmu, stsig, force, p_ot, p_oht, p_oat
     
s_t0 = theano.shared(name = 's_t0', value = numpy.zeros( (n_s, n_proc) ).astype( dtype = theano.config.floatX ), borrow = True )
a_t0 = theano.shared(name = 'a_t0', value = numpy.zeros( (n_oa, n_proc) ).astype( dtype = theano.config.floatX ), borrow = True )
o_t0 = theano.shared(name = 'o_t0', value = numpy.zeros( (n_o, n_proc) ).astype( dtype = theano.config.floatX ), borrow = True )
oh_t0 = theano.shared(name = 'oh_t0', value = numpy.zeros( (n_oh, n_proc) ).astype( dtype = theano.config.floatX ), borrow = True )
pos_t0 = theano.shared(name = 'pos_t0', value = -0.5 + 0.0*numpy.random.randn( n_o, n_proc ).astype( dtype = theano.config.floatX ), borrow = True )
v_t0 = theano.shared(name = 'v_t0', value = 0.0*numpy.random.randn( n_o, n_proc ).astype( dtype = theano.config.floatX ), borrow = True )
    
observations_th = T.tensor3('observations_th')
rewards_th = T.tensor3('rewards_th')
action_observations_th = T.tensor3('actions_in_th')
state_noise_th = T.tensor3('state_noise_th')
action_noise_th = T.tensor3('action_noise_th')
    
((states_th, oat_th, ot_th, oht_th, post_th, vt_th, FEt_th, KL_st_th, stmu_th, stsig_th, aht_th, p_ot_th, p_oht_th, p_oat_th), fe_updates) =\
                     theano.scan(fn=inner_fn,
                     sequences = [theano.shared(numpy.arange(n_run_steps).astype(dtype = theano.config.floatX))],
                     outputs_info=[s_t0, a_t0, o_t0, oh_t0, pos_t0, v_t0, None, None, None, None, None, None, None, None])
                     
last_stmus = stmu_th[-1,0,:].squeeze().dimshuffle('x',0)
last_stsigs = stsig_th[-1,0,:].squeeze().dimshuffle('x',0)

fixed_prior_cost = GaussianNLL(last_stmus, 0.3, 0.01)
                     
FE_mean = FEt_th.mean()# + fixed_prior_cost.mean()
KL_st_mean = KL_st_th.mean()
p_ot_mean = p_ot_th.mean()
p_oht_mean = p_oht_th.mean()
p_oat_mean = p_oat_th.mean()

run_agent_scan = theano.function(inputs = [], outputs = [states_th, oat_th, ot_th, oht_th, FEt_th, KL_st_th, stmu_th, stsig_th, aht_th], allow_input_downcast = True, on_unused_input='ignore')

#######################################################
#
# Propagate agent through the environment
# and Plot Results
#
#######################################################

states, actions, observations, rewards, FEs, KLs, stmus, stsigs, ahts = run_agent_scan()

plt.figure(1)
plt.subplot(3,4,5)
plt.title('$o(t)$ propagated')
for j in range(10):
    plt.plot(observations[:,0,j].squeeze())
plt.subplot(3,4,6)
plt.title('$o_h(t)$ propagated')
for j in range(10):
    plt.plot(rewards[:,0,j].squeeze())
plt.subplot(3,4,7)
plt.title('$o_a(t)$ propagated')
for j in range(10):
    plt.plot(actions[:,0,j].squeeze())
plt.subplot(3,4,8)
plt.title('$s_1(t)$ propagated')
for j in range(10):
    plt.plot(states[:,0,j].squeeze())

# Save mean action to use as template for constrained sampling later
oat_given_mean = actions.mean(axis = 2).squeeze()

########################################################
#
# Test free energy calculation
#
########################################################

free_energy = theano.function([], [FE_mean, KL_st_mean, p_ot_mean, p_oht_mean, p_oat_mean], allow_input_downcast = True, on_unused_input='ignore')

free_energy_sum = free_energy()

print 'Free Energy'
print free_energy_sum

#############################################################
#
# Define Sampling Function for Simulated Run
#
#############################################################

def inner_fn_sample(stm1):
    
    prior_stmu = T.tanh( T.dot(Wl_stmu_stm1, stm1) + bl_stmu )
    prior_stsig = T.nnet.softplus( T.dot(Wl_stsig_stm1, stm1) + bl_stsig ) + sig_min_states
    
    # Set explicit prior on score during last time step
    #prior_stmu = ifelse(T.lt(t,n_run_steps - 5),prior_stmu, T.set_subtensor(prior_stmu[0,:],0.1))
    #prior_stsig = ifelse(T.lt(t,n_run_steps - 5),prior_stsig, T.set_subtensor(prior_stsig[0,:],0.001))    
    
    st = prior_stmu + theano_rng.normal((n_s,n_samples))*prior_stsig
    
    ost = T.nnet.relu( T.dot(Wl_ost_st,st) + bl_ost )
    ost2 = T.nnet.relu( T.dot(Wl_ost2_ost,ost) + bl_ost2 )
    ost3 = T.nnet.relu( T.dot(Wl_ost3_ost2,ost2) + bl_ost3 )
    
    otmu = T.dot(Wl_otmu_st, ost3) + bl_otmu
    otsig = T.nnet.softplus(T.dot(Wl_otsig_st, ost3) + bl_otsig) + sig_min_obs
    
    ohtmu = T.dot(Wl_ohtmu_st, ost3) + bl_ohtmu
    ohtsig = T.nnet.softplus( T.dot(Wl_ohtsig_st, ost3) + bl_ohtsig ) + sig_min_obs
    
    oatmu = T.dot(Wl_oatmu_st, ost3) + bl_oatmu
    oatsig = T.nnet.softplus( T.dot(Wl_oatsig_st, ost3) + bl_oatsig ) + sig_min_obs
    
    ot = otmu + theano_rng.normal((n_o,n_samples))*otsig
    oht = ohtmu + theano_rng.normal((n_oh,n_samples))*ohtsig   
    oat = oatmu + theano_rng.normal((n_oa,n_samples))*oatsig   
    
    return st, ohtmu, ohtsig, ot, oht, oat, prior_stmu, prior_stsig
    
# Define initial state and action
    
s_t0_sample = theano.shared(name = 's_t0', value = numpy.zeros( (n_s,n_samples) ).astype( dtype = theano.config.floatX ), borrow = True )
    

    
((states_sampled, reward_probabilities_mu_sampled, reward_probabilities_sig_sampled, observations_sampled, rewards_sampled, actions_observations_sampled, stmus_sampled, stsigs_sampled), updates_sampling) =\
                     theano.scan(fn=inner_fn_sample,
                     outputs_info=[s_t0_sample, None, None, None, None, None, None, None],
                     n_steps = n_sample_steps)                     

########################################################
#
# Run and Plot Results for Free Sampling
#
########################################################

# Build Function
eval_Penalized_FE = theano.function([], [FE_mean, observations_sampled, rewards_sampled, actions_observations_sampled, states_sampled], allow_input_downcast = True, on_unused_input='ignore')

# Evaluate Function
#print 'Penalized FE'
results_eval =  eval_Penalized_FE()
#print results_eval

FE_min = results_eval[0]

observations = results_eval[1]
rewards = results_eval[2]
actions = results_eval[3]
states = results_eval[4]

print 'FE_min:'
print FE_min

plt.subplot(3,4,1)
plt.title('$o(t)$ sampled')
for j in range(10):
    plt.plot(observations[:,0,j].squeeze())
plt.subplot(3,4,2)
plt.title('$o_h(t)$ sampled')
for j in range(10):
    plt.plot(rewards[:,0,j].squeeze())
plt.subplot(3,4,3)
plt.title('$o_a(t)$ sampled')
for j in range(10):
    plt.plot(actions[:,0,j].squeeze())
plt.subplot(3,4,4)
plt.title('$s_1(t)$ sampled')
for j in range(10):
    plt.plot(states[:,0,j].squeeze())

###################################################################
#
# Define Sampling Function for Simulated Run with given Actions
#
###################################################################

# Inner function for MCMC sampler
def inner_fn_condition(st, ot, oht, oat, oat_given, stm1):
    
    hst =  T.nnet.relu( T.dot(Wq_hst_stm1,stm1) + T.dot(Wq_hst_ot,ot) + T.dot(Wq_hst_oht,oht) + T.dot(Wq_hst_oat,oat_given) + bq_hst )
    hst2 =  T.nnet.relu( T.dot(Wq_hst2_hst,hst) + bq_hst2 )

    stmu =  T.tanh( T.dot(Wq_stmu_hst2,hst2) + bq_stmu )
    stsig = T.nnet.softplus( T.dot(Wq_stsig_hst2,hst2) + bq_stsig ) + sig_min_states
    
    # Rescale representation to fit within linear response of the tanh-nonlinearity
    stmu = T.set_subtensor(stmu[0,:],0.1*ot[0,:])
    stsig = T.set_subtensor(stsig[0,:],0.01)
    
    st_new = stmu + theano_rng.normal((n_s,n_proc))*stsig
    
    ost = T.nnet.relu( T.dot(Wl_ost_st,st) + bl_ost )
    ost2 = T.nnet.relu( T.dot(Wl_ost2_ost,ost) + bl_ost2 )
    ost3 = T.nnet.relu( T.dot(Wl_ost3_ost2,ost2) + bl_ost3 )
    
    otmu = T.dot(Wl_otmu_st, ost3) + bl_otmu
    otsig = T.nnet.softplus(T.dot(Wl_otsig_st, ost3) + bl_otsig) + sig_min_obs
    
    ohtmu = T.dot(Wl_ohtmu_st, ost3) + bl_ohtmu
    ohtsig = T.nnet.softplus( T.dot(Wl_ohtsig_st, ost3) + bl_ohtsig ) + sig_min_obs
    
    oatmu = T.dot(Wl_oatmu_st, ost3) + bl_oatmu
    oatsig = T.nnet.softplus( T.dot(Wl_oatsig_st, ost3) + bl_oatsig ) + sig_min_obs
    
    ot_new = otmu + theano_rng.normal((n_o,n_samples))*otsig
    oht_new = ohtmu + theano_rng.normal((n_oh,n_samples))*ohtsig   
    oat_new = oatmu + theano_rng.normal((n_oa,n_samples))*oatsig        
    
    return st_new, ot_new, oht_new, oat_new

def inner_fn_sample_actions_given(oat_given, stm1):
    
    st0_condition = theano.shared(name = 'st0_condition', value = numpy.random.randn( n_s,n_samples ).astype( dtype = theano.config.floatX ), borrow = True )
    ot0_condition = theano.shared(name = 'ot0_condition', value = numpy.random.randn( n_o,n_samples ).astype( dtype = theano.config.floatX ), borrow = True )
    oht0_condition = theano.shared(name = 'oht0_condition', value = numpy.random.randn( n_oh,n_samples ).astype( dtype = theano.config.floatX ), borrow = True )
    oat0_condition = theano.shared(name = 'oat0_condition', value = numpy.random.randn( n_oa,n_samples ).astype( dtype = theano.config.floatX ), borrow = True )
    
    # Iterate MCMC sampler to approximate constrained probabilities
    # p(o,oh|oa) of observations, given a sequence of proprioceptive
    # inputs oa
    # c.f. https://arxiv.org/abs/1401.4082, Appendix F.
    ((st, ot, oht, oat), _) = theano.scan(fn=inner_fn_condition, 
                                           outputs_info=[st0_condition, ot0_condition, oht0_condition, oat0_condition],
                                           non_sequences=[oat_given, stm1],
                                           n_steps=n_iterations_ag)
    
    st = st[-1]
    ot = ot[-1]
    oht = oht[-1]
    oat = oat[-1]    
    
    return st, ot, oht, oat
    
# Define initial state and action    
st0_init = theano.shared(name = 'st0_init', value = 0.0*numpy.random.randn( n_s,n_samples ).astype( dtype = theano.config.floatX ), borrow = True )
ot0_init = theano.shared(name = 'ot0_init', value = -0.5 + 0.0*numpy.random.randn( n_o,n_samples ).astype( dtype = theano.config.floatX ), borrow = True )
oht0_init = theano.shared(name = 'oht0_init', value = 0.0*numpy.random.randn( n_oh,n_samples ).astype( dtype = theano.config.floatX ), borrow = True )
oat0_init = theano.shared(name = 'oat0_init', value = 0.0*numpy.random.randn( n_oa,n_samples ).astype( dtype = theano.config.floatX ), borrow = True )

oats_given_val = numpy.zeros( (n_sample_steps, 1, n_samples) )

for i in range(n_samples):
    j = shift_ag
    
    if j == 0:
        oats_given_val[:,0,i] = oat_given_mean
    elif j > 0:
        oats_given_val[j:,0,i] = oat_given_mean[:-j]
        oats_given_val[:j,0,i] = oat_given_mean[0]
    else:
        oats_given_val[:j,0,i] = oat_given_mean[-j:]
        oats_given_val[j:,0,i] = oat_given_mean[-1]

oats_given = theano.shared(name = 's_t0', value = oats_given_val.astype(dtype = theano.config.floatX), broadcastable = (False, False, False), borrow = True)
        
((states_sampled_ag, observations_sampled_ag, rewards_sampled_ag, actions_observations_sampled_ag), _) =\
                     theano.scan(fn=inner_fn_sample_actions_given,
                     outputs_info=[st0_init, None, None, None],
                     sequences = [oats_given]
                     )   
                     
########################################################
#
# Run and Plot Results for Constrained Sampling
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
plt.title('$o(t)$ constrained')
for j in range(10):
    plt.plot(observations_ag[:,0,j].squeeze())
plt.xlabel('steps')
plt.subplot(3,4,10)
plt.title('$o_h(t)$ constrained')
for j in range(10):
    plt.plot(rewards_ag[:,0,j].squeeze())
plt.xlabel('steps')
plt.subplot(3,4,11)
plt.title('$o_a(t)$ constrained')
for j in range(10):
    plt.plot(actions_ag[:,0,j].squeeze())
plt.xlabel('steps')
plt.subplot(3,4,12)
plt.title('$s_1(t)$ constrained')
for j in range(10):
    plt.plot(states_ag[:,0,j].squeeze())
plt.xlabel('steps')

plt.show()
