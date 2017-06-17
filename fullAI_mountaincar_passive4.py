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

learning_rate = 0.001

init_sig_obs = -3.0
init_sig_states = -3.0
init_sig_action = -3.0

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

# Generate List of Parameters to Optimize
params = []

# Parameters of approximate posterior

# Q(s_t | s_t-1, o_t, oh_t, oa_t)

Wq_hst_ot = theano.shared(
    value=initweight(n_s, n_o),
    name='Wq_hst_ot',
    borrow=True
)

params.append(Wq_hst_ot)

Wq_hst_oht = theano.shared(
    value=initweight(n_s, n_oh),
    name='Wq_hst_oht',
    borrow=True
)

params.append(Wq_hst_oht)

Wq_hst_oat = theano.shared(
    value=initweight(n_s, n_oa),
    name='Wq_hst_oat',
    borrow=True
)

params.append(Wq_hst_oat)

Wq_hst_stm1 = theano.shared(
    value=initweight(n_s, n_s),
    name='Wq_hst_stm1',
    borrow=True
)

params.append(Wq_hst_stm1)

bq_hst = theano.shared(
    value=initconst(n_s, 1),
    name='bq_hst',
    borrow=True,
    broadcastable=(False,True)
)

params.append(bq_hst)

Wq_hst2_hst = theano.shared(
    value=initweight(n_s, n_s),
    name='Wq_hst2_hst',
    borrow=True
)

params.append(Wq_hst2_hst)

bq_hst2 = theano.shared(
    value=initconst(n_s, 1),
    name='bq_hst2',
    borrow=True,
    broadcastable=(False,True)
)

params.append(bq_hst2)

Wq_stmu_hst2 = theano.shared(
    value=initweight(n_s, n_s),
    name='Wq_stmu_hst2',
    borrow=True
)

params.append(Wq_stmu_hst2)

bq_stmu = theano.shared(
    value=initconst(n_s, 1),
    name='bq_stmu',
    borrow=True,
    broadcastable=(False,True)
)

params.append(bq_stmu)

Wq_stsig_hst2 = theano.shared(
    value=initweight(n_s, n_s),
    name='Wq_stsig_hst2',
    borrow=True
)

params.append(Wq_stsig_hst2)

bq_stsig = theano.shared(
    value=initconst(n_s,1,init_sig_states),
    name='bq_stsig',
    borrow=True,
    broadcastable=(False,True)
)

params.append(bq_stsig)

# Define Parameters for Likelihood Function

# p( s_t | s_t-1 )

Wl_stmu_stm1 = theano.shared(
    value=initweight(n_s, n_s),#Wq_stmu_stm1.get_value(),#initortho(n_s, n_s),
    name='Wl_stmu_stm1',
    borrow=True
)

params.append(Wl_stmu_stm1)

bl_stmu = theano.shared(
    value=initconst(n_s, 1),
    name='bl_stmu',
    borrow=True,
    broadcastable=(False,True)
)

params.append(bl_stmu)

Wl_stsig_stm1 = theano.shared(
    value=initweight(n_s, n_s),#Wq_stsig_stm1.get_value(),#initweight(n_s, n_s),
    name='Wl_stsig_stm1',
    borrow=True
)

params.append(Wl_stsig_stm1)

bl_stsig = theano.shared(
    value=initconst(n_s, 1,init_sig_states),
    name='bl_stsig',
    borrow=True,
    broadcastable=(False,True)
)

params.append(bl_stsig)

Wl_ost_st = theano.shared(
    value=initweight(n_s, n_s),
    name='Wl_ost_st',
    borrow=True
)

params.append(Wl_ost_st)

bl_ost = theano.shared(
    value=initconst(n_s, 1),
    name='bl_ost',
    borrow=True,
    broadcastable=(False,True)
)

params.append(bl_ost)

# p( o_t | s_t )

Wl_otmu_st = theano.shared(
    value=initweight(n_o, n_s),
    name='Wl_otmu_st',
    borrow=True
)

params.append(Wl_otmu_st)

bl_otmu = theano.shared(
    value=initconst(n_o, 1),
    name='bl_otmu',
    borrow=True,
    broadcastable=(False,True)
)

params.append(bl_otmu)

Wl_otsig_st = theano.shared(
    value=initweight(n_o, n_s),
    name='Wl_otsig_st',
    borrow=True
)

params.append(Wl_otsig_st)

bl_otsig = theano.shared(
    value=initconst(n_o,1,init_sig_obs),
    name='bl_otsig',
    borrow=True,
    broadcastable=(False,True)
)

params.append(bl_otsig)

# p( oh_t | s_t )

Wl_ohtmu_st = theano.shared(
    value=initweight(n_oh, n_s),
    name='Wl_ohtmu_st',
    borrow=True
)

params.append(Wl_ohtmu_st)

bl_ohtmu = theano.shared(
    value=initconst(n_oh, 1),
    name='bl_ohtmu',
    borrow=True,
    broadcastable=(False,True)
)

params.append(bl_ohtmu)

Wl_ohtsig_st = theano.shared(
    value=initweight(n_oh, n_s),
    name='Wl_ohtsig_st',
    borrow=True
)

params.append(Wl_ohtsig_st)

bl_ohtsig = theano.shared(
    value=initconst(n_oh, 1,init_sig_obs),
    name='bl_ohtsig',
    borrow=True,
    broadcastable=(False,True)
)

params.append(bl_ohtsig)

# p( oa_t | s_t, a_t )

Wl_oatmu_st = theano.shared(
    value=initweight(n_oa, n_s),
    name='Wl_oatmu_st',
    borrow=True
)

params.append(Wl_oatmu_st)

bl_oatmu = theano.shared(
    value=initconst(n_oa, 1),
    name='bl_oatmu',
    borrow=True,
    broadcastable=(False,True)
)

params.append(bl_oatmu)

Wl_oatsig_st = theano.shared(
    value=initweight(n_oa, n_s),
    name='Wl_oatsig_st',
    borrow=True
)

params.append(Wl_oatsig_st)

bl_oatsig = theano.shared(
    value=initconst(n_oa, 1,init_sig_obs),
    name='bl_oatsig',
    borrow=True,
    broadcastable=(False,True)
)

params.append(bl_oatsig)

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

run_agent_scan = theano.function(inputs = [], outputs = [states_th, oat_th, ot_th, oht_th, FEt_th, KL_st_th, hst_th, hst2_th, stmu_th, stsig_th, aht_th, fixed_prior_cost], allow_input_downcast = True, on_unused_input='ignore')

#######################################################
#
# Test agent and plot some properties of
# the environment
#
#######################################################

states, actions, observations, rewards, FEs, KLs, hsts, hst2s, stmus, stsigs, ahts, fp_cost = run_agent_scan()

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

print 'shape:'
print states.shape

plt.figure(1)
for j in range(3):
    plt.plot(observations[:,0,j].squeeze())
plt.figure(2)
for j in range(3):
    plt.plot(rewards[:,0,0].squeeze())
plt.figure(3)
for j in range(3):
    plt.plot(observations[:-1,0,j].squeeze(), ahts[1:,0,j].squeeze(),"o")
plt.figure(4)
for j in range(3):
    plt.plot(actions[:,0,j].squeeze())
#for j in range(1):
#    plt.plot(state_noise[:,0,0].squeeze())
print 'fp_cost:'
print fp_cost

plt.show()

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

FE_min = oFE_mean

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
    
# Build sampling function
    
((states_sampled, reward_probabilities_mu_sampled, reward_probabilities_sig_sampled, observations_sampled, rewards_sampled, actions_observations_sampled, stmus_sampled, stsigs_sampled), updates_sampling) =\
                     theano.scan(fn=inner_fn_sample,
                     outputs_info=[s_t0_sample, None, None, None, None, None, None, None],
                     n_steps = n_sample_steps)   
    
########################################################################
#
# Prepare Optimization
#
########################################################################

# Create List of Updates
updates = []

for param in params:
    p_update = Adam(FE_mean, param, 0.9, 0.999, learning_rate, epsilon = 10e-6)
    updates = updates + p_update.updates    
    
# Define Training Function
train = theano.function(
        inputs=[],
        outputs=[FE_mean, KL_st_mean, p_ot_mean, p_oht_mean, p_oat_mean, fixed_prior_cost.mean()], 
        updates=updates,
        on_unused_input='ignore',
        allow_input_downcast = True
    )
  
########################################################################
#
# Run Optimization
#
########################################################################

# Optimization Loop
for i in range(n_steps):
        
    # Take the time for each loop
    start_time = timeit.default_timer()
    
    print 'Iteration: %d' % i    

    # Perform stochastic gradient descent using ADAM updates
    print 'Descending on Free Energy...'    
    [oFE_mean, oKL_st_mean, op_ot_mean, op_oht_mean, po_oat_mean, ofixed_prior_cost] = train()
    
    print [oFE_mean, oKL_st_mean, op_ot_mean, op_oht_mean, po_oat_mean, ofixed_prior_cost]
       
    if i == 0:
        with open("log_fullAI_mountaincar_passive4.txt", "w") as myfile:
            myfile.write("%f %f %f %f %f %f %f\n" % (oFE_mean, oKL_st_mean, op_ot_mean, op_oht_mean, po_oat_mean, ofixed_prior_cost, FE_min))
    else:
        with open("log_fullAI_mountaincar_passive4.txt", "a") as myfile:
            myfile.write("%f %f %f %f %f %f %f\n" % (oFE_mean, oKL_st_mean, op_ot_mean, op_oht_mean, po_oat_mean, ofixed_prior_cost, FE_min))
    
    # Stop time
    end_time = timeit.default_timer()
    print 'Time for iteration: %f' % (end_time - start_time)
    
    # Save current parameters every nth loop
    if i % 100 == 0:
        with open('fullAI_mountaincar_passive4.pkl', 'w') as f:
            cPickle.dump(params, f)
            
    # Save best parameters
    if oFE_mean < FE_min:
        FE_min = oFE_mean
        with open('fullAI_mountaincar_passive4_best.pkl', 'w') as f:
            cPickle.dump(params, f)
        

# Save final parameters
with open('fullAI_mountaincar_passive4.pkl', 'w') as f:
    cPickle.dump(params, f)


