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
n_s = 2 # States

n_o = 1 # Sensory Input
n_oh = 2 # Rewards (OPTIONAL!)
n_oa = 1 # Proprioception (OPTIONAL!)

n_run_steps = 5
n_proc = 1000
n_perturbations = 10000# 20000

n_steps = 1000000

sig_min_obs = 0.001
sig_min_states = 0.001
sig_min_perturbations = 1e-6

learning_rate = 1e-3

init_sig_obs = -3.0
init_sig_states = -3.0
init_sig_perturbations = 0.05

sig_test = 0.0

# Initialize RNG
ii32 = numpy.iinfo(numpy.int32)
theano_rng = RandomStreams(numpy.random.randint(ii32.max)) # ADD RANDOM SEED!  

# Helper Functions and Classes

def initweight(shape1, shape2, minval =-0.5, maxval = 0.5):
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

param1 = theano.shared(
    value=initweight(n_s, n_o).reshape(1,n_s,n_o),
    name='param1',
    borrow=True,
    broadcastable=(True, False, False)
    
)

params.append(param1)

param2 = theano.shared(
    value=initweight(n_s, n_oh).reshape(1,n_s,n_oh),
    name='param2',
    borrow=True,
    broadcastable=(True, False, False)
)

params.append(param2)

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
        r_param = params[i] + r_epsilon*( T.nnet.softplus( sigmas[i] ) + sig_min_perturbations )
        r_params.append(r_param)
        r_epsilons.append(r_epsilon)
        
    return r_params, r_epsilons
    
####################################################################
#
# Create randomly perturbed version of parameters
#
####################################################################
    
sigmas = initialize_sigmas(params, init_sig_perturbations)

[sigma1, sigma2] = sigmas
tsigma1 = ( T.nnet.softplus( sigma1 ) + sig_min_perturbations )
tsigma2 = ( T.nnet.softplus( sigma2 ) + sig_min_perturbations )
    
[r_params, r_epsilons] = randomize_parameters( params, sigmas, sig_min_perturbations )

[r_param1, r_param2] = r_params

###################################################################
#  
# Define Variational Free Energy for Simulated Run
#
###################################################################

def inner_fn(t, target_0, target_1, r_param1, r_param2):
   
    Error1 = ( T.sqr( r_param1 - target_0 ).dimshuffle(0,1,2,'x') + sig_test*theano_rng.normal( (n_perturbations, n_s, n_o, n_proc) ) ).mean(axis = [1,2])
    Error2 = ( T.sqr( r_param2 - target_1 ).dimshuffle(0,1,2,'x') + sig_test*theano_rng.normal( (n_perturbations, n_s, n_o, n_proc) ) ).mean(axis = [1,2])
    
    FEt = Error1 + Error2
    
    return  FEt
 
target_0 = theano.shared(name = 'target_0', value = numpy.ones( (1,n_s,n_o) ).astype( dtype = theano.config.floatX ), borrow = True, broadcastable = (True, False, False) )
target_1 = theano.shared(name = 'target_1', value = -1.0*numpy.ones( (1,n_s,n_oh) ).astype( dtype = theano.config.floatX ), borrow = True, broadcastable = (True, False, False) )
   
((FEt_th), fe_updates) =\
                     theano.scan(fn=inner_fn,
                     sequences = [numpy.arange(n_run_steps).astype(dtype = theano.config.floatX)],
                     non_sequences = [target_0, target_1, r_param1, r_param2],
                     outputs_info=[None])

           
cError1 = ( T.sqr( r_param1 - target_0 ).dimshuffle('x',0,1,2,'x') + sig_test*theano_rng.normal( (n_run_steps, n_perturbations, n_s, n_o, n_proc) ) ).mean(axis = [2,3])
cError2 = ( T.sqr( r_param2 - target_1 ).dimshuffle('x',0,1,2,'x') + sig_test*theano_rng.normal( (n_run_steps, n_perturbations, n_s, n_o, n_proc) ) ).mean(axis = [2,3])
    
cFEt_th = cError1 + cError2      

diff = FEt_th - cFEt_th

cFE_mean = cFEt_th.mean()

cFE_time = cFEt_th.mean(axis = [1,2])
                     
FE_mean = FEt_th.mean()

FE_time = FEt_th.mean(axis = [1,2])

FE_mean_perturbations = FEt_th.mean(axis = [0,2])

FE_mean_mean_perturbations = FE_mean_perturbations.mean()

FE_rank = n_perturbations - T.argsort( T.argsort(FE_mean_perturbations) )

FE_rank_score = T.clip( numpy.log(0.5*n_perturbations+1) - T.log(FE_rank) , 0.0, 10000.0).astype(dtype = theano.config.floatX)

FE_rank_score_normalized = FE_rank_score/FE_rank_score.sum() - 1.0/n_perturbations

########################################################
#
# Test free energy calculation
#
########################################################

free_energy = theano.function([], [FE_mean, FEt_th, FE_mean_perturbations, r_epsilons[0], FE_mean_mean_perturbations, FE_time, r_params[0], diff, cFE_mean, cFE_time], allow_input_downcast = True, on_unused_input='ignore')

[free_energy_mean, free_energy_coords, free_energy_mean_perturbations, epsilons_perturbation, perturbations_mean, oFE_time, r_param, cdiff, ocFE_mean, ocFE_time] = free_energy()

print 'Free Energy'
print free_energy_mean
print 'SHAPE:'
print free_energy_coords.shape
print 'SHAPE PERTURBATIONS:'
print free_energy_mean_perturbations.shape
print 'SHAPE EPSILONS_PERTURBATION'
print epsilons_perturbation.shape
print 'AVERAGED ALONG INDIVIDUAL AXES:'
print perturbations_mean
print 'FREE ENERGY TIMECOURSE:'
print oFE_time
print 'PERTURBED FREE ENERGIES:'
print free_energy_mean_perturbations
print 'PARAMS SHAPE:'
print r_param.shape
print 'Diff:'
print cdiff.shape
print cdiff.min()
print cdiff.max()
print 'cFE_mean:'
print ocFE_mean
print 'cFE_time'
print ocFE_time


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
    delta_backprop = T.grad(cost = FE_mean_mean_perturbations, wrt = params[i])
    
    if i == 0:
        deltas = delta.flatten()
    else:
        deltas = T.concatenate([deltas, delta.flatten()], axis = 0 )
    
    if i == 0:
        deltas_backprop = delta_backprop.flatten()
    else:
        deltas_backprop = T.concatenate([deltas_backprop, delta_backprop.flatten()], axis = 0 )
    
    # USE ADAM OPTIMIZER
    #p_adam = Adam(delta, params[i], 0.9, 0.999, learning_rate, epsilon = 10e-6)
    #p_adam = Adam(delta_rank, params[i], 0.9, 0.999, learning_rate, epsilon = 10e-6)
    print 'Creating Adam object'
    p_adam = Adam(delta_backprop, params[i], 0.9, 0.999, learning_rate, epsilon = 10e-6)
    updates = updates + p_adam.updates
    
grad_corr = T.dot(deltas, deltas_backprop)/(deltas.norm(2)*deltas_backprop.norm(2))
    
# Add updates for sigmas

for i in range(len(sigmas)):
    
    print 'Creating updates for std dev of parameter %d...' % i
    
    print 'Calculating derivative'
    normalization = T.nnet.softplus( sigmas[i] ) + sig_min_perturbations
    outer_der = (r_epsilons[i]*r_epsilons[i]-1.0)/normalization
    inner_der = T.exp(sigmas[i])/(1.0 + T.exp(sigmas[i]))
    delta_sigma = T.tensordot(FE_mean_perturbations,outer_der*inner_der,axes = [[0],[0]])/n_perturbations
    delta_backprop_sigma = T.grad(cost = FE_mean_mean_perturbations, wrt = sigmas[i])
    
    if i == 0:
        deltas_sigma = delta_sigma.flatten()
    else:
        deltas_sigma = T.concatenate([deltas_sigma, delta_sigma.flatten()], axis = 0 )
    
    if i == 0:
        deltas_backprop_sigma = delta_backprop_sigma.flatten()
    else:
        deltas_backprop_sigma = T.concatenate([deltas_backprop_sigma, delta_backprop_sigma.flatten()], axis = 0 )
    
    # USE ADAM OPTIMIZER
    #p_adam = Adam(delta, params[i], 0.9, 0.999, learning_rate, epsilon = 10e-6)
    #p_adam = Adam(delta_rank, params[i], 0.9, 0.999, learning_rate, epsilon = 10e-6)
    print 'Creating Adam object'
    p_adam = Adam(delta_backprop_sigma, sigmas[i], 0.9, 0.999, learning_rate, epsilon = 10e-6)
    updates = updates + p_adam.updates
    
    # USE SIMPLE GRADIENT DESCENT
    #update = (params[i], params[i] - learning_rate*delta)
    #update = (params[i], params[i] - learning_rate*delta_rank)
    #update = (params[i], params[i] - learning_rate*delta_backprop)
    #updates.append(update)
    
grad_corr_sigma = T.dot(deltas_sigma, deltas_backprop_sigma)/(deltas_sigma.norm(2)*deltas_backprop_sigma.norm(2))
        
# Define Training Function
train = theano.function(
        inputs=[],
        outputs=[FE_mean, FE_mean_perturbations, grad_corr, grad_corr_sigma, deltas, deltas_backprop, deltas_sigma, deltas_backprop_sigma, param1, param2, tsigma1, tsigma2], 
        updates=updates,
        on_unused_input='ignore',
        allow_input_downcast = True
    )

########################################################################
#
# Run Optimization
#
########################################################################

[FE_min, oFE_mean_perturbations, ograd_corr, ograd_corr_sigma, odeltas, odeltas_backprop, odeltas_sigma, odeltas_backprop_sigma, oparam1, oparam2, osigma1, osigma2] = train()

print 'Initial FE:'
print [FE_min]

numpy.savetxt('initial_deltas.txt',odeltas)
numpy.savetxt('initial_deltas_backprop.txt',odeltas_backprop)
numpy.savetxt('initial_deltas_sigma.txt',odeltas_sigma)
numpy.savetxt('initial_deltas_backprop_sigma.txt',odeltas_backprop_sigma)

# Optimization Loop
for i in range(n_steps):
    
    #print 'Constraint weight:'
    #print constraint_weight.get_value()
    
    # Take the time for each loop
    start_time = timeit.default_timer()
    
    print 'Iteration: %d' % i    
    
    # Perform stochastic gradient descent using ADAM updates
    print 'Descending on Free Energy...'    
    [oFE_mean, oFE_mean_perturbations, ograd_corr, ograd_corr_sigma, odeltas, odeltas_backprop, odeltas_sigma, odeltas_backprop_sigma, oparam1, oparam2, osigma1, osigma2] = train()
    
    print 'Free Energy:'
    print [oFE_mean]
       
    print 'Correlation between gradients: %f' % ograd_corr   
    print 'Norms of Gradients: %f vs. %f' % (numpy.linalg.norm(odeltas), numpy.linalg.norm(odeltas_backprop))
    
    print 'Correlation between gradients for SIGMAS: %f' % ograd_corr_sigma   
    print 'Norms of Gradients: %f vs. %f' % (numpy.linalg.norm(odeltas_sigma), numpy.linalg.norm(odeltas_backprop_sigma))
    
    print 'Param1:'
    print oparam1
    
    print 'Param2:'
    print oparam2
    
    print 'Sigma1:'
    print osigma1
    
    print 'Sigma2:'
    print osigma2
       
    if i == 0:
        with open("log_evAI_mountaincar_minimal.txt", "w") as myfile:
            myfile.write("%f %f %f\n" % (oFE_mean, FE_min, ograd_corr))
    else:
        with open("log_evAI_mountaincar_minimal.txt", "a") as myfile:
            myfile.write("%f %f %f\n" % (oFE_mean, FE_min, ograd_corr))
    
    # Stop time
    end_time = timeit.default_timer()
    print 'Time for iteration: %f' % (end_time - start_time)
    
    # Save current parameters every nth loop
    if i % 100 == 0:
        with open('evAI_mountaincar_minimal.pkl', 'w') as f:
            cPickle.dump(params, f)
            
    # Save best parameters
    if oFE_mean < FE_min:
        FE_min = oFE_mean
        with open('evAI_mountaincar_minimal_best.pkl', 'w') as f:
            cPickle.dump(params, f)
        

# Save final parameters
with open('evAI_mountaincar_minimal.pkl', 'w') as f:
    cPickle.dump(params, f)



