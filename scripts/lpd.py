# this script implement LPD (similar to topic modeling) to uncovering latent metabolic funcitonal states
import os
import sys
from collections import OrderedDict
from copy import deepcopy
from time import time
#
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from sklearn.datasets import fetch_20newsgroups
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
import pandas as pd
#
import theano
import theano.tensor as tt
from theano import shared
from theano.sandbox.rng_mrg import MRG_RandomStream
#
import pymc3 as pm
from pymc3 import Dirichlet
from pymc3 import math as pmmath
from sklearn.decomposition import LatentDirichletAllocation

# environmental variables
os.environ["THEANO_FLAGS"]="device=cpu,floatX=float64"
#
plt.style.use("seaborn-darkgrid")
def logp_lda_doc(beta,theta):
    """Returns the log-likelihood function for given documents.
    K : number of functional states in the model
    N : number of features
    V : number of levels (size of vocabulary)
    D : number of time points (in a mini-batch)
    Parameters
    ----------
    beta : tensor (K x N x V)
        level distributions.
    theta : tensor (D x K)
        functional states distributions for samples.
    """
    def ll_docs_f(data):
        dixs,vixs=data.nonzero(data>=0)
        val=data[dixs,vixs]
        ll_docs=(val*pmmath.logsumexp(tt.log(theta[dixs])+tt.log(beta.T[vixs]), axis=1).ravel())
        # Per-levels log-likelihood times num of tokens in the whole dataset
        return tt.sum(ll_docs) / (tt.sum(vfreqs) + 1e-9) * n_tokens
    
    return ll_docs_f

class LDAEncoder:
    """Encode (term-frequency) document vectors to variational means and (log-transformed) stds."""

    def __init__(self, n_levels, n_hidden, n_topics, p_corruption=0, random_seed=1):
        rng = np.random.RandomState(random_seed)
        self.n_levels = n_levels
        self.n_hidden = n_hidden
        self.n_topics = n_topics
        self.w0 = shared(0.01 * rng.randn(n_levels, n_hidden).ravel(), name="w0")
        self.b0 = shared(0.01 * rng.randn(n_hidden), name="b0")
        self.w1 = shared(0.01 * rng.randn(n_hidden, 2 * (n_topics - 1)).ravel(), name="w1")
        self.b1 = shared(0.01 * rng.randn(2 * (n_topics - 1)), name="b1")
        self.rng = MRG_RandomStreams(seed=random_seed)
        self.p_corruption = p_corruption

    def encode(self, xs):
        if 0 < self.p_corruption:
            dixs, vixs = xs.nonzero()
            mask = tt.set_subtensor(
                tt.zeros_like(xs)[dixs, vixs],
                self.rng.binomial(size=dixs.shape, n=1, p=1 - self.p_corruption),
            )
            xs_ = xs * mask
        else:
            xs_ = xs

        w0 = self.w0.reshape((self.n_levels, self.n_hidden))
        w1 = self.w1.reshape((self.n_hidden, 2 * (self.n_topics - 1)))
        hs = tt.tanh(xs_.dot(w0) + self.b0)
        zs = hs.dot(w1) + self.b1
        zs_mean = zs[:, : (self.n_topics - 1)]
        zs_rho = zs[:, (self.n_topics - 1) :]
        return {"mu": zs_mean, "rho": zs_rho}

    def get_params(self):
        return [self.w0, self.b0, self.w1, self.b1]

def reduce_rate(a, h, i):
    s.set_value(η / ((i / minibatch_size) + 1) ** 0.7)

def print_top_words(beta, feature_names, n_top_words=10):
    for i in range(len(beta)):
        print(
            ("Topic #%d: " % i)
            + " ".join([feature_names[j] for j in beta[i].argsort()[: -n_top_words - 1 : -1]])
        )

def calc_pp(ws, thetas, beta, wix):
    """
    Parameters
    ----------
    ws: ndarray (N,)
        Number of times the held-out word appeared in N documents.
    thetas: ndarray, shape=(N, K)
        Topic distributions for N documents.
    beta: ndarray, shape=(K, V)
        Word distributions for K topics.
    wix: int
        Index of the held-out word

    Return
    ------
    Log probability of held-out words.
    """
    return ws * np.log(thetas.dot(beta[:, wix]))


def eval_lda(transform, beta, docs_te, wixs):
    """Evaluate LDA model by log predictive probability.

    Parameters
    ----------
    transform: Python function
        Transform document vectors to posterior mean of topic proportions.
    wixs: iterable of int
        Word indices to be held-out.
    """
    lpss = []
    docs_ = deepcopy(docs_te)
    thetass = []
    wss = []
    total_words = 0
    for wix in wixs:
        ws = docs_te[:, wix].ravel()
        if 0 < ws.sum():
            # Hold-out
            docs_[:, wix] = 0

            # Topic distributions
            thetas = transform(docs_)

            # Predictive log probability
            lpss.append(calc_pp(ws, thetas, beta, wix))

            docs_[:, wix] = ws
            thetass.append(thetas)
            wss.append(ws)
            total_words += ws.sum()
        else:
            thetass.append(None)
            wss.append(None)

    # Log-probability
    lp = np.sum(np.hstack(lpss)) / total_words

    return {"lp": lp, "thetass": thetass, "beta": beta, "wss": wss}

def transform_pymc3(docs):
    return sample_vi_theta(docs)

def transform_sklearn(docs):
    thetas = lda.transform(docs)
    return thetas / thetas.sum(axis=1)[:, np.newaxis]

# The number of words in the vocabulary
n_levels=3
#
print("Loading dataset...")
whl_data=pd.read_csv('/Users/yuewu/Dropbox (Edison_Lab@UGA)/Projects/Bioinformatics_modeling/spectral.related/lpd/data/input.csv',header=None)
whl_data_np=whl_data.to_numpy()
use_data=whl_data_np[1:52,]
# plt.plot(use_data[:10,:])
# plt.show()
n_samples_tr=40
n_samples_te=use_data.shape[0]-n_samples_tr
data_tr=use_data[:n_samples_tr,:]
data_te=use_data[n_samples_tr:,:]
print("Number of samples for training = {}".format(data_tr.shape[0]))
print("Number of samples for test = {}".format(data_te.shape[0]))
#
n_tokens=3
print(f"Number of tokens in training set = {n_tokens}")
#
n_states=5
minibatch_size=52#training set small so should be fine to use the whole set as the batch size

# defining minibatch
doc_t_minibatch = pm.Minibatch(docs_tr.toarray(), minibatch_size)
doc_t = shared(docs_tr.toarray()[:minibatch_size])
with pm.Model() as model:
    theta = Dirichlet(
        "theta",
        a=pm.floatX((1.0 / n_topics) * np.ones((minibatch_size, n_topics))),
        shape=(minibatch_size, n_topics),
        # do not forget scaling
        total_size=n_samples_tr,
    )
    beta = Dirichlet(
        "beta",
        a=pm.floatX((1.0 / n_topics) * np.ones((n_topics, n_levels))),
        shape=(n_topics, n_levels),
    )
    # Note, that we defined likelihood with scaling, so here we need no additional `total_size` kwarg
    doc = pm.DensityDist("doc",logp_lda_doc(beta,theta),observed=doc_t)

        
encoder = LDAEncoder(n_levels=n_levels, n_hidden=100, n_topics=n_topics, p_corruption=0.0)
local_RVs = OrderedDict([(theta, encoder.encode(doc_t))])
local_RVs

encoder_params = encoder.get_params()
encoder_params

η = 0.1
s = shared(η)
with model:
    approx = pm.MeanField(local_rv=local_RVs)
    approx.scale_cost_to_minibatch = False
    inference = pm.KLqp(approx)
inference.fit(
    10000,
    callbacks=[reduce_rate],
    obj_optimizer=pm.sgd(learning_rate=s),
    more_obj_params=encoder_params,
    total_grad_norm_constraint=200,
    more_replacements={doc_t: doc_t_minibatch},
)

plt.plot(approx.hist[10:]);

doc_t.set_value(docs_tr.toarray())
samples = pm.sample_approx(approx, draws=100)
beta_pymc3 = samples["beta"].mean(axis=0)

print_top_words(beta_pymc3, feature_names)

lda = LatentDirichletAllocation(
    n_components=n_topics,
    max_iter=5,
    learning_method="online",
    learning_offset=50.0,
    random_state=0,
)
%time lda.fit(docs_tr)
beta_sklearn = lda.components_ / lda.components_.sum(axis=1)[:, np.newaxis]

print_top_words(beta_sklearn, feature_names)


    
inp = tt.matrix(dtype="int64")
sample_vi_theta = theano.function(
    [inp], approx.sample_node(approx.model.theta, 100, more_replacements={doc_t: inp}).mean(0)
)



    
result_pymc3 = eval_lda(\
          transform_pymc3, beta_pymc3, docs_te.toarray(), np.arange(100)\
      )
print("Predictive log prob (pm3) = {}".format(result_pymc3["lp"]))


result_sklearn = eval_lda(\
          transform_sklearn, beta_sklearn, docs_te.toarray(), np.arange(100)\
      )
print("Predictive log prob (sklearn) = {}".format(result_sklearn["lp"]))
