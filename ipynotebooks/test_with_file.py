
# coding: utf-8

# In[1]:

get_ipython().magic(u'load_ext autoreload')


# In[2]:

get_ipython().magic(u'autoreload 2')


# In[3]:

from theano import *
from lasagne.layers import EmbeddingLayer, InputLayer, get_output
import lasagne
import lasagne.layers
import theano.tensor as T
import theano
import numpy as np


# In[4]:

theano.__version__


# In[5]:

l_in = lasagne.layers.InputLayer((1,50))


# In[6]:

l_hidden = lasagne.layers.DenseLayer(l_in, num_units=10, name='dens1')


# In[7]:

l_out = lasagne.layers.DenseLayer(l_hidden, num_units=1, name='dens2')


# In[8]:

out = lasagne.layers.get_output(l_out)


# In[9]:

print out


# In[10]:

target = T.ivector('y')
#loss = lasagne.objectives.binary_crossentropy(out, target).mean()
loss_vec = (out.reshape((out.size,)) - target.astype('float32')) ** 2


# In[11]:

all_params = lasagne.layers.get_all_params(l_out)


# In[12]:

update = lasagne.updates.adagrad(loss_vec.mean(), all_params, .01, .9)


# In[13]:

train = theano.function(
    [l_in.input_var, target], loss_vec, updates=update,
    allow_input_downcast=True,
    mode='DebugMode',
)


# In[14]:

fun_loss = theano.function([l_in.input_var, target], (out, target, loss_vec), mode='DebugMode')
#fun_out = theano.function([l_in.input_var], out, mode='DebugMode', on_unused_input='warn')


# In[15]:

target_v = (np.random.randint(0,2,1) * 2 - 1).astype('int32')


# In[16]:

target_t = np.random.randn(1, 50)


# In[17]:

fun_loss(target_t, target_v)


# In[18]:

train(target_t, target_v)


# In[19]:

res_out, res_target


# In[ ]:

res_out.shape, res_target.shape


# In[ ]:

theano.printing.debugprint(fun_loss)


# In[ ]:

target_t.shape


# In[ ]:

train(target_t, target_v)


# In[ ]:




# In[ ]:

eval_v = {l_in.input_var: target_t, target: target_v}
l_out.input_shape


# In[ ]:

[(a.shape.eval(), a) for a in all_params]


# In[ ]:

target_v.shape, target_t.shape


# In[ ]:

target_t.dot(all_params[0].get_value()).dot(all_params[2].get_value()) > 0


# In[ ]:

theano.printing.debugprint(train)


# In[ ]:

theano.config.DebugMode.


# In[ ]:

theano.__version__


# In[ ]:

from IPython.display import Image
res = theano.printing.pydotprint(train, outfile="functions/train.png")
Image("functions/train.png")


# In[20]:

updates_func = theano.function((l_in.input_var, target), update.values(), updates=update.items())


# In[ ]:

update.items()


# In[24]:

updates_func(target_t, target_v)


# In[ ]:
