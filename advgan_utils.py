import json
import sys
import os
import numpy as np
import tensorflow as tf
from tensorflow.keras.layers import Dense, BatchNormalization, Activation
from tensorflow.keras.layers import AveragePooling2D, Input, Flatten, Concatenate
from tensorflow.keras.layers import Dropout, Lambda, UpSampling2D, Conv2DTranspose
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Reshape, Layer, Multiply,Add
from tensorflow.keras.layers import LeakyReLU, BatchNormalization,Embedding
from tensorflow.keras.models import Sequential, Model
from tensorflow.keras.initializers import Constant
from tensorflow.keras.regularizers import l2
from tensorflow.keras import initializers
from tensorflow.keras import backend as K
from grid_report import shaded_data_table

class Logger(object):
  def __init__(self,folder,filename):
    self.terminal = sys.stdout
    self.log = open(folder+"/"+filename,"a")
    self.logging = True
  def write(self,message):
    self.terminal.write(message)
    if self.logging:
      self.log.write(message)
  def flush(self):
    pass
  def isatty(self):
    return sys.stderr.isatty()
  def set_logging(self,logging):
    self.logging = logging

def metric_convert(x,base_unit='s'):
  x=float(x)
  met_sym = ['n','u','m','','k','M','G']
  met_conv = {k:10**(3*v) for k,v in zip(met_sym,range(-3,4))} 
  for k,v in met_conv.items():
    if x < 1000*v and x >= v:
      return "%.3G" % (x/v) + k + base_unit
  return str(x) + base_unit

def deep_update(orig_dict, new_dict):
  import collections

  for key, val in new_dict.items():
    if isinstance(val, collections.Mapping):
      tmp = deep_update(orig_dict.get(key, {}), val)
      orig_dict[key] = tmp
    elif isinstance(val, list):
      orig_dict[key] = (orig_dict.get(key, []) + val)
    else:
      orig_dict[key] = new_dict[key]
  return orig_dict

def build_discriminator(D_arch=['C8','C16','C32','FC'],input_shape=(28,28,1),clip_min=0,clip_max=1,output_type="probability"):
  import re
  D = Sequential()

  D.add(Lambda(lambda x: K.clip(x,min_value=clip_min,max_value=clip_max),
              input_shape=input_shape))
  all_layers = re.compile('([A-Z]+)([0-9]*)')
  first_layer = True
  for L in D_arch:
    L = re.match(all_layers,L)
    assert L, 'Not a valid layer {}'.format(L)

    Ltype = L.group(1)
    Fcount = L.group(2)
    if Fcount == '':
      Fcount = 1
    else:
      Fcount = int(Fcount)
    if Ltype == 'C':
      D.add(Conv2D(Fcount,kernel_size=(4,4),
            kernel_initializer = 'glorot_normal',
            bias_initializer = 'zeros',
            padding = 'same'))
      if not first_layer:
        D.add(BatchNormalization())
      D.add(MaxPooling2D(pool_size=(2,2)))
      D.add(LeakyReLU(0.2))
    elif Ltype == 'FC':
      D.add(Flatten())
      if Fcount==1:
        if output_type=="probability":
          D.add(Dense(1,activation='sigmoid'))
        elif output_type=="unbounded":
          D.add(Dense(1))
        elif output_type=="tanh":
          D.add(Dense(1,activation='tanh'))
      else:
        D.add(Dense(Fcount))
        D.add(BatchNormalization())
        D.add(LeakyReLU(0.2))
    else:
      print("Unknown layer label {}".format(L.group(0)))
      quit()
    first_layer = False
  return D

def build_ATN(architecture=1, input_shape=[28,28,1], num_classes=10):
  if architecture == 0:
    image = Input(shape=input_shape)
    target = Input(shape=(num_classes,))
    #target_int = Lambda(lambda x:K.argmax(x,axis=-1))(target)
    x1 = Flatten()(image)
    #x2 = Embedding(10,20,input_length=1)(target_int)
    #x2 = Lambda(lambda x: K.squeeze(x, -2))(x2)
    x = Concatenate(axis=-1)([x1,target])
    x = Dense(2048,
              kernel_initializer='glorot_normal',
              bias_initializer='zeros')(x)
    x = BatchNormalization()(x)
    x = Activation('relu') (x)
    x = Dense(np.prod(input_shape),activation='sigmoid',
              bias_initializer='zeros')(x)
    x = Reshape(input_shape)(x)
    cnn = Model(inputs=[image, target], outputs=x)
  elif architecture == 1:
    image = Input(shape=input_shape)
    target = Input(shape=(num_classes,))
    x1 = Flatten()(image)
    x = Concatenate(axis=-1)([x1,target])
    x = Dense(1024,
              kernel_initializer='glorot_normal',
              bias_initializer='zeros')(x)
    x = BatchNormalization()(x)
    x = Activation('relu') (x)
    x = Dense(1024,
          kernel_initializer='glorot_normal',
          bias_initializer='zeros')(x)
    x = BatchNormalization()(x)
    x = Activation('relu')(x)
    x = Dense(np.prod(input_shape),activation='sigmoid',
              bias_initializer='zeros')(x)
    x = Reshape(input_shape)(x)
    cnn = Model(inputs=[image, target], outputs=x)
  elif architecture == -1:
    cnn = Sequential()
    cnn.add(Flatten(input_shape=input_shape))
    cnn.add(Dense(2048, activation='relu',
        kernel_initializer='glorot_normal',
        bias_initializer='zeros'))
    cnn.add(Dropout(0.25))
    cnn.add(Dense(np.prod(input_shape), activation='sigmoid',
        kernel_initializer='glorot_normal',
        bias_initializer='zeros'))
    cnn.add(Reshape(input_shape))
  elif architecture == -2:
    cnn = Sequential()
    cnn.add(Conv2D(64, kernel_size=(3, 3),
                 activation='relu',
                 kernel_initializer='glorot_normal',
                 bias_initializer='zeros',#Constant(-0.5),
                 kernel_regularizer=l2(0.005),
                 input_shape=input_shape,
                 padding='same'))
    cnn.add(Conv2D(128, kernel_size=(3, 3),
                 activation='relu',
                 kernel_initializer='glorot_normal',
                 bias_initializer='zeros',
                 kernel_regularizer=l2(0.005),
                 padding='same'))
    cnn.add(MaxPooling2D(pool_size=(2, 2)))

    cnn.add(Conv2D(256, kernel_size=(3, 3),
                 activation='relu',
                 kernel_initializer='glorot_normal',
                 bias_initializer='zeros',
                 kernel_regularizer=l2(0.005),
                 padding='same'))
    cnn.add(MaxPooling2D(pool_size=(2, 2)))
    cnn.add(Flatten())
    cnn.add(Dense(2048, activation='relu',
        kernel_initializer='glorot_normal',
        bias_initializer='zeros',
        kernel_regularizer=l2(0.05)))
    cnn.add(Dropout(0.25))
    cnn.add(Dense(np.prod(input_shape), activation='sigmoid',
        kernel_initializer='glorot_normal',
        bias_initializer='zeros'))
    cnn.add(Reshape(input_shape))
  elif architecture == 2:
    cnn = Sequential()
    cnn.add(Conv2D(256, kernel_size=(3, 3),activation='relu',input_shape=input_shape,padding='same',
              use_bias=True, kernel_initializer='glorot_normal', bias_initializer='zeros'))
    cnn.add(MaxPooling2D(pool_size=(2, 2)))
    cnn.add(Dropout(0.5))
    cnn.add(Conv2D(512, kernel_size=(3, 3),activation='relu',padding='same',
              use_bias=True, kernel_initializer='glorot_normal', bias_initializer='zeros'))
    #cnn.add(MaxPooling2D(pool_size=(2, 2)))
    #cnn.add(Dropout(0.5))
    #cnn.add(Conv2D(512, kernel_size=(3, 3),activation='relu',padding='same',
    #          use_bias=True, kernel_initializer='glorot_normal', bias_initializer='zeros'))
    #cnn.add(UpSampling2D(data_format='channels_last'))
    #cnn.add(Dropout(0.5))
    #cnn.add(Conv2DTranspose(256, kernel_size=(3,3), padding='same', activation='relu',
    #          use_bias=True, kernel_initializer='glorot_normal', bias_initializer='zeros'))
    cnn.add(UpSampling2D(data_format='channels_last'))
    cnn.add(Dropout(0.5))
    cnn.add(Conv2DTranspose(256, kernel_size=(3,3), padding='same', activation='relu',
              use_bias=True, kernel_initializer='glorot_normal', bias_initializer='zeros'))
    cnn.add(Dropout(0.5))
    cnn.add(Conv2DTranspose(1, kernel_size=(3,3), padding='same', activation='sigmoid',
              use_bias=True, kernel_initializer='glorot_normal', bias_initializer='zeros'))
  return cnn

def rerank(y, target, alpha):
  #y, as an input, is a tensor with shape (num vectors, num classes)
  #yp is the reranked output
  if alpha == np.inf:
    return target
  if type(y) == type(np.array([])):
    y_rr = alpha * target + y
    y_rr /= np.sum(y_rr,axis=-1,keepdims=True)
  else:
    y_rr = alpha*target + y
    y_rr = y_rr/K.sum(y_rr,axis=-1,keepdims=True)
  return y_rr
def L_X(x_true,x_pred):
  # define a distance measure between the original input
  # and the adversarial example. We use the L2 norm
  # inputs are (NxNx1)

  if type(x_true) == type(np.array([])):
    return np.sqrt(np.sum(np.square(x_true-x_pred),axis=(-1,-2,-3)))
  else:
    return K.sqrt(K.sum(K.square(x_true-x_pred),axis=(-1,-2,-3)))

def L_hinge(c,ord=2,elementwise=False):
  def L_h(x_true,x_pred):
    # define a distance measure with a hinge loss which is max(0,L_X-c)
    if ord == 1:
      if type(x_true) == type(np.array([])):
        if elementwise:
          return np.sum(np.maximum(np.zeros(x_true.shape[0]),np.abs(x_true-x_pred)-c),axis=(-1,-2,-3))          
        else:
          return np.maximum(np.zeros(x_true.shape[0]), np.sum(np.abs(x_true-x_pred),axis=(-1,-2,-3))-c)
      else:
        if elementwise:
          z = tf.zeros_like(x_true)
          return K.sum(K.maximum(z,K.abs(x_true-x_pred)-K.constant(c,dtype='float32')),axis=(-1,-2,-3))
        else:
          z = tf.zeros(tf.stack([tf.shape(x_true)[0],],0))
          return K.maximum(z,K.sum(K.abs(x_true-x_pred),axis=(-1,-2,-3))-K.constant(c,dtype='float32'))
    if ord == 2:
      if type(x_true) == type(np.array([])):
        if elementwise:
          return np.sqrt(np.sum(np.maximum(np.zeros(x_true.shape),np.square(x_true-x_pred)-c**2),axis=(-1,-2,-3)))
        else:
          return np.maximum(np.zeros(x_true.shape[0]), np.sqrt(np.sum(np.square(x_true-x_pred),axis=(-1,-2,-3)))-c)
      else:
        if elementwise:
          z = tf.zeros_like(x_true)
          return K.sqrt(K.sum(K.maximum(z,K.square(x_true-x_pred)-K.constant(c**2,dtype='float32')),axis=(-1,-2,-3)))
        else:
          z = tf.zeros(tf.stack([tf.shape(x_true)[0],],0))
          return K.maximum(z,K.sqrt(K.sum(K.square(x_true-x_pred),axis=(-1,-2,-3)))-K.constant(c,dtype='float32'))
    if ord == np.inf:
      if type(x_true) == type(np.array([])):
        return np.maximum(np.zeros(x_true.shape[0]), np.amax(x_true-x_pred,axis=(-1,-2,-3)-c))
      else:
        z = tf.zeros(tf.stack([tf.shape(x_true)[0],],0))
        return K.maximum(z,K.max(K.abs(x_true-x_pred),axis=(-1,-2,-3))-K.constant(c,dtype='float32'))
  return L_h

def L_X_sub(x_true,x_pred):
  # define a distance measure between the original input
  # and the adversarial example which only subtracts. We use the L2 norm
  # inputs are (NxNx1)

  if type(x_true) == type(np.array([])):
    return np.sqrt(np.sum(np.square(x_true-x_pred),axis=(-1,-2,-3)))
  else:
    return K.sqrt(K.sum(K.square(K.clip(x_pred-x_true,0,1))+0.2*K.square(K.clip(x_true-x_pred,0,1)),axis=(-1,-2,-3)))
def L_Y(y_true,y_pred):
  # define a distance measure between the original prediction
  # and the adversarial example prediction. We use the L2 norm
  # inputs are size (10)
  if type(y_true) == type(np.array([])):
    return np.sqrt(np.sum(np.square(y_true-y_pred),axis=-1))
  return K.sqrt(K.sum(K.square(y_true-y_pred),axis=-1))
def L_Y_CW(k):
  def l_y_cw(y_true,y_pred):
    if type(y_true) == type(np.array([])):
      diff=np.max(y_pred*(1-y_true),axis=-1)-K.max(y_pred*y_true,axis=-1)
      return np.max(diff,k)
    diff=K.max(y_pred*(K.constant(1,dtype='float32')-y_true),axis=-1)-K.max(y_pred*y_true,axis=-1)
    return K.max(diff,K.constant(k,dtype='float32'))

def L_Y_untargeted(y_true,y_pred):
  if type(y_true) == type(np.array([])):
    return np.sum(y_true*y_pred,axis=-1)
  return K.sum(y_true*y_pred,axis=-1)

# Discriminator output is the probability that the input example X is real
def L_GAN_ADV(y_true,y_pred):
  eps=1E-4
  return y_true*K.log(K.clip(y_pred,eps,1.-eps)) + (1-y_true)*K.log(1-K.clip(y_pred,eps,1.-eps))

def L_GAN_CE(y_true,y_pred):
  eps=1E-4
  return -y_true*K.log(K.clip(y_pred,eps,1.-eps)) - (1-y_true)*K.log(1-K.clip(y_pred,eps,1.-eps))
def L_LSGAN(y_true,y_pred):
  return (1/2)*K.square(y_pred-y_true)
def target_first(y_true,y_pred):
  #for use with find_scale
  if type(y_true) == type(np.array([])):
    return np.argmax(y_true,axis=-1)==np.argmax(y_pred,axis=-1)
  else:
    return K.mean(K.cast(K.equal(K.argmax(y_true,axis=-1),
        K.argmax(y_pred,axis=-1)),K.floatx()))

def orig_second_conditional(y_true, y_pred):
  _,ind_true = tf.nn.top_k(y_true,k=2,sorted=True)
  _,ind_pred = tf.nn.top_k(y_pred,k=2,sorted=True)
  s = target_first(y_true,y_pred)
  return tf.where(tf.less(s, 1e-7),s, K.mean(K.cast(K.equal(ind_true,ind_pred),K.floatx()))/s)

def orig_second_unconditional(y_true, y_pred):
  _,ind_true = tf.nn.top_k(y_true,k=2,sorted=True)
  _,ind_pred = tf.nn.top_k(y_pred,k=2,sorted=True)
  return K.mean(K.cast(K.equal(ind_true[1],ind_pred[1]),K.floatx()))
def find_scale(x_orig,x_fake,classifier,target=None,iterations=10):
  if x_orig.shape[0]==0:
    return np.array([])
  perturbation = x_fake-x_orig
  y_orig = np.argmax(classifier.predict(x_orig),axis=-1)
  x_expanded = np.expand_dims(x_orig,1)
  perturbation = np.expand_dims(perturbation,1)
  scales = []
  count = 0
  targeted = True
  if target==None:
    targeted = False
  for p,x,y in zip(perturbation,x_expanded,y_orig):
    count = count + 1
    it = iterations
    if y==target:
      scales.append(0.)
      continue
    #find the s(caling) required to fool the classifier
    s = 1
    while s < it + 1:
      scaled_x_class = np.argmax(classifier.predict_on_batch(np.clip(x+p*s,0,1)))
      if targeted: increase = scaled_x_class!=target
      else: increase = scaled_x_class==y
      if increase:
        s = s + 1
      else:
        break
    if s >= it + 1:
      scales.append(-1.)
      continue
    it = it - s
    i = 1
    s = float(s)
    while i < it + 1:
      scaled_x_class=np.argmax(classifier.predict_on_batch(np.clip(x+p*s,0,1)))
      if targeted: increase = scaled_x_class!=target
      else: increase = scaled_x_class==y

      if increase:
        s = s + 2**(-i)
      else:
        s = s - 2**(-i)
      if i == it:
        s = s + 2**(-i)

      i = i + 1
      scaled_x_class=np.argmax(classifier.predict_on_batch(np.clip(x+p*s,0,1)))
    if targeted: increase = scaled_x_class!=target
    else: increase = scaled_x_class==y
    if increase: s = -1.0
    scales.append(s)
  return np.asarray(scales)

def shuffle_in_unison(*args):
  rng_state = np.random.get_state()
  for arg in args:
    np.random.set_state(rng_state)
    np.random.shuffle(arg)

def read_from_meta(dataset='CIFAR10'):
  with open('meta.json','r') as fp:
    return json.load(fp)

def write_to_meta(meta, dataset='CIFAR10'):
  with open('meta.json','w') as fp:
    json.dump(meta,fp,indent=4,skipkeys=True)

def menu(choices):
  #present a list of choices (strings) to the user and return the choice
  #choices will typically be keys to a dictionary
  i = 1
  l=[]
  if not isinstance(choices, dict): 
    choices = {str(k):k for k in choices}
  for c in choices.keys():
      print('{}: {}'.format(i,c))
      i+=1
      l.append(c)
  choice = input('>>')
  if choice.isdigit():
      choice = int(choice)
      if choice in range(1,i):
        return choices[l[choice-1]]
  elif choice in choices.keys():
      return choices[choice]
  print('invalid choice: {}'.format(choice))


class Shift_Scale(Layer):

    def __init__(self,w,b, **kwargs):
        #output = w*(input+b)
        super(Shift_Scale, self).__init__(**kwargs)
        self.w = w
        self.b = b
        self.class_name = 'Shift_Scale'

    def build(self, input_shape):
        _input_shape = [1,]+input_shape.as_list()[1:]
        self.w_tf = K.constant(self.w,dtype='float32',shape=_input_shape)
        self.b_tf = K.constant(self.b,dtype='float32',shape=_input_shape)
        super(Shift_Scale, self).build(input_shape)  # Be sure to call this at the end
    def call(self, x):
        return (x + self.b_tf)*self.w_tf
    def get_output_shape(self,input_shape):
        return input_shape
    def get_config(self):
        config = {'w': self.w,'b': self.b}
        base_config = super(Shift_Scale, self).get_config()
        config = dict(list(base_config.items()) + list(config.items()))
        return config

class Subtract_Mean(Layer):

    def __init__(self, bias_initializer, **kwargs):
        super(Subtract_Mean, self).__init__(**kwargs)
        self.bias_initializer = initializers.get(bias_initializer)
        self.class_name = 'Subtract_Mean'
        #self.dtype = dtype

    def build(self, input_shape):

        self.bias = self.add_weight(
            name='bias',
            shape=input_shape[1:],
            initializer=self.bias_initializer,
            regularizer=None,
            constraint=None,
            dtype=self.dtype,
            trainable=False)
        super(Subtract_Mean, self).build(input_shape)  # Be sure to call this at the end

    def call(self, x):
        return tf.subtract(x, self.bias)

    def get_output_shape(self,input_shape):
        return input_shape

    def get_config(self):
        config = {
            'bias_initializer': initializers.serialize(self.bias_initializer),
        }
        base_config = super(Subtract_Mean, self).get_config()
        config = dict(list(base_config.items()) + list(config.items()))
        from tensorflow.python.keras.utils.generic_utils import deserialize_keras_object
        #make sure we can deserialize
        deserialize_keras_object(    
            config['bias_initializer'],
            module_objects=globals(),
            printable_module_name='initializer')
        return config

def custom_object(architecture='resnet'):
  #takes an architecture from model meta and return 
  # keras custom_objects for use with load_model  
  if "resnet" in architecture.lower():
    return {'Subtract_Mean':Subtract_Mean,
            'OuterProduct2D':OuterProduct2D,
            'artanh':artanh,
            'Shift_Scale':Shift_Scale,
            'clip01':clip01}
  else:
    return None
def add_border(x,thickness=4,color=(255,255,255)):
  #Inputs:
  #  x: image expressed as a numpy array with RGB channels or single channel
  #  thickness: integer thickness in pixels
  #  color: rgb tuple ()
  #Outputs:
  #  x_b: x wrapped with a border. If image was originally single channel, image is now 3 channel rgb  
  color = np.array(color,dtype=x.dtype).reshape((-1,1,1,3))
  if x.shape[-1] == 1:
    x = np.repeat(x,3,axis=-1)
  if len(x.shape) == 3:
    x = np.expand_dims(x,axis=0) 
  assert x.shape[-1]==3
  assert len(x.shape)==4
  border_top_bot = np.tile(color,(int(x.shape[0]/color.shape[0]),thickness,x.shape[2],1))
  border_left_right = np.tile(color,(int(x.shape[0]/color.shape[0]),x.shape[1]+2*thickness,thickness,1))
  x = np.concatenate((border_top_bot,x,border_top_bot),axis=1)
  x = np.concatenate((border_left_right,x,border_left_right),axis=2)
  if x.shape[0]==1:
    x = np.squeeze(x,0)
  return x
def test_add_border():
  from PIL import Image
  xs = (np.random.random((36,36,3)),
       np.random.random((56,34,1)),
       np.random.random((1000,56,34,1)),
       np.random.random((3,23,34,3)))

  fuscia = (255,51,255)
  royal_purple = (76,0,153)
  black = (255,255,255)
  colors = (fuscia,royal_purple,black,fuscia+royal_purple+black)
  thicknesses = (3,5,1,8)
  i = 0
  for x,color,thickness in zip(xs,colors,thicknesses):
    i+=1
    x = (x*255).astype(np.uint8)
    x_b = add_border(x,thickness,color)

    if len(x.shape) == 4:
      x = x.reshape((x.shape[0]*x.shape[1],)+(x.shape[2:]))
      x_b = x_b.reshape((x_b .shape[0]*x_b.shape[1],)+(x_b.shape[2:]))
    if x.shape[2]==1:
      x = np.squeeze(x,2)
    im = Image.fromarray(x)
    im_b = Image.fromarray(x_b)
    im.save('tmp/testcase{}.png'.format(i))
    im_b.save('tmp/testcase_b{}.png'.format(i))
def get_new_key(key,meta):
  tok = key.split('_')
  alphabet=[]
  
  alphabet=[chr(letter) for letter in range(97,123)]

  if tok[0]=='model':#                     1   2 3      
  # generate new model token               v   v v
  # if model 'key' input has 3 parts eg: model_1_a
  # the speified key is a parent model. Generate a
  # model key with the same number but a new letter
  # To generate a new number, input: key='model_new' 
  # and the function will return a new model key with
  # two parts eg: model_1_0 this is a parent model to
  # child 'model_1_?' where ? is a lowercase letter
    assert len(tok)==3 or tok[1] == 'new'
    _meta = meta['model']
    max_id = 0
    for k,v in _meta.items():
      tok_k=k.split('_',3)
      max_id = max(max_id,int(tok_k[1]))
      if tok[0]==tok_k[0] and tok[1]==tok_k[1]:
        if tok_k[2] in alphabet:
          alphabet.remove(tok_k[2])
        else:
          print('warning: identifier "{}" in key "{}" not in lowercase alphabet'
            .format(tok_k[2],k))
    if tok[1]=='new':
      return 'model_'+str(max_id+1)+'_0'
    return tok[0]+'_'+tok[1]+'_'+alphabet[0]
  elif tok[0]=='GAN':
    assert len(tok)==3
    _meta = meta['advgan']
    max_id = -1
    for k,v in _meta.items():
      tok_k=k.split('_',2)
      if tok[1]==tok_k[1]:
        max_id = max(max_id,int(tok_k[2]))
    return 'GAN_'+tok[1]+'_'+str(max_id+1).zfill(4)
  else:
    assert (tok[0]=='advgan' and len(tok)==3) or len(tok)==2
    _meta = meta['attacker']
    max_id = 0
    for k,v in _meta.items():
      tok_k=k.split('_',3)
      if tok[0]==tok_k[0]:
        if tok_k[1].isdigit():
          max_id = max(max_id,int(tok_k[1]))
        elif tok_k[1] in alphabet:
          alphabet.remove(tok_k[1])
        else:
          print('warning: identifier "{}" in key "{}" not in lowercase alphabet'
            .format(tok_k[1],k))
    if tok[1] == 'new':
      return tok[0]+'_'+str(max_id+1)+'_a'
    return tok[0]+'_'+alphabet[0]
def validate_meta(meta):

  for k in ['meta','model','attacker','advgan']:
    assert k in meta.keys()
  ks = []
  for k,v in meta['advgan'].items():
    assert k not in ks, 'duplicate advgan key error'
    ks.append(k)
def attack_statistics(x_true,x_adv):
  if type(x_true) == type(np.array([])):
    L1 = np.mean(np.sum(np.abs(x_adv-x_true),axis=(-1,-2,-3)))
    L2 = np.mean(np.sqrt(np.sum(np.square(x_true-x_pred),axis=(-1,-2,-3))))
    Linf = np.mean(np.max(np.abs(x_true-x_adv),axis=(-1,-2,-3)))
    eps = 1/256
    mod_perc = 100*np.mean(np.greater(np.abs(x_true-x_adv),eps).astype('float'))
    return {'L1':L1,'L2':L2,'Linf':Linf,'%pix':mod_perc}
  # calculate average L1,L2,Linf norms
  # as well as % of pixels modified
  L1 = tf.reduce_mean(K.sum(K.abs(x_adv-x_true),axis=(-1,-2,-3)))
  L2 = tf.reduce_mean(K.sqrt(K.sum(K.square(x_adv-x_true),axis=(-1,-2,-3))))
  Linf = tf.reduce_mean(K.max(K.abs(x_true-x_adv),axis=(-1,-2,-3)))
  eps = tf.constant(1/256,shape=x_true.shape.as_list()[1:])
  mod_perc = 100*tf.reduce_mean(K.cast(K.greater(K.abs(x_true-x_adv),eps),dtype='float'))
  return {'L1':L1,'L2':L2,'Linf':Linf,'%pix':mod_perc}

def resnet_block(inputs,
                 num_filters=16,
                 kernel_size=3,
                 strides=1,
                 transposed=False,
                 activation='relu',
                 conv_first=True):
  """2D Convolution-Batch Normalization-Activation stack builder

  # Arguments
      inputs (tensor): input tensor from input image or previous layer
      num_filters (int): Conv2D number of filters
      kernel_size (int): Conv2D square kernel dimensions
      strides (int): Conv2D square stride dimensions
      activation (string): activation name
      conv_first (bool): conv-bn-activation (True) or
          bn-activation-conv (False)

  # Returns
      x (tensor): tensor as input to the next layer
  """
  if conv_first:
    x = inputs
  else:
    x = BatchNormalization()(inputs)
    if activation:
      x = Activation('relu')(x)  
  if transposed:
    x = Conv2DTranspose(num_filters, 
                        kernel_size=kernel_size, 
                        padding='same',
                        strides=strides,
                        kernel_initializer='he_normal',
                        kernel_regularizer=l2(1e-4),
                        use_bias='False')(x)
  else:
    x = Conv2D(num_filters,
               kernel_size=kernel_size,
               strides=strides,
               padding='same',
               kernel_initializer='he_normal',
               kernel_regularizer=l2(1e-4),
               use_bias='False')(x)
  if conv_first:
    x = BatchNormalization()(x)
    if activation:
      x = Activation('relu')(x)  
  return x 
def totuple(a):
  try:
    return tuple(totuple(i) for i in a)
  except TypeError:
    return a

def rvs(dim=3):
  #generates orthogonal matrix of dimension dim
  random_state = np.random
  H = np.eye(dim)
  D = np.ones((dim,))
  for n in range(1, dim):
    x = random_state.normal(size=(dim-n+1,))
    D[n-1] = np.sign(x[0])
    x[0] -= D[n-1]*np.sqrt((x*x).sum())
    # Householder transformation
    Hx = (np.eye(dim-n+1) - 2.*np.outer(x, x)/(x*x).sum())
    mat = np.eye(dim)
    mat[n-1:, n-1:] = Hx
    H = np.dot(H, mat)
    # Fix the last sign such that the determinant is 1
  D[-1] = (-1)**(1-(dim % 2))*D.prod()
  # Equivalent to np.dot(np.diag(D), H) but faster, apparently
  H = (D*H.T).T
  return H

def outer_product_2D(inputs):
    """
    inputs: list of two tensors [x,y]
      x: (?,dx1,dx2,dx3)
      y: (?,dy1)
    output: 
      z: (?,dx1,dx2,dx3*dy1)
      z[:,:,:,dx*n:dx*n+1] = x*y[:,n]
    """

    x, y = inputs
    x_shape = K.shape(x)
    y_shape = K.shape(y)
    tf.assert_equal(tf.size(x_shape),4)
    tf.assert_equal(tf.size(y_shape),2)
    tf.assert_equal(x_shape[0],y_shape[0])
    batchSize = x_shape[0]
    y_static_size = y.shape.as_list()[1]
    output_shape = [-1]+x.shape.as_list()[1:]
    output_shape[3] *= y_static_size

    x = K.expand_dims(x,-2)
    y = K.reshape(y,(-1,1,1,y_static_size,1))

    outer_product = x * y
    outer_product = K.reshape(outer_product, output_shape)
    tf.assert_rank(outer_product,4)
    # returns a flattened batch-wise set of tensors
    return outer_product
class OuterProduct2D(Layer):
  def __init__(self, **kwargs):
      super(OuterProduct2D, self).__init__(**kwargs)
      self.class_name = 'OuterProduct2D'
      #self.dtype = dtype

  def call(self, x):
      return outer_product_2D(x)

  def get_output_shape(self,input_shape):
    if isinstance(input_shape, list):
      input_shape = [
        tuple(tensor_shape.TensorShape(x).as_list()) for x in input_shape]
    else:
      raise ValueError('A merge layer should be called ' 'on a list of inputs.')
    output_shape = input_shape[0][1:]
    output_shape *= input_shape[1][-1]
    return output_shape

class ClipLinf(Layer):
  def __init__(self,eps,**kwargs):
      super(ClipLinf, self).__init__(**kwargs)
      self.class_name = 'ClipLinf'
      self.eps = eps
      #self.dtype = dtype
  def call(self, inputs):
    x_adv, x_orig  = inputs
    x_adv_shape = K.shape(x_adv)
    x_orig_shape = K.shape(x_orig)
    tf.assert_equal(x_adv_shape,x_orig_shape)
    x = x_orig + K.clip(x_adv-x_orig,-self.eps,self.eps)
    return x
  def get_output_shape(self,input_shape):
    if isinstance(input_shape, list):
      input_shape = [
        tuple(tensor_shape.TensorShape(x).as_list()) for x in input_shape]
    else:
      raise ValueError('A merge layer should be called ' 'on a list of inputs.')
    #output_shape = input_shape[0][1:]
    return input_shape[0][1:]

def artanh(x):
  e = 10e-6
  return (1/2) * K.log((1+x)/(1+e-x))
def invsigmoid(x):
  e = 1/256
  return -K.log((e+1-x)/(e+x))
def clip01(x):
  return K.clip(x,0,1)

def G_resnet(n=1,block_strides=[1,2,2,1/2,1/2],input_shape=[32,32,3],num_classes=10,num_filters=16,inner_loop_concat=False):
  image = Input(shape=input_shape)
  target = Input(shape=(num_classes,))
  subtract_pixel_mean = True
  x = image

  #subract the pixel mean of the training set in the first layer
  if subtract_pixel_mean:
    x_train_mean = np.load('saved_models/cifar10_input_mean.npy')
    constant_init = Constant(value=totuple(x_train_mean))
    x=Subtract_Mean(bias_initializer=constant_init)(x)
  #  x = Concatenate(axis=-1)([target[:,i]*x for i in range(num_classes)])

  x = OuterProduct2D()([x,target])

  print('Building Generator: {} layer resnet'.format(2*n*len(block_strides)+2)) 

  x = resnet_block(inputs=x,
                   kernel_size=5,
                   num_filters=num_filters)
  # Instantiate convolutional base (stack of blocks).
  for i,m in enumerate(block_strides):
    for j in range(n):
      strides = 1
      is_first_layer_but_not_first_block = j == 0 and i > 0
      transposed = False
      if is_first_layer_but_not_first_block:
        strides = m
        if m<1:
          transposed = True
          strides = int(1/m)
        num_filters = int(m*num_filters)
      y = resnet_block(inputs=x,
                       num_filters=num_filters,
                       strides=strides,
                       transposed = transposed)
      if inner_loop_concat:
        y = OuterProduct2D()([y,target])
      y = resnet_block(inputs=y,
                       num_filters=num_filters,
                       activation=None)
      if is_first_layer_but_not_first_block:
        x = resnet_block(inputs=x,
                         num_filters=num_filters,
                         kernel_size=1,
                         strides=strides,
                         transposed = transposed,
                         activation=None)
      x = tf.keras.layers.add([x, y])
      x = Activation('relu')(x)
  x = OuterProduct2D()([x,target])
  x = resnet_block(inputs=x,
                   kernel_size=5,
                   num_filters=int(num_filters/2),
                   activation=None)
  x = OuterProduct2D()([x,target])
  x = Conv2D(3,
             kernel_size=1,
             padding='same',
             kernel_initializer='he_normal',
             bias_initializer='zeros')(x)

  #image_predist = Activation(invsigmoid)(image)
  #image_predist = Shift_Scale(w=4,b=-0.5)(image)
  x = Add()([x,image])
  output = Activation(clip01)(x)
  model = Model(inputs=[image, target], outputs=output,name='model_G')
  return model

def is_keras_loadable(model,model_arch):
  tmp_file = 'test.hd5'
  tf.keras.models.save_model(model,tmp_file)
  tf.keras.models.load_model(tmp_file,
    custom_objects=custom_object(model_arch))
  os.remove(tmp_file)
  return True

def build_generator_cifar(architecture='Dense3', input_shape=[32,32,3], num_classes=10):
  if architecture == 'ResNet12v1':
    model = G_resnet(n=1,num_filters=16)
  elif architecture == 'ResNet22v2':
    model = G_resnet(n=2,num_filters=16,inner_loop_concat=True)
  elif architecture == 'ResNet12v2':
    model = G_resnet(n=1,num_filters=16,inner_loop_concat=True)
  elif architecture == 'ResNet20Xiao':
    model = G_resnet(n=1,num_filters=8,block_strides=[1,2,2,1,1,1,1,1/2,1/2])
  elif architecture == 'Dense2':
    # 2 Dense Layers
    image = Input(shape=input_shape)
    target = Input(shape=(num_classes,))
    #target_int = Lambda(lambda x:K.argmax(x,axis=-1))(target)
    x1 = Flatten()(image)
    #x2 = Embedding(10,20,input_length=1)(target_int)
    #x2 = Lambda(lambda x: K.squeeze(x, -2))(x2)
    x = Concatenate(axis=-1)([x1,target])
    x = Dense(2048,
              kernel_initializer='glorot_normal',
              bias_initializer='zeros')(x)
    x = BatchNormalization()(x)    
    x = Activation('relu') (x)
    x = Dense(np.prod(input_shape),activation='sigmoid',
              bias_initializer='zeros')(x)
    x = Reshape(input_shape)(x)
    model = Model(inputs=[image, target], outputs=x,name='model_G')
  elif architecture == 'Dense3':
    # 3 Dense Layers
    image = Input(shape=input_shape)
    target = Input(shape=(num_classes,))
    x1 = Flatten()(image)
    x = Concatenate(axis=-1)([x1,target])
    print(x)
    x = Dense(1024,
              kernel_initializer='glorot_normal',
              bias_initializer='zeros')(x)
    x = BatchNormalization()(x)
    x = Activation('relu') (x)
    x = Dense(1024,
          kernel_initializer='glorot_normal',
          bias_initializer='zeros')(x)
    x = BatchNormalization()(x)
    x = Activation('relu')(x)
    x = Dense(np.prod(input_shape),activation='sigmoid',
              bias_initializer='zeros')(x)
    x = Reshape(input_shape)(x)
    model = Model(inputs=[image, target], outputs=x,name='model_G')
  elif architecture == 'Dense4':
    image = Input(shape=input_shape)
    target = Input(shape=(num_classes,))
    x1 = Flatten()(image)
    x = Concatenate(axis=-1)([x1,target])
    x = Dense(1024,
              kernel_initializer='glorot_normal',
              bias_initializer='zeros')(x)
    x = BatchNormalization()(x)
    x = Activation('relu') (x)
    x = Dense(1024,
          kernel_initializer='glorot_normal',
          bias_initializer='zeros')(x)
    x = BatchNormalization()(x)
    x = Activation('relu') (x)
    x = Dense(1024,
          kernel_initializer='glorot_normal',
          bias_initializer='zeros')(x)
    x = BatchNormalization()(x)
    x = Activation('relu')(x)
    x = Dense(np.prod(input_shape),activation='sigmoid',
              bias_initializer='zeros')(x)
    x = Reshape(input_shape)(x)
    model = Model(inputs=[image, target], outputs=x,name='model_G')
 
  print('Model Generator Name',model.name)
  if is_keras_loadable(model,architecture):
    return model