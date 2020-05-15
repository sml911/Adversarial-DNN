"""
This tutorial shows how to generate adversarial examples using FGSM
and train a model using adversarial training with TensorFlow.
The original paper can be found at:
https://arxiv.org/abs/1412.6572
"""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals

import logging
import numpy as np
import tensorflow as tf
import json
import os
from copy import deepcopy,copy

from tensorflow.keras import backend as K
from tensorflow.keras.models import save_model
from cleverhans.utils_keras import KerasModelWrapper
import cifar10_Load_Model as c10load
import cifar10_resnet as c10r

from cleverhans.augmentation import random_horizontal_flip, random_shift
from cleverhans.compat import flags
from cleverhans.dataset import CIFAR10
from cleverhans.loss import CrossEntropy
from cleverhans.train import train
from cleverhans.utils import AccuracyReport, set_log_level
from cleverhans.utils_tf import model_eval
from advgan_utils import read_from_meta, write_to_meta, menu, get_new_key,custom_object
FLAGS = flags.FLAGS

NB_EPOCHS = 200
BATCH_SIZE = 128
LEARNING_RATE = 0.005
BACKPROP_THROUGH_ATTACK = False
ADV_TRAINING = False
TESTING = True
MODEL_KEY = 'model_3_0'
ATTACKER_KEYS = ['fgsm_b','pgd_a','advgan_4_a']


rng = np.random.RandomState([2017, 8, 30])
def cifar10_train_on_untargeted(train_start=0, train_end=60000, test_start=0,
                               test_end=10000, nb_epochs=NB_EPOCHS, batch_size=BATCH_SIZE,
                               learning_rate=LEARNING_RATE,
                               testing=True, adv_training=False,
                               backprop_through_attack=BACKPROP_THROUGH_ATTACK,
                               num_threads=None,threat_model='white_box',
                               model_key='model_1_a',attacker_key='clean',
                               label_smoothing=0.1):
  """
  CIFAR10 cleverhans training
  :param train_start: index of first training set example
  :param train_end: index of last training set example
  :param test_start: index of first test set example
  :param test_end: index of last test set example
  :param nb_epochs: number of epochs to train model
  :param batch_size: size of training batches
  :param learning_rate: learning rate for training
  :param clean_train: perform normal training on clean examples only
                      before performing adversarial training.
  :param testing: if true, complete an AccuracyReport for unit tests
                  to verify that performance is adequate
  :param backprop_through_attack: If True, backprop through adversarial
                                  example construction process during
                                  adversarial training.
  :param label_smoothing: float, amount of label smoothing for cross entropy
  :return: an AccuracyReport object
  """

  # Object used to keep track of (and return) key accuracies
  report = AccuracyReport()

  # Set TF random seed to improve reproducibility
  tf.set_random_seed(1234)

  # Set logging level to see debug information
  set_log_level(logging.DEBUG)

  # Create TF session
  if num_threads:
    config_args = dict(intra_op_parallelism_threads=1)
  else:
    config_args = {}
  sess = tf.Session(config=tf.ConfigProto(**config_args))

  K.set_learning_phase(0)

  ## Create TF session and set as Keras backend session
  K.set_session(sess)

  # Create a new model and train it to be robust to Attacker
  #keras_model = c10load.load_model(version=2,subtract_pixel_mean=True)
  meta = read_from_meta()
  attacker_meta = meta['attacker'][attacker_key]
  model_meta = meta['model'][model_key]
  attack_type = attacker_meta['attack_type']
  
  if threat_model == 'black_box_A':
    print('Using training set A')
    train_end = int(train_end/2)
    assert 'black_box_A' in meta['model'][model_key]['threat_models']
    dataset_section = 'A'
  elif threat_model == 'black_box_B':
    print('Using training set B')
    train_start = int(train_end/2)
    dataset_section = 'B'
    assert 'black_box_B' in meta['model'][model_key]['threat_models']
  elif threat_model == 'white_box':
    print('Using full training set')
    dataset_section = ''
  else:
    raise NotImplementedError

  # Get CIFAR10 data
  data = CIFAR10(train_start=train_start, train_end=train_end,
                 test_start=test_start, test_end=test_end)
  dataset_size = data.x_train.shape[0]
  dataset_train = data.to_tensorflow()[0]
  dataset_train = dataset_train.map(
      lambda x, y: (random_shift(random_horizontal_flip(x)), y), 4)
  dataset_train = dataset_train.batch(batch_size)
  dataset_train = dataset_train.prefetch(16)
  x_train, y_train = data.get_set('train')
  x_test, y_test = data.get_set('test')

  # Use Image Parameters
  img_rows, img_cols, nchannels = x_test.shape[1:4]
  nb_classes = y_test.shape[1]

  # Define input TF placeholder
  x = tf.placeholder(tf.float32, shape=(None, img_rows, img_cols,
                                        nchannels))
  y = tf.placeholder(tf.float32, shape=(None, nb_classes))


  attack_params = {}
  attack_params.update(meta['attacker']['default']['attack_params'])
  attack_params.update(attacker_meta['attack_params'])
  for k,v in attack_params.items():
    if isinstance(v,str):
      attack_params[k] = eval(v)
  if 'meta_key' in attacker_meta.keys() and attack_type == 'advgan':
    folderpath = meta['advgan'][attacker_meta['meta_key']]['train_params']['output_folder']
    attack_params.update({'generator_filepath':os.path.join(folderpath,'generator.hd5')})

  model_filename = model_meta['file_name']
  if 'black_box' in threat_model:
    model_filename = model_filename.replace('cifar10','cifar10B')
  model_filepath=model_meta['folder_path']+'/'+model_filename
  
  keras_model=tf.keras.models.load_model(
    filepath=model_filepath,
    custom_objects=custom_object())
  model = KerasModelWrapper(keras_model)

  def attack_statistics(x_true,x_adv):
    # calculate average L1,L2,Linf norms
    # as well as % of pixels modified
    L1 = tf.reduce_mean(K.sum(K.abs(x_adv-x_true),axis=(-1,-2,-3)))
    L2 = tf.reduce_mean(K.sqrt(K.sum(K.square(x_adv-x_true),axis=(-1,-2,-3))))
    
    Linf = tf.reduce_mean(K.max(K.abs(x_true-x_adv),axis=(-1,-2,-3)))
    eps = tf.constant(1/255,shape=x_true.shape.as_list()[1:])
    mod_perc = 100*tf.reduce_mean(K.cast(K.greater(K.abs(x_true-x_adv),eps),dtype='float'))
    return {'L1':L1,'L2':L2,'Linf':Linf,'%pix':mod_perc}

  def do_eval(preds, x_set, y_set, report_key, is_adv=None):
    eval_params = {'batch_size': batch_size}
    acc = model_eval(sess, x, y, preds, x_set, y_set, args=eval_params)
    setattr(report, report_key, acc)
    if is_adv is None:
      report_text = None
    elif is_adv:
      report_text = 'adversarial'
    else:
      report_text = 'legitimate'
    if report_text:
      print('Test accuracy on %s examples: %0.4f' % (report_text, acc))

  #define attacker
  if attack_type == 'cwl2':
    from cleverhans.attacks import CarliniWagnerL2
    attacker = CarliniWagnerL2(model, sess=sess)
  elif attack_type == 'fgsm':
    from cleverhans.attacks import FastGradientMethod
    attacker = FastGradientMethod(model, sess=sess)
  elif attack_type == 'pgd':
    from cleverhans.attacks import MadryEtAl
    attacker = MadryEtAl(model, sess=sess)
  elif attack_type == 'advgan':
    from cleverhans.attacks.adversarial_gan import AdvGAN
    attacker = AdvGAN(model,sess=sess)
  elif attack_type == None or attack_type=='clean':
    attacker = None
  else:
    print(attack_type+' is not a valid attack type')

  def attack(x):
    if attacker:
      print('attack_params',attack_params)
      return attacker.generate(x,**attack_params)
    else: 
      return x
  loss = CrossEntropy(model, smoothing=label_smoothing, attack=attack)
  preds = model.get_logits(x)
  adv_x = attack(x)

  if not backprop_through_attack:
    # For the fgsm attack used in this tutorial, the attack has zero
    # gradient so enabling this flag does not change the gradient.
    # For some other attacks, enabling this flag increases the cost of
    # training, but gives the defender the ability to anticipate how
    # the attacker will change their strategy in response to updates to
    # the defender's parameters.
    adv_x = tf.stop_gradient(adv_x)
  preds_adv = model.get_logits(adv_x)

  def evaluate():
    # Accuracy of adversarially trained model on legitimate test inputs
    do_eval(preds, x_test, y_test, 'adv_train_clean_eval', False)
    # Accuracy of the adversarially trained model on adversarial examples
    do_eval(preds_adv, x_test, y_test, 'adv_train_adv_eval', True)
  
  #print_attack info
  with sess.as_default():
    print('attack type: '+ attack_type)
    attack_stats = attack_statistics(x,adv_x)
    feed_dict={x:x_test[:batch_size],y:y_test[:batch_size]}
    attack_stats_eval = sess.run(attack_stats,feed_dict=feed_dict)
    attack_stats_eval = {k:str(v)[:10] for k,v in attack_stats_eval.items()}
    print(attack_stats_eval)

  if adv_training:
      # Train an CIFAR10 model
    reeval_breaks = 10
    train_params = {
        'batch_size': batch_size,
        'learning_rate': learning_rate
    }
    nb_e = nb_epochs
    prev_acc = 0
    # Perform and evaluate adversarial training
    for rb in range(reeval_breaks,0,-1):
      train_params.update({'nb_epochs': int(np.ceil(nb_e/rb))})
      if nb_e < train_params['nb_epochs'] < 0:
        train_params['nb_epochs'] = nb_e
      print("Starting training {} of {}".format(nb_epochs-nb_e, nb_epochs))
      train(sess, loss, None, None,
          dataset_train=dataset_train, dataset_size=dataset_size,
          evaluate=evaluate, args=train_params, rng=rng)

      nb_e-=train_params['nb_epochs'] 

      #put accuracies in dictionary fr json serializability 
      report_dict = {attr:str(getattr(report,attr))[:10] for attr in dir(report) 
                      if type(getattr(report,attr)) in [float,np.float32,np.float64]}
      print(report_dict)
      #save to meta
      new_meta = read_from_meta()
      new_model = deepcopy(model_meta)
      new_model.update({'adv_training':True,
                        'attacker_key':attacker_key,
                        'parent_key':model_key,
                        'threat_models':[threat_model],
                        'attack_stats':attack_stats_eval,
                        'report':report_dict,
                        'train_params': {
                          'batch_size': batch_size,
                          'learning_rate': learning_rate,
                          'nb_epochs': nb_epochs-nb_e,
                        },
                        'reeval':False
                       })
      if nb_e > 0:
        new_model.update({'training_finished':False,
          'file_name': model_meta['file_name'].replace('clean',attacker_key+'_train_epoch_'+str(new_model['train_params']['nb_epochs']))})
      else:
        new_model.update({'training_finished':True,
          'file_name': model_meta['file_name'].replace('clean',attacker_key+'_train')})

      new_model_key = get_new_key(model_key,meta)
      new_meta['model'].update({new_model_key:new_model})
      write_to_meta(new_meta)
      
      save_filename = new_model['file_name']
      if 'black_box' in threat_model:
        save_filename = save_filename.replace('cifar10','cifar10'+dataset_section) 
      save_model(keras_model,filepath=new_model['folder_path']+'/'+save_filename)

      if report.adv_train_adv_eval >= 0.9:
        break
      elif report.adv_train_adv_eval <= 0.01:
        #increase_lr
        lr = train_params['learning_rate']
        train_params.update({'learning_rate':lr*1.5})
        print('no learning! Increasing learning rate to {}'
          .format(train_params['learning_rate']))
        
      elif prev_acc<=report.adv_train_adv_eval:
        #update_lr
        lr = train_params['learning_rate']
        train_params.update({'learning_rate':lr*0.8})
        print('decreasing learning rate to {}'
          .format(train_params['learning_rate']))
      prev_acc = copy(report.adv_train_adv_eval)

      if nb_e<=0:
        break

  # Calculate training errors
  elif testing:
    do_eval(preds, x_train, y_train, 'train_adv_train_clean_eval')
    do_eval(preds_adv, x_train, y_train, 'train_adv_train_adv_eval')
    report_dict = {attr:str(getattr(report,attr))[:10] for attr in dir(report) 
                    if type(getattr(report,attr)) in [float,]}
    print('report_dict')
    print(report_dict)
  return report

def main(argv=None):
  
  meta=read_from_meta()

  for key,attacker in meta['attacker'].items():
    adv_training = FLAGS.adv_training
    testing = FLAGS.testing
    if (attacker['active'] == False) or (key not in FLAGS.attacker_keys):
      continue

    threat_model = attacker['threat_model']
    threat_models = meta['model'][FLAGS.model_key]['threat_models']
    parent_set = set(threat_models)
    
    if threat_model=='white_box':
      assert 'white_box' in parent_set
      train_set = 'white_box'
    else:
      train_set = parent_set.copy()

    for m_key,m in meta['model'].items():
      #skip any adversaries which have already been trained and do not require retraining
      if (m['parent_key'] == FLAGS.model_key) and (m['attacker_key']==key):
        if 'reeval' in m.keys():
          if m['reeval']==True:
            break
        if threat_model == 'white_box' and 'white_box' in m['threat_models']:
          adv_training = False
        elif threat_model == 'black_box':
          child_set = set(m['threat_models'])
          train_set.discard('white_box')
          train_set = train_set - child_set

    if not(adv_training or testing):
      continue
    for tm in train_set:
      print('running',tm) #for each threat model availible
      report=cifar10_train_on_untargeted(nb_epochs=FLAGS.nb_epochs, 
                     batch_size=FLAGS.batch_size, learning_rate=FLAGS.learning_rate, 
                     testing=testing, adv_training=adv_training,
                     model_key=FLAGS.model_key, attacker_key=key,threat_model=tm,
                     backprop_through_attack=FLAGS.backprop_through_attack)
if __name__ == '__main__':
  flags.DEFINE_integer('nb_epochs', NB_EPOCHS,
                       'Number of epochs to train model')
  flags.DEFINE_integer('batch_size', BATCH_SIZE,
                       'Size of training batches'                     'Learning rate for training')
  flags.DEFINE_bool('backprop_through_attack', BACKPROP_THROUGH_ATTACK,
                    ('If True, backprop through adversarial example '
                     'construction process during adversarial training'))
  flags.DEFINE_bool('adv_training', ADV_TRAINING,
                    'If True, train the classifier on the adversarial examples.')
  flags.DEFINE_bool('testing', TESTING,
                    'If True, test the trained classifier on the adversarial training examples.')
  flags.DEFINE_string('model_key', MODEL_KEY,
                    'model key for the model to be adversarially trained. See meta.json')
  flags.DEFINE_float('learning_rate',LEARNING_RATE,
                      'The starting learning rate for adversarial training')
  flags.DEFINE_list('attacker_keys',ATTACKER_KEYS,'list of attacker keys to train as defined in meta file')
  tf.app.run()
