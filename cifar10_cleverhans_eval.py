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
import pickle
import os
import time
from copy import deepcopy,copy
import matplotlib
import matplotlib.pyplot as plt

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
from cleverhans.utils import set_log_level
from cleverhans.utils_tf import model_eval
from advgan_utils import read_from_meta, write_to_meta, menu, attack_statistics, deep_update, shuffle_in_unison, metric_convert, shaded_data_table,custom_object, dataset_filename_modifier
import cleverhans.attacks as cha 

FLAGS = flags.FLAGS

SWEEP_EPS = False
TARGETED = False
REEVAL = False
DATASET = 'CIFAR10'
CREATE_REPORTS = True
THREAT_MODEL = 'white_box'
MODEL_KEYS = ['model_2']
EVAL_MODEL_KEYS = ['model_3']
ATTACKER_KEYS = ['fgsm_b','pgd_a','advgan_c']
THREAT_MODEL = 'white_box'

rng = np.random.RandomState([2017, 8, 30])

def cifar10_eval_attacks(train_start=0, train_end=60000, test_start=0,
                         test_end=10000,
                         sweep_eps = SWEEP_EPS, targeted=TARGETED,
                         model_key='model_1_a',attacker_keys='clean',
                         eval_model_keys = None,threat_model='white_box',
                         generate_examples=True):
  """
  CIFAR10 cleverhans training
  :param train_start: index of first training set example
  :param train_end: index of last training set example
  :param test_start: index of first test set example
  :param test_end: index of last test set example
  :param model_key: name of the keras model to be loaded and tested
  :param attacker_key: name or list of names to be loaded 
                       and used to attack the model
 :return: an AccuracyReport object
  """

  if threat_model == 'white_box':
    eval_model_keys = [model_key,]
    attacker_partition = ''
    defender_partition = '' 
  if threat_model == 'black_box':
    attacker_partition = 'A'
    defender_partition = 'B'
    if not isinstance(eval_model_keys,list):
      raise ValueError('eval_model_keys must be list for black_box')
    #TODO: add white-box info to meta-data
    """     v<the eval model
        "model_1_g": {     v< the surrogate model
        "advgan_b->model_1_e": {
            "model_acc": "saved_models/model_1_cifar10_ResNet20_v2\\pickle\\model_1_g_advgan_b_model_acc.p",
            "target_acc": "saved_models/model_1_cifar10_ResNet20_v2\\pickle\\model_1_g_advgan_b_target_acc.p",
            "attack_stats": {
                "L1": 127.04542236328125,
                "L2": 2.9744277954101563,
                "Linf": 0.2539639711380005,
                "%pix": 93.39645385742188,
                "num_batches": 20,
                "time": "97.7us"
            "threat_model":"black_box"
    """



  # Set TF random seed to improve reproducibility
  tf.set_random_seed(1234)

  # Set logging level to see debug information
  set_log_level(logging.DEBUG)

  # Create TF session
  sess = tf.Session()

  K.set_learning_phase(0)

  ## Create TF session and set as Keras backend session
  K.set_session(sess)

  # Get CIFAR10 data
  data = CIFAR10(train_start=train_start, train_end=train_end,
                 test_start=test_start, test_end=test_end)
  dataset_size = data.x_train.shape[0]
  dataset_train = data.to_tensorflow()[0]
  #dataset_train = dataset_train.map(
  #    lambda x, y: (random_shift(random_horizontal_flip(x)), y), 4)
  #dataset_train = dataset_train.batch(batch_size)
  #dataset_train = dataset_train.prefetch(16)
  #x_train, y_train = data.get_set('train')
  x_test, y_test = data.get_set('test')
  #nb_train = x_train.shape[0]
  nb_test = x_test.shape[0]

  # Use Image Parameters
  img_rows, img_cols, nchannels = x_test.shape[1:4]
  nb_classes = y_test.shape[1]

  # Define input TF placeholder
  x = tf.placeholder(tf.float32, shape=(None, img_rows, img_cols,
                                        nchannels))
  y = tf.placeholder(tf.float32, shape=(None, nb_classes))
  y_target = tf.placeholder(tf.float32, shape=(None, nb_classes))

  meta = read_from_meta()
  model_meta = meta['model'][model_key]
  filename = model_meta['file_name'].replace('CIFAR10','CIFAR10'+attacker_partition)
  keras_model=tf.keras.models.load_model(
    filepath=model_meta['folder_path']+'/'+filename,
    custom_objects=custom_object())
  model = KerasModelWrapper(keras_model)

  attacker_keys = list(attacker_keys)
  report = dict()
  for attacker_key in attacker_keys:
    # Create a new model and train it to be robust to Attacker
    #keras_model = c10load.load_model(version=2,subtract_pixel_mean=True)
    attacker_meta = meta['attacker'][attacker_key]
    attack_type = attacker_meta['attack_type']
    attack_params = {}
    attack_params.update(meta['attacker']['default']['attack_params'])
    attack_params.update(attacker_meta['attack_params'])
    if 'spsa' in attacker_key:
      eval_par = {'batch_size':1}
    else:
      eval_par = {'batch_size': attack_params['batch_size']}
    for k,v in attack_params.items():
      if isinstance(v,str):
        attack_params[k] = eval(v)
    #define attacker

    if attack_type == 'advgan' or 'g+' in attack_type:
      if 'meta_key' in attacker_meta.keys():
        folderpath = meta['advgan'][attacker_meta['meta_key']]['train_params']['output_folder']
        attack_params.update({'generator_filepath':os.path.join(folderpath,'generator.hd5'),
                            'custom_objects':custom_object()})
      else:
        raise NotImplementedError("Must provide attacker meta with existing meta_key")

    standard_attackers = {'cwl2':cha.CarliniWagnerL2,
                          'fgsm':cha.FastGradientMethod,
                          'pgd':cha.MadryEtAl,
                          'jsma':cha.SaliencyMapMethod,
                          'stm':cha.SpatialTransformationMethod,
                          'advgan':cha.AdvGAN,
                          'spsa':cha.SPSA,
                          'g+pgd':cha.GanInformedPGD,
                          'g+spsa':cha.GanInformedSPSA
                          #'g+fgsm':cha.GanInformedFGM
                          }
    if attack_type in standard_attackers.keys():
      attacker = standard_attackers[attack_type](model, sess=sess)
    elif attack_type == None or attack_type=='clean':
      attacker = None
    else:
      print(attack_type+' is not a valid attack type')

    pkl_folderpath = os.path.join(model_meta['folder_path'],'pickle',attacker_key)
    if not os.path.isdir(pkl_folderpath):
        os.makedirs(pkl_folderpath)
########
    if targeted:
      # get target labels
      target_test = np.repeat(range(nb_classes),nb_test)
      x_test_shuf = np.array(np.tile(x_test,(nb_classes,1,1,1)))
      y_test_shuf = np.array(np.tile(y_test,(nb_classes,1)))
      y_target_test_shuf = tf.keras.utils.to_categorical(target_test,nb_classes)
      #do not shuffle
      #shuffle_in_unison(x_test_shuf,y_test_shuf,y_target_test_shuf)
      x_test_by_t_o = [[None]*nb_classes for n in range(nb_classes)]
      y_test_by_t_o = [[None]*nb_classes for n in range(nb_classes)]
      y_target_test_by_t_o = [[None]*nb_classes for n in range(nb_classes)]
      nb_test_by_t_o = np.zeros((nb_classes+1,nb_classes+1))
      print(y_target_test_shuf)
      for t in range(nb_classes):
        for o in range(nb_classes):
          if t==o:
            continue
          index = np.logical_and(y_target_test_shuf[:,t],y_test_shuf[:,o])
          nb_test_by_t_o[t,o] = np.count_nonzero(index)
          x_test_by_t_o[t][o] = x_test_shuf[index]
          
          y_test_by_t_o[t][o] = y_test_shuf[index]
          y_target_test_by_t_o[t][o] = y_target_test_shuf[index]
      np.testing.assert_array_equal(y_target_test_by_t_o[0][1], y_target_test_by_t_o[0][2], err_msg='', verbose=True)
      nb_test_by_t_o[nb_classes,:] = np.sum(nb_test_by_t_o,axis=0)
      nb_test_by_t_o[:,nb_classes] = np.sum(nb_test_by_t_o,axis=1)
      attack_params.update({'y_target':y_target})

      def model_eval_wrapper(preds, acc_target='original_class', adv_x=None):
        if acc_target == 'original_class':
          acc_target = y_test_by_t_o
        elif acc_target == 'target_class':
          acc_target = y_target_test_by_t_o
        else:
          raise ValueError('invalid value for accuracy_target: '+acc_target)
        accuracy_by_t_o = np.zeros((nb_classes+1,nb_classes+1))
        orig_accuracy_by_t_o = np.zeros((nb_classes+1,nb_classes+1))
        for t in range(nb_classes+1):
          for o in range(nb_classes):
            if t==o:
              continue
            row_scale = nb_test_by_t_o[t,o]/nb_test_by_t_o[t,nb_classes]
            col_scale = nb_test_by_t_o[t,o]/nb_test_by_t_o[nb_classes,o]
            if t<nb_classes:
              feed = {y_target:y_target_test_by_t_o[t][o][:eval_par['batch_size'],:]}
              if generate_examples:
                assert adv_x is not None, 'adv_x tensor must be supplied when generating examples'
                pickle_x_file = os.path.join(pkl_folderpath,pickle_file_head+"x_test_targeted_{}_{}.p".format(t,o))
                if os.path.exists(pickle_x_file):
                  adv_x_test = pickle.load( open( pickle_x_file, "rb" ) )
                else:
                  adv_x_test = gen_np(sess,x_test_by_t_o[t][o],x,adv_x,y_target_test_by_t_o[t][o],y_target)
                  pickle.dump(adv_x_test ,open( pickle_x_file, "wb" ) )
                
                accuracy_by_t_o[t,o] = model_eval(sess, adv_x, y, preds, 
                  adv_x_test, acc_target[t][o],args=eval_par)
                orig_accuracy_by_t_o[t,o] = model_eval(sess, adv_x, y, preds, 
                  x_test_by_t_o[t][o], acc_target[t][o],args=eval_par)
              else:
                accuracy_by_t_o[t,o] = model_eval(sess, x, y, preds, 
                  x_test_by_t_o[t][o], acc_target[t][o], feed=feed,args=eval_par)
              accuracy_by_t_o[nb_classes,o] += accuracy_by_t_o[t,o]*col_scale
              orig_accuracy_by_t_o[nb_classes,o] += orig_accuracy_by_t_o[t,o]*col_scale
            accuracy_by_t_o[t,nb_classes] += accuracy_by_t_o[t,o]*row_scale
            orig_accuracy_by_t_o[t,nb_classes] += orig_accuracy_by_t_o[t,o]*row_scale
        if adv_x is not None:
          # fill diagonal with original accuracies 
          for o in range(nb_classes):
            accuracy_by_t_o[o,o] = orig_accuracy_by_t_o[nb_classes,o]
        return accuracy_by_t_o   
    else:
      x_test_shuf = x_test
      y_test_shuf = y_test

    def attack(x,attack_params=attack_params):    
      if attacker:
        return attacker.generate(x,**attack_params)
      else: 
        return x
    def gen_np(sess,X,x,adv_x,Y_target=None,y_target=None):
      #inputs:
      #  sess (required) : tf session
      #  X (required) : numpy input data
      #  x (required) : placeholder for model input
      #  adv_x (required) : tensor for generator output
      #  Y_target (optional) : optional numpy array speccifying the target class
      #  y_target (optional) : optional placeholder for the target inputs
      #outputs:
      # 
      if attacker:
        with sess.as_default():
          _batch_size = eval_par['batch_size']
          nb_x = X.shape[0]
          nb_batches = int(np.ceil(float(nb_x) / _batch_size))
          assert nb_batches * _batch_size >= nb_x
          adv_x_np = np.zeros((0,)+X.shape[1:],dtype=X.dtype)
          for batch in range(nb_batches):
            start = batch * _batch_size
            end = min(nb_x, start + _batch_size)
            feed_dict = {x: X[start:end]}
            if not Y_target is None:
              feed_dict.update({y_target: Y_target[start:end]})
            adv_x_cur = adv_x.eval(feed_dict=feed_dict)
            adv_x_np = np.concatenate([adv_x_np,adv_x_cur],axis=0)
          assert end >= nb_x
          return adv_x_np
      else:
        return x

    def attack_stats_eval(x,adv_x,num_batches=1):
      # Return attack info
      with sess.as_default():
        _batch_size = eval_par['batch_size']
        _as_eval = dict()
        cum_time = 0.
        attack_stats = attack_statistics(x,adv_x)
        for batch in range(num_batches):
          feed_dict={x:x_test_shuf[batch*_batch_size:(batch+1)*_batch_size],
                     y:y_test_shuf[batch*_batch_size:(batch+1)*_batch_size]}
          if targeted:
            feed_dict.update({y_target:y_target_test_shuf[batch*_batch_size:(batch+1)*_batch_size]})
          _as = sess.run(attack_stats,feed_dict=feed_dict)

          if batch == 0:
            _as_eval = deepcopy(_as)
          else:
            _as_eval = {k:v+_as[k] for k,v in _as_eval.items()} 

          t_1 = time.process_time()
          adv_x.eval(feed_dict=feed_dict)
          t_2 = time.process_time()
          cum_time += t_2-t_1
      cum_time /= num_batches*_batch_size     

      _as_eval = {k:v/num_batches for k,v in _as_eval.items()}
      _as_eval.update({'num_batches':num_batches,
                      'time':metric_convert(cum_time,'s')})
      return _as_eval
    
    report.update({attacker_key:{'model_acc':{}}})

    for eval_model_key in eval_model_keys:
      #Sweep over models to evaluate on. "White Box" attacks
      #only have one eval_model_key "Black Box" attack may 
      #have several eval_model_key "defenses"
      report_view = report[attacker_key]

      if threat_model == 'white_box':
        assert model_key == eval_model_key,('for white_box attacks, ',
          'generating model and eval model must be the same')
        eval_model = model
      elif threat_model == 'black_box':
        #add black box eval model to report and update report head
        if not 'black_box' in report_view.keys(): 
          report_view.update({'black_box':{eval_model_key:{'model_acc':{}}}})
        else:
          report_view['black_box'].update({eval_model_key:{'model_acc':{}}})
        report_view = report_view['black_box'][eval_model_key]

        #load eval model trained on defense dataset
        eval_model_meta = meta['model'][eval_model_key]
        filename = eval_model_meta['file_name'].replace('CIFAR10','CIFAR10'+defender_partition)
        keras_model=tf.keras.models.load_model(
          filepath=eval_model_meta['folder_path']+'/'+filename,
          custom_objects=custom_object())
        eval_model = KerasModelWrapper(keras_model)

      #evaluate model on clean examples
      preds = eval_model.get_logits(x)
      model_acc = model_eval(sess, x, y, preds, 
                x_test, y_test,args=eval_par)
      print('Test accuracy on clean examples %0.4f\n' % model_acc)
      report_view.update({'clean_model_acc':model_acc})

      t1 = 0
      #sweep epsilon
      if sweep_eps and attack_type!='clean':
        max_eps = 2*attack_params['eps']
        if 'eps_iter' in attack_params.keys():
          max_eps_iter = 2*attack_params['eps_iter']
        epsilons = np.linspace(1/255, max_eps, min(int(max_eps*255),16))
        sweep_e = dict()
        for e in epsilons:
          scaled_e = str(int(e*255))
          t1 = time.time()
          attack_params.update({'eps': e})
          if 'eps_iter' in attack_params.keys():
            attack_params.update({'eps_iter': max_eps_iter*e/max_eps})
          adv_x = attack(x, attack_params)
          attack_stats_cur = attack_stats_eval(x,adv_x,1)
          preds_adv = eval_model.get_probs(adv_x)
          if targeted:
            model_acc = model_eval_wrapper(preds_adv,acc_target='original_class',adv_x=adv_x)
            target_acc = model_eval_wrapper(preds_adv,acc_target='target_class',adv_x=adx_x)
            pickle_file_head = '{}_{}_{}_'.format(model_key,attacker_key,e)
            pickle_m_file=os.path.join(pkl_folderpath,pickle_file_head+"model_acc.p")
            pickle_t_file=os.path.join(pkl_folderpath,pickle_file_head+"target_acc.p")
            pickle.dump( model_acc, open( pickle_m_file, "wb" ) )
            pickle.dump( target_acc, open( pickle_t_file, "wb" ) )
            sweep_e.update({scaled_e:
                              {'model_acc': pickle_m_file,
                               'target_acc': pickle_t_file,
                               'attack_stats':attack_stats_cur}})      
          else:
            if generate_examples:
              pickle_x_file = os.path.join(pkl_folderpath,pickle_file_head+"x_test_untargeted.p")
              if os.path.exists(pickle_x_file):
                adv_x_test = pickle.load( open( pickle_x_file, "rb" ) )
              else:
                adv_x_test = gen_np(sess,x_test,x,adv_x)
                pickle.dump(adv_x_test ,open( pickle_x_file, "wb" ) )
              model_acc = model_eval(sess, adv_x, y, preds, 
                adv_x_test, y_test, args=eval_par)
            else:
              model_acc = model_eval(sess, x, y, preds, 
                      x_test, y_test, args=eval_par)
            sweep_e.update({scaled_e:
                              {'model_acc': model_acc,
                              'attack_stats':attack_stats_cur}})
          print('Epsilon %.2f, accuracy on adversarial' % e,
                'examples %0.4f\n' % model_acc)
          print(sweep_e[scaled_e])
        report_view.update({'sweep_eps':sweep_e})
        t2 = time.time()
      else:
        if 'eps' in attack_params:
          cond_eps=attack_params['eps']
        else:
          cond_eps='N/A'
        print('evaluating {}->{} examples on {} (single epsilon: {})'.format(
          attacker_key,model_key,eval_model_key,cond_eps))

        t1 = time.time()
        adv_x = attack(x, attack_params)
        preds_adv = eval_model.get_probs(adv_x)
        pickle_file_head = '{}_{}_'.format(model_key,attacker_key)
        if targeted:
          model_acc = model_eval_wrapper(preds_adv,acc_target='original_class',adv_x=adv_x)
          target_acc = model_eval_wrapper(preds_adv,acc_target='target_class',adv_x=adv_x)
          
          if threat_model == 'black_box':
            pickle_m_file=os.path.join(pkl_folderpath,pickle_file_head+eval_model_key+"_model_acc.p")
            pickle_t_file=os.path.join(pkl_folderpath,pickle_file_head+eval_model_key+"_target_acc.p")
          else:
            pickle_m_file=os.path.join(pkl_folderpath,pickle_file_head+"_model_acc.p")
            pickle_t_file=os.path.join(pkl_folderpath,pickle_file_head+"_target_acc.p")
          pickle.dump( model_acc, open( pickle_m_file, "wb" ) )
          pickle.dump( target_acc, open( pickle_t_file, "wb" ) )
          report_view.update({'model_acc': pickle_m_file,
                              'target_acc': pickle_t_file,
                              'attack_stats':attack_stats_eval(x,adv_x,20)})      
        else:
          if generate_examples:
            pickle_x_file = os.path.join(pkl_folderpath,pickle_file_head+"x_test_untargeted.p")
            if os.path.exists(pickle_x_file):
              adv_x_test = pickle.load( open( pickle_x_file, "rb" ) )
            else:
              adv_x_test = gen_np(sess,x_test,x,adv_x)
              pickle.dump(adv_x_test ,open( pickle_x_file, "wb" ) )
            #evaluate on self and, if black box, all other eval models
            model_acc = model_eval(sess, adv_x, y, preds_adv, 
              adv_x_test, y_test, args=eval_par)
          else:
            model_acc = model_eval(sess, x, y, preds_adv, 
                          x_test, y_test, args=eval_par)
          report_view.update({'model_acc': model_acc,
                              'attack_stats':attack_stats_eval(x,adv_x,20)})
        t2 = time.time()
        if targeted:
          print('Test accuracy on adversarial examples %0.4f\n' % model_acc[nb_classes,nb_classes])
          print('Target accuracy on adversarial examples %0.4f\n' % target_acc[nb_classes,nb_classes])
        else:
          print('Test accuracy on adversarial examples %0.4f\n' % model_acc)

      print("Took", t2 - t1, "seconds")
  return report
def create_reports(directory,targeted=True):
  if targeted:
    # generate shaded box graphs from targeted attack
    # accuracy matrices
    # directory is the model directory in saved_models 
    print('creating targeted attack accuracy matrices at '+directory)
    color_limits = [0,1]
    colors = [(0.2,0.2,1),(0.3,0.3,0.3),(1,0.2,0.2),(0.8,0.8,0.8),(0.2,1,0.2)]
    if FLAGS.dataset == 'CIFAR10':
      ltran = np.array(['airplane','automobile','bird','cat','deer',
              'dog','frog','horse','ship','truck'])
    labels = [l[:8] for l in ltran]
    labels.append('total')
    pickle_dir = os.path.join(directory,'pickle')
    if not os.path.exists(pickle_dir):
      print('none found')
      return
    print('looking in {}'.format(pickle_dir))
    for foldername in os.listdir(pickle_dir):
      new_pickle_dir=os.path.join(pickle_dir,foldername)
      if os.path.isdir(new_pickle_dir):
        for filename in os.listdir(new_pickle_dir):
          if filename.endswith("acc.p"): 
            print(filename)
            target_matrix = pickle.load(open(os.path.join(new_pickle_dir,filename),'rb'))
            shaded_data_table(target_matrix,labels,['target','original'],
                              colors=colors,color_limits=color_limits)
            no_ext = filename.replace('.p','')
            plt.title(no_ext.replace('_',' '))
            print('saving {}.png'.format(no_ext))
            plt.savefig(os.path.join(new_pickle_dir,no_ext+'.png'))
            plt.close()
def main(argv=None):
  print('sweep_eps: {} ({})'.format(FLAGS.sweep_eps,type(FLAGS.sweep_eps)))
  print('targeted: {} ({})'.format(FLAGS.targeted,type(FLAGS.targeted)))
  print('reeval: {} ({})'.format(FLAGS.reeval,type(FLAGS.reeval)))
  print('create_reports: {} ({})'.format(FLAGS.create_reports,type(FLAGS.create_reports)))
  print('attacker_keys: {} ({})'.format(FLAGS.attacker_keys,type(FLAGS.attacker_keys)))
  print('model_keys: {} ({})'.format(FLAGS.model_keys,type(FLAGS.model_keys)))
  print('eval_model_keys: {} ({})'.format(FLAGS.eval_model_keys,type(FLAGS.eval_model_keys)))

  meta=read_from_meta()
  #load report for updating
  required_keys = ['model_acc','attack_stats']
  if FLAGS.sweep_eps:
    required_keys.append('sweep_eps')
  if FLAGS.targeted:
    file_name = 'targeted_attack_report.json'
    required_keys.append('sweep_target')
  else:
    file_name = 'untargeted_attack_report.json'
  for key,model in meta['model'].items():

    # evaluates model if it is in model_keys
    # softly fails when model_key does not exist
    # can specify subset of a model key to evaluate
    # many models eg. model_keys = ['model_1'] will 
    # evaluate model_1_a, model_1_b, etc.
    cont = False
    for model_key in FLAGS.model_keys:
      if model_key in key:
        cont = True
    if cont == False:
      continue
    if FLAGS.threat_model=='black_box':
      if not 'black_box_A' in model['threat_models']:
        print('black_box_A (attacker dataset) not supported',
              'for surrogate model candidate:',model_key)
        continue
      eval_model_keys = []
      for ekey,emodel in meta['model'].items():
        cont = False
        for eval_model_key in FLAGS.eval_model_keys:
          if eval_model_key in ekey and eval_model_key!=model_key:
            cont = True
        if cont == False:
          continue
        if not 'black_box_B' in emodel['threat_models']:
          print('black_box_A (attacker dataset) not supported',
          'for surrogate model candidate:',model_key)
          continue
        eval_model_keys.append(ekey)
      if not eval_model_keys:
        print('no valid black box model candidates for surrogate model:', model_key)
        continue
      print('evaluating black box models:',eval_model_keys)
    else:
      eval_model_keys = None
      if not 'white_box' in model['threat_models']:
        print('white_box not supported for model candidate:',model_key)
        continue
    report_path = os.path.join(model['folder_path'],file_name)
    if os.path.isfile(report_path):
      with open(report_path,'r') as fp:
        report_update = json.load(fp)
    else:
      report_update = dict()    

    attacker_keys = []
    if key in report_update.keys():
      print(key)

      #do not update attackers that have already been computed
      for attacker_key in FLAGS.attacker_keys:
        if FLAGS.reeval:
          acomp = True
        else:
          acomp = False
          print(required_keys)
          print(report_update[key].keys())
          if attacker_key in report_update[key].keys():
            for required_key in required_keys:
              print(required_key)
              if required_key not in report_update[key][attacker_key].keys():
                acomp = True
                print(required_key +' not in report')
          else:
            acomp = True
        if acomp:
          attacker_keys.append(attacker_key)
          print('computing ' + attacker_key + ' accuracy')
        else:
          print('skipping ' + attacker_key)

    report=cifar10_eval_attacks(
      targeted=FLAGS.targeted, sweep_eps=FLAGS.sweep_eps,
      model_key=key, eval_model_keys=eval_model_keys,
      attacker_keys=attacker_keys,threat_model=FLAGS.threat_model,
      generate_examples=True)

    with open(report_path,'w') as fp:
      deep_update(report_update,{key:report})
      json.dump(report_update,fp,indent=4,skipkeys=True)
    if FLAGS.create_reports:
      create_reports(model['folder_path'],targeted=FLAGS.targeted)
if __name__ == '__main__':
  flags.DEFINE_bool('sweep_eps',SWEEP_EPS,'Flag to sweep epsilon in evaluation')
  flags.DEFINE_bool('targeted',TARGETED,'Flag for targeted attack')
  flags.DEFINE_bool('reeval',REEVAL, ('Flag for reevaluating all attackers'
                      'regardless of whether they have been previously computed'))
  flags.DEFINE_string('dataset',DATASET,'flag for dataset to eval. Options are CIFAR10')
  flags.DEFINE_bool('create_reports',CREATE_REPORTS,'Flag whether to create reports')
  flags.DEFINE_list('attacker_keys',ATTACKER_KEYS,'list of attacker keys to evaluate as defined in meta file')
  flags.DEFINE_list('model_keys',MODEL_KEYS,'list of model keys to evaluate as defined in meta file'
                     'If threat_model is "black_box" this is a list of surrogate models')
  flags.DEFINE_list('eval_model_keys',EVAL_MODEL_KEYS,"if threat_model is 'black_box' "
                    "evaluate adversarial examples on each compatable model in this list."
                    "These are your black box models")
  flags.DEFINE_string('threat_model',THREAT_MODEL,'attack-defense threat model. Choices are "white_box" or "black_box"')
  tf.app.run()
