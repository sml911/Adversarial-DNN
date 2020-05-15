import os
#os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2' 
import tensorflow as tf
from copy import deepcopy
import warnings
import numpy as np
from PIL import Image
import pprint
import matplotlib.pyplot as plt
try:
    import cPickle as pickle
except ImportError:
    import pickle
from collections import defaultdict
from tensorflow.keras.models import Model, load_model, save_model
from tensorflow.keras.datasets import cifar10
import cifar10_resnet as c10r
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import ModelCheckpoint, LearningRateScheduler
from tensorflow.keras.callbacks import ReduceLROnPlateau
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.regularizers import l2
from tensorflow.keras import backend as K
import cifar10_Load_Model as c10lm
from advgan_utils import *

#Logger, rerank, L_X, L_hinge, L_Y, 
#L_GAN_ADV, L_GAN_CE, L_LSGAN,target_first,
#orig_second_conditional, orig_second_unconditional,
#find_scale, shuffle_in_unison

flags = tf.app.flags
FLAGS = flags.FLAGS
def lr_schedule(epoch):
    #Learning Rate Schedule
    lr = 1e-4
    if epoch > 80:
        lr *= 0.5e-1
    elif epoch > 60:
        lr *= 1e-1
    elif epoch > 20:
        lr *= 1e-1
    print('Learning rate: ', lr)
    return lr
def train_AdvGAN(x_loss,alpha,beta,c_hinge,rr_alpha,
                batch_size,epochs,d_loss,train_gan,
                g_arch,d_arch,model_key,dataset,
                output_folder, targeted=True,
                parent_key=None,y_loss='L2',
                clip_eps=None,clip_ord=None,
                threat_model='white_box',
                distillation_method=None,
                black_box_model_key=None,
                **kwargs):

  if dataset == 'CIFAR10':
    data_augmentation = True
    num_classes = 10
    # Subtracting pixel mean improves accuracy
    subtract_pixel_mean = False
    # Load the CIFAR10 data.
    (x_train, y_train), (x_test, y_test) = cifar10.load_data()
    if threat_model=='black_box':
      #attacker sees dataset A (first half)
      n_samp_half = int(x_train.shape[0]/2)
      x_train = x_train[:n_samp_half]
      y_train = y_train[:n_samp_half]
    ltran = np.array(['airplane','automobile','bird','cat','deer',
            'dog','frog','horse','ship','truck'])

  elif dataset == 'MNIST':
    quit('not_implemented')
  if FLAGS.debug:
    epochs = 3
    nm_qk = batch_size*10
    (x_test, y_test) = (x_test[:nm_qk], y_test[:nm_qk])
    (x_train, y_train) = (x_train[:nm_qk], y_train[:nm_qk])
  if 'L1' in x_loss:
    ord_x = 1
  elif 'L2' in x_loss:
    ord_x = 2
  elif 'Linf' in x_loss:
    ord_x = np.inf
  else:
    raise ValueError('invalid x_loss')
  if 'elem' in x_loss:
    elementwise_x = True
  else:
    elementwise_x = False
  L_X_advgan = L_hinge(c_hinge,ord=ord_x,elementwise=elementwise_x)
  if targeted:
    if y_loss=='L2':
      L_Y_advgan = L_Y
    if y_loss=='CW':
      L_Y_advgan = L_Y_CW(k=0)

  targets = range(num_classes)

  # Input image dimensions.
  # We assume data format "channels_last".
  img_rows = x_train.shape[1]
  img_cols = x_train.shape[2]
  channels = x_train.shape[3]
  input_shape = (img_rows, img_cols, channels)

  # Normalize data.
  x_train = x_train.astype('float32') / 255
  x_test = x_test.astype('float32') / 255

  num_train, num_test = x_train.shape[0], x_test.shape[0]
 
  # Convert class vectors to binary class matrices.
  y_train = tf.keras.utils.to_categorical(y_train, num_classes)
  y_test = tf.keras.utils.to_categorical(y_test, num_classes)

  print('Dataset:', dataset)
  print('x_train shape:', x_train.shape)
  print(num_train, 'train samples')
  print(num_test, 'test samples')
  print('y_train shape:', y_train.shape)


  # Prepare model saving directory.
  #  save_dir = os.path.join(os.getcwd(),'saved_models',dataset)
  #   if not os.path.isdir(save_dir):
  #      os.makedirs(save_dir)

  GAN_metrics = {'main_output':[target_first, orig_second_conditional, orig_second_unconditional]}
  GAN_loss_metric_titles = ["Total Loss",
                            "L_X Loss",
                            "L_Y Loss",
                            "L_D Loss",
                            "1st Place Correct",
                            "2nd Place Correct (Conditional)",
                            "2nd Place Correct (Unconditional)",
                            "False positive rate (real examples labeled as fake)",
                            "True negative rate (fake examples labeled as real)"]
  ###################################################################

  if d_loss == 'least squares':
    disc_activ = 'unbounded'
  elif d_loss == 'sigmoid cross entropy':
    disc_activ = 'probability'
  else:
    print('disc_activ not recognized')
  #titles = [i for i in range(0,5)]
  #scores = [[None] * 2] * 5
  #classifiers = [None] * 5

  score = [[None] * 2]
  if not output_folder:
    warnings.warn('output_folder not supplied. Using default')
    output_folder = "{6}_output_G[{3}]_D[{7}]_f[{0}]_alph[{1:.3f}]_beta[{2:.3f}]_c[{4:.1f}]_rr[{5}]".format(
      model_key,alpha,beta,g_arch,c_hinge,rr_alpha,dataset,d_arch)
  if not os.path.exists(output_folder):
    os.makedirs(output_folder)
  train_gan_ind = {True:"training",False:"evaluation"}
  print(train_gan)
  logfile = "{}_log.txt".format(train_gan_ind[train_gan])
  #stdout is redirected to both stdout and the logfile

  sys.stdout = Logger(output_folder,logfile)
  print(
    """
    dataset: {}
    rr_alpha: {}
    alpha: {}
    beta: {}
    c_hinge: {}
    batch_size: {}
    num_classes: {}
    epochs: {}
    d_loss: {}
    train_gan: {}
    classifier architecture: {}
    generator architecture: {}
    discriminator architecture: {}
    parent_key: {} 
    """.format(dataset,rr_alpha, alpha, beta, c_hinge, batch_size, num_classes,
      epochs, d_loss, train_gan, model_key, g_arch, d_arch,parent_key)
  )
  """
    classifier = build_classifier(title,input_shape)
    classifier.compile(loss=tf.keras.losses.categorical_crossentropy,
                  optimizer=tf.keras.optimizers.Adadelta(),
                  metrics=['accuracy'])
    if train_classifier == True:
      classifier.fit(x_train, y_train,
                batch_size=batch_size,
                epochs=c_epochs,
                verbose=1,
                validation_data=(x_test, y_test))
      if not os.path.exists("saved_models"):
        os.makedirs("saved_models")
      classifier.save_weights("saved_models/cifar10_resnet_model.{0}.h5".format(title))
    else:
      classifier.load_weights("saved_models/cifar10_resnet_model.{0}.h5".format(title))
  """
  starting_epoch=0
  meta = read_from_meta()
  if parent_key is not None:
    parent_folder = meta['advgan'][parent_key]['train_params']['output_folder']
    starting_epoch = meta['advgan'][parent_key]['epochs_trained']
    if FLAGS.debug:
      epochs = 3+starting_epoch


  classifier_entry = meta['model'][model_key]

  if threat_model == 'black_box':
    bb_classifier_entry = meta['model'][black_box_model_key]
    assert 'black_box_A' in classifier_entry['threat_models']
    assert 'black_box_B' in bb_classifier_entry['threat_models']
    if dataset == 'CIFAR10':
      filename = classifier_entry['file_name'].replace('cifar10','cifar10A')
      bb_filename = bb_classifier_entry['file_name'].replace('cifar10','cifar10B')

    bb_load_model_path = os.path.join(os.getcwd(),bb_classifier_entry['folder_path'],bb_filename)
    bb_classifier = load_model(bb_load_model_path,compile=True,
          custom_objects = custom_object(classifier_entry['architecture']))    
    
    fsdist = filename.replace('.h5','_sdist_{}.h5'.format(black_box_model_key))
    fsdist_path = os.path.join(os.getcwd(),classifier_entry['folder_path'],fsdist)
    load_model_path = os.path.join(os.getcwd(),classifier_entry['folder_path'],filename)

    if fsdist in os.listdir(classifier_entry['folder_path']):
      print('loading distilled model')
      classifier = load_model(fsdist_path,compile=True,
            custom_objects = custom_object(classifier_entry['architecture']))
      if distillation_method == 'dynamic':
        bb_y_train = bb_classifier.predict(x_train)
      bb_y_test = bb_classifier.predict(x_test)
      bb_model_acc = np.count_nonzero(np.argmax(y_test,axis=-1) == np.argmax(bb_y_test,axis=-1))/y_test.shape[0]
      print('\tBlack Box Test Accuracy:', bb_model_acc)

    else:
      classifier = load_model(load_model_path,compile=True,
            custom_objects = custom_object(classifier_entry['architecture']))
      bb_y_train = bb_classifier.predict(x_train)
      bb_y_test = bb_classifier.predict(x_test)
      
      bb_model_acc = np.count_nonzero(np.argmax(y_test,axis=-1) == np.argmax(bb_y_test,axis=-1))/y_test.shape[0]
      print('\tBlack Box Test Accuracy:', bb_model_acc)
      # Prepare callbacks for model saving and for learning rate adjustment.
      """checkpoint = ModelCheckpoint(filepath=fsdist_path.replace('.h5','.{epoch:02d}-{val_loss:.2f}.h5'),
                                   monitor='val_acc',
                                   verbose=1,
                                   save_best_only=True)"""
      lr_scheduler = LearningRateScheduler(lr_schedule)
      lr_reducer = ReduceLROnPlateau(factor=np.sqrt(0.1),
                                     cooldown=0,
                                     patience=5,
                                     min_lr=0.5e-6)
      callbacks = [checkpoint, lr_reducer, lr_scheduler]
      classifier.fit(x_train, bb_y_train,
                batch_size=batch_size,
                epochs=100,
                validation_data=(x_test, bb_y_test),
                shuffle=True,
                callbacks=callbacks)

      save_model(classifier,fsdist_path,True)
  else:
    filename = classifier_entry['file_name']
    load_model_path = os.path.join(os.getcwd(),classifier_entry['folder_path'],filename)
    print('loading classifier model {}'.format(model_key))
    classifier = load_model(load_model_path,
            custom_objects = custom_object(classifier_entry['architecture']))

  sys.stdout.set_logging(False)
  score = classifier.evaluate(x_test, y_test, verbose=0)
  sys.stdout.set_logging(True)


  print(model_key,':')
  print('\tTest loss:', score[0])
  print('\tTest accuracy:', score[1])

  y_test_cls = classifier.predict(x_test,batch_size=32)
  y_train_cls = classifier.predict(x_train,batch_size=32)

  ####################Build the generator#####################

  # define adam parameters
  adam_lr=0.001 #from .0005
  adam_beta1=0.9
  adam_beta2=0.999
  adam_epsilon=1e-08

  if starting_epoch != 0:
    adam_lr /= np.sqrt(starting_epoch)

  if d_loss == 'least squares':
    d_real = -1   # a
    d_fake = 1    # b
    d_target = 0  # c
    d_loss_G = L_LSGAN
    d_loss_D = L_LSGAN
  else:
    d_real = 1
    d_fake = 0
    d_target = d_fake
    d_loss_G = L_GAN_ADV
    d_loss_D = L_GAN_CE

  # define inputs and outputs
  generator = build_generator_cifar(g_arch)
  discriminator = build_discriminator(D_arch=['C8','C16','C32','FC'],input_shape=input_shape,clip_min=0,clip_max=1, output_type=disc_activ)

  if not parent_key is None:
    generator.load_weights('{}/generator.hd5'.format(parent_folder))
    discriminator.load_weights('{}/discriminator.hd5'.format(parent_folder))


  discriminator.compile(loss=d_loss_D, optimizer=tf.keras.optimizers.Adadelta(), metrics=['accuracy'])
  print(discriminator.inputs,discriminator.outputs)

  # Build "frozen discriminator" to avoid discrepency warning
  frozen_discriminator = tf.keras.Model(
      discriminator.inputs,
      discriminator.outputs,
      name='frozen_discriminator'
  )

  #compile network with classifier frozen and save initial weights
  classifier.trainable = False
  for layer in classifier.layers:
    layer.trainable = False

  frozen_discriminator.trainable = False
  for layer in frozen_discriminator.layers:
    layer.trainable = False

  image = Input(shape = input_shape)
  target = Input(shape = (num_classes,))
  fake_image = generator([image,target])
  if clip_eps is None:
    fake_out = classifier(fake_image)
    fake_disc = frozen_discriminator(fake_image)
  else:
    fake_image_clipped = ClipLinf(clip_eps)([fake_image,image])
    fake_out = classifier(fake_image_clipped)
    fake_disc = frozen_discriminator(fake_image_clipped)
  
  print("Generator Summary")
  generator.summary()
  print("Discriminator Summary")
  discriminator.summary()
  print("Frozen Discriminator Summary")
  frozen_discriminator.summary()
  print(("Classifier Summary"))
  classifier.summary()
  
  # give a name to the output for metrics and create model
  fake_image = Reshape(target_shape=input_shape,name='aux_output')(fake_image)
  fake_out = Reshape(target_shape=(num_classes,),name='main_output')(fake_out)
  fake_disc = Reshape(target_shape=(1,),name='disc_output')(fake_disc)
  print('creating combined model')
  combined = Model([image,target],[fake_image,fake_out,fake_disc])

  combined.compile(
    optimizer=Adam(lr=adam_lr, beta_1=adam_beta1),
    loss=[L_X_advgan,L_Y_advgan, d_loss_G],
    loss_weights = [beta, 1., alpha], metrics = GAN_metrics)
  weights_init = combined.get_weights()
  # give a summary output
  print('Combined model:')
  combined.summary()
  print(combined.metrics_names)
  ########################################################
  if train_gan:
    # initialize data recording containers
    train_history = defaultdict(list)
    test_history = defaultdict(list)

    #initialize the training network
    combined.set_weights(weights_init)
    combined.compile(
      optimizer=Adam(lr=adam_lr, beta_1=adam_beta1),
      loss=[L_X_advgan, L_Y_advgan,d_loss_G],
      loss_weights = [beta, 1.,alpha],metrics = GAN_metrics)

    # get target labels
    _target_train = np.repeat(range(num_classes),num_train)
    target_train = tf.keras.utils.to_categorical(_target_train,num_classes)

    for epoch in range(starting_epoch,epochs):

      print('Epoch {}/{}'.format(epoch,epochs)) #target#

      num_batches =int(num_train*num_classes/batch_size)
      progress_bar = tf.keras.utils.Progbar(target=num_batches)

      epoch_generator_loss = []
      epoch_classifier_loss = []
      x_train_shuf = np.array(np.tile(x_train,(num_classes,1,1,1)))
      y_train_shuf = np.array(np.tile(y_train,(num_classes,1)))
      target_train_shuf = np.array(target_train)
      shuffle_in_unison(x_train_shuf,y_train_shuf,target_train_shuf)

      d_label_batch = d_target*np.ones((batch_size,),dtype=float)
      for index in range(num_batches):

        # get a batch of real images from the training set
        image_batch_orig = x_train_shuf[index * batch_size : (index + 1) * batch_size]
        label_batch_GT = y_train_shuf[index * batch_size : (index + 1) * batch_size]
        target_batch = target_train_shuf[index * batch_size : (index + 1) * batch_size]
        label_batch_orig = classifier.predict_on_batch(image_batch_orig)
        label_batch_rr = rerank(label_batch_orig,target_batch,rr_alpha) #target#
        if index==0:
          c_weights = classifier.get_weights()
          d_weights = discriminator.get_weights()
        # train the generator to minimize alpha*Lgan + beta*L_X + L_Y
        epoch_generator_loss.append(combined.train_on_batch(
          [image_batch_orig,target_batch],[image_batch_orig,label_batch_rr,d_label_batch])) ## image -> label
        if index==0:
          assert_weight_equality(d_weights,discriminator.get_weights())
          assert_weight_equality(d_weights,frozen_discriminator.get_weights())
          assert_weight_equality(c_weights,classifier.get_weights())

        # Only for GAN training
        image_batch_fake = generator.predict_on_batch([image_batch_orig,label_batch_GT])

        #interleave examples
        image_batch_combined = np.empty((batch_size*2,)+input_shape,dtype=float)
        label_batch_combined = np.empty((batch_size*2,num_classes),dtype=float)
        dlabel_batch_combined = d_real*np.ones((batch_size*2,),dtype=float)
        image_batch_combined[0::2] = image_batch_orig
        image_batch_combined[1::2] = image_batch_fake
        label_batch_combined[0::2] = label_batch_GT
        label_batch_combined[1::2] = label_batch_GT
        dlabel_batch_combined[1::2] = d_fake*np.ones((batch_size,),dtype=float)


        # If L_x is below the threshold
        """
        if epoch_generator_loss[-1][1] < Lx_threshold:
          samp_weights = (Lx_threshold/10)/
                         (np.concatenate(
                           (L_X(image_batch_orig,image_batch_orig),
                            L_X(image_batch_orig,image_batch_fake))) +
                         (Lx_threshold/10)*np.ones(2*image_batch_orig.shape[0]))
        """
        discriminator.train_on_batch(image_batch_combined,dlabel_batch_combined)
        if index == 0:
          assert_weight_inequality(d_weights,discriminator.get_weights())
          assert_weight_inequality(d_weights,frozen_discriminator.get_weights())

        if threat_model == 'black_box' and distillation_method == 'dynamic':
          # distill classifier with new adversarial examples
          bb_pred_batch_combined = bb_classifier.predict(image_batch_combined)
          classifier.train_on_batch(image_batch_combined,bb_pred_batch_combined)
          if index == 0:
            assert_weight_inequality(c_weights,classifier.get_weights())

        sys.stdout.set_logging(False)
        progress_bar.update(index + 1)
        sys.stdout.set_logging(True)

      for target in targets:

        #Set up data recording containers
        train_history['generator'].append([])
        train_history['classifier'].append([])
        test_history['generator'].append([])
        test_history['classifier'].append([])
        if threat_model == 'black_box':
          train_history['bb_classifier'].append([])
          test_history['bb_classifier'].append([])

        # evaluate testing loss
        print('Testing for epoch {}:'.format(epoch))
        generator_train_loss = np.mean(np.array(epoch_generator_loss), axis=0)
        print('generator Training Loss: {}'.format(generator_train_loss))

        y_target = np.zeros((num_test,num_classes),dtype='float64') #target#
        y_target[:,target] = 1 #target#
        x_train_samp = np.random.permutation(x_train)[:num_test]
        y_train_cls_samp = classifier.predict(x_train_samp,batch_size=32)
        y_train_rr_eval = rerank(y_train_cls_samp,y_target,rr_alpha)   #target#
        y_test_rr_eval = rerank(y_test_cls,y_target,rr_alpha)
        d_eval = d_target*np.ones((num_test,),dtype=float)
        print(y_target[0,:]) #target#

        [x_test_fake, y_test_fake,d_test_fake] = combined.predict([x_test,y_target],batch_size=32)
        if not clip_eps == None:
          x_test_fake = x_test + np.clip(x_test_fake-x_test,-clip_eps,clip_eps)
        d_test_true = discriminator.predict(x_test)
        print("Discriminator output for adversarial examples. Discriminator Target is",d_fake)
        print(d_test_fake[0:10])
        print("Discriminator output for real examples. Discriminator Target is {}",d_real)
        print(d_test_true[0:10])
        disc_type_I_error = np.mean(np.abs(d_test_true-d_real)>np.abs(d_test_true-d_fake))
        disc_type_II_error = np.mean(np.abs(d_test_fake-d_real)<np.abs(d_test_fake-d_fake))
        print("False positive rate (real examples mislabeled as fake): ",disc_type_I_error)
        print("True negative rate (adversarial examples mislabeled as real): ",disc_type_II_error)
        sys.stdout.set_logging(False)
        [_,model_acc] = classifier.evaluate(x_test, y_test, verbose=0)
        [_,adv_model_acc] = classifier.evaluate(x_test_fake, y_test, batch_size=32, verbose=0)
        [_,target_acc] = classifier.evaluate(x_test_fake,y_target,batch_size=32, verbose=0)
        
        if threat_model == 'black_box':
          [_,bb_model_acc] = bb_classifier.evaluate(x_test, y_test, verbose=0)
          [_,bb_adv_model_acc] = bb_classifier.evaluate(x_test_fake, y_test, batch_size=32, verbose=0)
          [_,bb_target_acc] = bb_classifier.evaluate(x_test_fake,y_target,batch_size=32, verbose=0)
          train_history['bb_classifier'][target].append([0,]) #
          test_history['bb_classifier'][target].append([bb_adv_model_acc,])

        generator_train_loss = combined.evaluate([x_train_samp,y_target],[x_train_samp,y_train_rr_eval,d_eval])+[disc_type_I_error,disc_type_II_error]
        generator_test_loss = combined.evaluate([x_test,y_target],[x_test, y_test_rr_eval,d_eval])+[disc_type_I_error,disc_type_II_error]
        sys.stdout.set_logging(True)
        print('generator train loss:',generator_train_loss)
        print('generator test loss:',generator_test_loss)

        #save the epoch information
        train_history['generator'][target].append(generator_train_loss) #target#
        train_history['classifier'][target].append([0,]) #target#
        test_history['generator'][target].append(generator_test_loss) #target#
        test_history['classifier'][target].append([adv_model_acc,])

        print('test history: ',test_history['generator'][target]) #target#
        print('train history: ',train_history['generator'][target]) #target#

        if threat_model == 'black_box':
          print('Black Box Model')
          print('Model accuracy on clean examples: {}'.format(bb_model_acc))
          print('Model accuracy on adversarial examples: {}'.format(bb_adv_model_acc))
          print('Adversarial examples classified as target class: {}'.format(bb_target_acc))
          print('\nDistilled Model')
        
        print('Model accuracy on clean examples: {}'.format(model_acc))
        print('Model accuracy on adversarial examples: {}'.format(adv_model_acc))
        print('Adversarial examples classified as target class: {}'.format(target_acc))
        """for i in range(5):
          print("Sample\n",i)
          print("y_test\n",y_test[i])
          print("y_test_cls\n",y_test[i])
          print("y_test_rr\n",y_test_rr[i])
          print("y_test_fake\n",y_test_fake[i])
          print("y_test_target\n",y_test_target[i])"""
        print("L_X:",L_X_advgan(x_test[:20],x_test_fake[:20]))
        print("L2_X:",L_X(x_test[:20],x_test_fake[:20]))
        print("L_Y:",L_Y_advgan(y_target[:20],y_test_fake[:20]))

        # generate some digits to display
        # arrange them into a grid
        x_diff = np.clip((x_test_fake - x_test) + 0.5,0.,1.)
        x_err = 10*np.clip(np.square(x_test_fake - x_test),0.,1.)

        if dataset=='MNIST':            
          img = (np.concatenate([r.reshape(-1, img_rows)
                 for r in np.split(np.concatenate((x_test[0:16],x_test_fake[0:16],x_diff[0:16],x_err[0:16])), 8)
                 ], axis=-1)*255).astype(np.uint8)
          np.repeat(img,4,axis=0)
          np.repeat(img,4,axis=1)
        elif dataset=='CIFAR10':
          ex_per_lbl = 2
          order = [[] for i in range(10)]
          for i,lbl in enumerate(np.argmax(y_test[:100],axis=-1)):
            if len(order[lbl])<ex_per_lbl:
              order[lbl].append(i)
          order = np.array(order).T.reshape((-1,))
          clean_pred = np.argmax(y_test_cls[order],axis=-1)
          clean_label = np.argmax(y_test[order],axis=-1)
          fake_pred = np.argmax(y_test_fake[order],axis=-1)
          color1 = np.array([(255,0,0),]*(num_classes*ex_per_lbl))
          color1[clean_pred==target] = (0,0,255)
          color1[clean_pred==clean_label] = (0,200,0)
          color2 = np.array([(255,0,0),]*(num_classes*ex_per_lbl))
          color2[fake_pred==target] = (0,0,255)
          color2[fake_pred==clean_label] = (0,200,0)

          thumbs = np.concatenate((x_test[order],x_test_fake[order],x_diff[order],x_err[order]))
          color = np.concatenate((color1,color2,color2,color2))
          thumbs = add_border((thumbs*255).astype(np.uint8),thickness=1,color=color)
          img = np.concatenate([r.reshape(-1, thumbs.shape[2], 3)
                 for r in np.split(thumbs, 4*ex_per_lbl)], axis=1)
          np.repeat(img,3,axis=0)
          np.repeat(img,3,axis=1)          

        Image.fromarray(img).save(
            '{0}/plot_target_{1}_epoch_{2:03d}_generated.png'.format(output_folder,ltran[target],epoch))
        if not FLAGS.debug:
          save_model(generator,'{}/generator_checkpoint.hd5'.format(output_folder), True)
          save_model(discriminator,'{}/discriminator_checkpoint.hd5'.format(output_folder), True)
          if distillation_method == 'dynamic' and threat_model == 'black_box':
            fddist = output_folder+'/classifier_ddist_checkpoint.h5'
    if not FLAGS.debug:
      save_model(generator,'{}/generator.hd5'.format(output_folder), True)
      save_model(discriminator,'{}/discriminator.hd5'.format(output_folder), True)
      if distillation_method == 'dynamic' and threat_model == 'black_box':
        fddist = output_folder+'/classifier_ddist.h5'
    if epochs<50: 
      tick_spacing = 2
    elif epochs<100:
      tick_spacing = 5
    else:
      tick_spacing = 10

    if not parent_key is None:
      load_history = pickle.load(open('{}/generator_history.pkl'.format(parent_folder), 'rb'))
    
    for target in range(num_classes):
      for key,train_entry in train_history.items():
        train_entry[target] = np.swapaxes(np.asarray(train_entry[target]),0,1) 
        if not parent_key is None:
          train_entry[target] = np.concatenate([load_history['train'][key][target],train_entry[target]],axis=-1)
      for key,test_entry in test_history.items():
        test_entry[target] = np.swapaxes(np.asarray(test_entry[target]),0,1)
        if not parent_key is None:
          test_entry[target] = np.concatenate([load_history['test'][key][target],test_entry[target]],axis=-1)

      for titl, el1, el2 in zip(GAN_loss_metric_titles,train_history['generator'][target],test_history['generator'][target]):
        plt.figure()
        plt.plot(el1,label='train')
        plt.plot(el2,label='test')
        plt.xticks(list(range(0,epochs,tick_spacing)))
        plt.title('Target ' + ltran[target] + ' ' + titl)
        plt.xlabel('Epoch')
        plt.legend(loc = 'upper right')
        plt.savefig('{}/target_{}_{}.png'.format(output_folder,ltran[target],titl))
        plt.close()
    for i, titl in enumerate(GAN_loss_metric_titles):
      plt.figure()
      for target in range(num_classes): 
        plt.plot(train_history['generator'][target][i],color='C'+str(target),linestyle='-',label=ltran[target])
        plt.plot(test_history['generator'][target][i],color='C'+str(target),linestyle='--')
      plt.xticks(list(range(0,epochs,tick_spacing)))
      plt.title('All Targets ' + titl)
      plt.xlabel('Epoch')
      plt.legend(loc = 'best')
      plt.savefig('{}/all_targets_{}.png'.format(output_folder,titl))
      plt.close()

    pickle.dump({'train': train_history, 'test': test_history},
                open('{}/generator_history.pkl'.format(output_folder), 'wb'))
    return {'epochs_trained':epochs}
  else:
    #history = pickle.load(open('{}/generator_history.pkl'.format(output_folder), 'rb'))
    #no need for this clutter at the moment
    #print(history)
    find_scales=True
    report_matrix_names = ['model_acc','target_acc',x_loss+'_perturbation_size']
    report_matrix = np.zeros((num_classes,num_classes+1,len(report_matrix_names)),dtype='float')
    report_matrix_by_class_rank = np.zeros((num_classes,num_classes,len(report_matrix_names)),dtype='float')
    class_rank_count = np.zeros((num_classes,num_classes,1),dtype='float')
    class_orig_count = np.zeros((num_classes,num_classes+1,1),dtype='float')
    y_test_target_all = np.zeros((num_test,num_classes,num_classes),dtype='float64')
    target_index_all = np.zeros((num_test,num_classes),dtype=int)
    output_ranks = y_test_cls.argsort(axis=1)
    output_ranks = output_ranks.argsort(axis=1)
    for target in range(num_classes):
      y_test_target_all[:,target,target] = 1
      target_index_all[:,target]=np.argmax(y_test,axis=-1)==target
    for target in range(num_classes):
      y_test_target = np.squeeze(y_test_target_all[:,:,target])
      not_the_target = np.squeeze(target_index_all[:,target]==0)
      #single_item_classifier = build_classifier(title,input_shape)
      #single_item_classifier.compile(loss=tf.keras.losses.categorical_crossentropy,
      #          optimizer=tf.keras.optimizers.Adadelta(),
      #          metrics=['accuracy'])
      
      generator = load_model('{}/generator.hd5'.format(output_folder),compile=False,
                              custom_objects = custom_object(g_arch))
      
      #generator = build_generator_cifar(g_arch)
      #generator.load_weights('{}/generator.hd5'.format(output_folder))

      #x_test is the test set of 30,000 examples give or take
      x_test_cur = x_test[not_the_target,:,:,:]
      y_test_cur = y_test[not_the_target,:]
      y_test_target = y_test_target[not_the_target,:]
      nb_target = y_test_cur.shape[0]

      x_test_fake = generator.predict([x_test_cur,y_test_target])
      
      (model_acc,adv_model_acc,target_acc,perturbation_size) = (0,0,0,0)

      y_test_cls_cur = y_test_cls[not_the_target] 
      adv_y_test_cls_cur = classifier.predict(x_test_fake, batch_size=32, verbose=0)

      for orig_class in range(num_classes):
        
        if orig_class == target:
          continue
        orig_ind = np.argmax(y_test_cur,axis=-1) == orig_class
        nb_orig = np.count_nonzero(orig_ind)

        _model_acc = np.count_nonzero(np.argmax(y_test_cls_cur[orig_ind],axis=-1) == orig_class)
        _adv_model_acc = np.count_nonzero(np.argmax(adv_y_test_cls_cur[orig_ind],axis=-1) == orig_class)
        _target_acc = np.count_nonzero(np.argmax(adv_y_test_cls_cur[orig_ind],axis=-1) == target)
        _perturbation_size = np.sum(L_X(x_test_fake[orig_ind],x_test_cur[orig_ind]))

        report_matrix[orig_class,orig_class,:] = (_model_acc,_model_acc,0)
        report_matrix[target,orig_class,:] = (_adv_model_acc,_target_acc,_perturbation_size)
        class_orig_count[orig_class,orig_class,:] = nb_orig
        class_orig_count[target,orig_class,:] = nb_orig
        model_acc += _model_acc
        adv_model_acc += _adv_model_acc
        target_acc += _target_acc
        perturbation_size += _perturbation_size

        for class_rank in range(num_classes):
          rank_ind = (output_ranks[not_the_target,target] == class_rank) & orig_ind
          nb_rank = np.count_nonzero(rank_ind)

          _adv_model_acc = np.count_nonzero(np.argmax(adv_y_test_cls_cur[rank_ind],axis=-1) == class_rank)
          _target_acc = np.count_nonzero(np.argmax(adv_y_test_cls_cur[rank_ind],axis=-1) == target)
          _perturbation_size = np.sum(L_X(x_test_fake[rank_ind],x_test_cur[rank_ind]))

          report_matrix_by_class_rank[class_rank,orig_class,:] += np.array((_adv_model_acc,_target_acc,_perturbation_size))
          class_rank_count[class_rank,orig_class,:] += nb_rank

      report_matrix[target,num_classes,:] = (adv_model_acc,target_acc,perturbation_size)
      class_orig_count[target,num_classes,:] = nb_target
      (model_acc,adv_model_acc,target_acc,perturbation_size) = (x/nb_target for x in (model_acc,adv_model_acc,target_acc,perturbation_size))

      if find_scales:
        #find scale required for misclassfication
        zero_scales = find_scale(x_test_cur,x_test_fake,classifier,target,iterations=10)
        success = zero_scales > -0.5
        x_test_failed = x_test_cur[~success,:,:,:]
        x_test_fake_failed = x_test_fake[~success,:,:,:]
        y_test_failed = y_test_cur[~success,:]
        L_X_thresh = 0
        zero_scales_failed = find_scale(x_test_failed,x_test_fake_failed,classifier,target=None,iterations=10)
        misclass = zero_scales_failed>-0.5
        zero_scales_failed = zero_scales_failed[misclass]

        zero_scales_big_failed = np.tile(np.reshape(zero_scales_failed,(-1,1,1,1)),(1,img_rows,img_cols,1))
        x_test_marginal_failed = np.clip(x_test_failed[misclass]+(x_test_fake_failed[misclass]
                                 -x_test_failed[misclass])*zero_scales_big_failed,0,1)
        marginal_error_failed = L_X(x_test_failed[misclass],x_test_marginal_failed)
        original_error_failed = L_X(x_test_failed[misclass],x_test_fake_failed[misclass])
        original_error_f2 = L_X(x_test_failed[~misclass],x_test_fake_failed[~misclass])

        x_test_cur = x_test_cur[success,:,:,:]
        x_test_fake = x_test_fake[success,:,:,:]
        zero_scales = zero_scales[success]
        y_test_target = y_test_target[success,:]
        zero_scales_big = np.tile(np.reshape(zero_scales,(-1,1,1,1)),(1,img_rows,img_cols,1))
        x_test_marginal = np.clip(x_test_cur+(x_test_fake-x_test_cur)*zero_scales_big,0,1)
        marginal_error = L_X(x_test_cur,x_test_marginal)
        original_error = L_X(x_test_cur,x_test_fake)
        sys.stdout.set_logging(False)
        if len(success[success]) != 0:
          [_,target_acc_1] = classifier.evaluate(x_test_fake,y_test_target,batch_size=32, verbose=0)
          [_,target_acc_2] = classifier.evaluate(x_test_marginal,y_test_target,batch_size=32, verbose=0)
        else:
          target_acc_1 = target_acc_2 = 0
        if len(misclass[misclass]) != 0:
          [_,misclass_acc_1] = classifier.evaluate(x_test_fake_failed[misclass],
                             y_test_failed[misclass,:],batch_size=32, verbose=0)
          [_,misclass_acc_2] = classifier.evaluate(x_test_marginal_failed,
                             y_test_failed[misclass,:],batch_size=32, verbose=0)

        else:
          misclass_acc_1 = 0.
          misclass_acc_2 = 0.
        sys.stdout.set_logging(True)
        misclass_acc_1 = 1 - misclass_acc_1
        misclass_acc_2 = 1 - misclass_acc_2
        success_percentage = 100*len(success[success])/len(success)
        misclass_percentage = 100*len(misclass[misclass])/len(success)
        f2_percentage = 100*len(misclass[~misclass])/len(success)
        report_args = (len(success[success]),success_percentage,           # 0,1
                       np.mean(original_error),100*target_acc_1,           # 2,3
                       np.mean(marginal_error),100*target_acc_2,           # 4,5
                       len(misclass[misclass]),misclass_percentage,        # 6,7
                       np.mean(original_error_failed),100*misclass_acc_1,  # 8,9
                       np.mean(marginal_error_failed),100*misclass_acc_2,  # 10,11
                       len(misclass[~misclass]),f2_percentage,             # 12,13
                       np.mean(original_error_f2),target,L_X_thresh,       # 14,15,16
                       len(success),model_acc,adv_model_acc,target_acc)    # 17,18,19,20
        
        print(
        """
      Target: {15}, Threshold: {16}
      All Examples
        Count:              {17}
        Baseline Accuracy:  {18:.4f}
        Acc on Adv Exmpls:  {19:.4f}
        Target Accuracy:    {20:.4f}
      Successful Examples
        Count:              {0} ({1:.2f}%)
        Original
          Avg L2 Error:     {2}
          Target Accuracy:  {3:.2f}%
        Scaled
          Avg L2 Error:     {4}
          Target Accuracy:  {5:.2f}%
      Untargeted Misclassifications
        Count:              {6} ({7:.2f})%
        Original
          Avg L2 Error:     {8}
          Misclass Acc:     {9:.2f}%
        Scaled
          Avg L2 Error:     {10}
          Misclass Acc:     {11:.2f}%
      Failed Perturbation Directions
        Count:              {12} ({13:.2f}%)
        Avg L2 Error:       {14}
          """.format(*report_args)
          )
        #plt.hist(zero_scales,bins=100,range=[[0,2],[0,1]],normed=True)
        plt.plot(zero_scales,marginal_error,"r.")
        plt.plot(zero_scales,original_error,"b.")
        plt.plot(zero_scales_failed,marginal_error_failed,"k.")
        plt.plot(zero_scales_failed,original_error_failed,"g.")
        plt.title("scale vs loss of adversarial examples Target "+ltran[target])
        plt.xlabel("scale")
        #plt.ylabel("frequency")
        plt.ylabel("L_X loss")
        #plt.show()
        plt.savefig('{}/target_{}_Scale.png'.format(output_folder,ltran[target]))
        plt.clf()
    labels1 = [l[:8] for l in ltran]
    labels1.append('total')
    labels2 = [list(range(1,num_classes+1)),labels1]
    labels2[0].append('total')
    class_rank_count = np.vstack((np.sum(class_rank_count,axis=0,keepdims=True),class_rank_count))
    class_rank_count = np.hstack((class_rank_count,np.sum(class_rank_count,axis=1,keepdims=True)))
    report_matrix_by_class_rank = np.vstack((np.sum(report_matrix_by_class_rank,axis=0,keepdims=True),report_matrix_by_class_rank))
    report_matrix_by_class_rank = np.hstack((report_matrix_by_class_rank,np.sum(report_matrix_by_class_rank,axis=1,keepdims=True)))
    assert np.abs(np.sum(class_rank_count[:,num_classes])-np.sum(class_rank_count[0,:]))<=1e-4
    assert np.abs(np.sum(report_matrix_by_class_rank[:,num_classes])-np.sum(report_matrix_by_class_rank[0,:]))<=1e-4
    report_matrix_by_class_rank/=class_rank_count
    off_diag_mask = np.expand_dims(np.hstack((1-np.eye(num_classes),np.ones((num_classes,1)))),-1)
    class_orig_count = np.vstack((class_orig_count,np.sum(class_orig_count*off_diag_mask,axis=0,keepdims=True)))
    report_matrix = np.vstack((report_matrix,np.sum(report_matrix*off_diag_mask,axis=0,keepdims=True)))/class_orig_count

    for counter,value in enumerate(report_matrix_names):
      if counter in [2,]:
        color_limits = [np.min(report_matrix),np.max(report_matrix)]
      else:
        color_limits = [0,1]
      colors = [(0.2,0.2,1),(0.3,0.3,0.3),(1,0.2,0.2),(0.8,0.8,0.8),(0.2,1,0.2)]
      shaded_data_table(np.squeeze(report_matrix[:,:,counter]),labels1,['target','original'],
                        colors=colors,color_limits=color_limits)
      plt.title(value.replace('_',' '))
      plt.savefig('{}/{}.png'.format(output_folder,'rpt_'+value))
      shaded_data_table(np.flipud(np.squeeze(report_matrix_by_class_rank[:,:,counter])),
                        labels2,['rank','original'], colors=colors,color_limits=color_limits)
      plt.title(value.replace('_',' '))
      plt.savefig('{}/{}.png'.format(output_folder,'rpt_by_rank_'+value))

    return {'tested':True}

def main(argv=None):
  import time
  start_time = time.time()
  kwargs = {'alpha':FLAGS.alpha,'beta':FLAGS.beta,
          'c_hinge':FLAGS.c_hinge,'rr_alpha':FLAGS.rr_alpha,
          'd_loss':FLAGS.d_loss,'x_loss':'L2',
          'g_arch':FLAGS.g_arch,'d_arch':FLAGS.d_arch,
          'model_key':FLAGS.model_key,'dataset':FLAGS.dataset,
          'targeted':True,'parent_key':FLAGS.parent_key,
          'clip_eps':FLAGS.clip_eps,
          'threat_model':FLAGS.threat_model,
          'distillation_method':FLAGS.distillation_method,
          'black_box_model_key':FLAGS.black_box_model_key}
  if FLAGS.train_gan:
    train_gan = FLAGS.train_gan
  else:
    train_gan = menu({'Train':True,'Test':False})

  meta = read_from_meta()
  validate_meta(meta)
  meta_choice_key = None

  if menu({'Use Flags Params':False,
           'Choose Params From Meta Entry':True}):
    while(1):  
      meta_choice_key = menu(meta['advgan'].keys())
      tmp = meta['advgan'][meta_choice_key]['train_params']
      output_folder = "{5}_{3}_D[{6}]_f[{0}]_alph[{1:.3f}]_beta[{2:.3f}]_c[{4:.1f}]".format(
        tmp['model_key'],tmp['alpha'],tmp['beta'],meta_choice_key,
        tmp['c_hinge'],tmp['dataset'],tmp['d_arch'])
      tmp.update({'output_folder':output_folder})
      print(meta_choice_key)
      pprint.pprint(meta['advgan'][meta_choice_key])
      assert tmp['output_folder']==meta['advgan'][meta_choice_key]['train_params']['output_folder']
      assert 'GAN_'+tmp['g_arch'] in meta_choice_key
      if train_gan:
        query = 'Retrain? (y/n): '
      else:
        if meta['advgan'][meta_choice_key]['tested']:
          query = 'Retest? (y/n): '
        else:
          query = 'Test? (y/n): '
      if input(query).lower()=='n':
        quit()
      else:
        break
  else:
    advgans = deepcopy(meta['advgan'])
    for k,v in advgans.items():  
      for k2 in meta['advgan'][k]['train_params'].keys():
        if k2 not in kwargs:
          v['train_params'].pop(k2)
      if v['train_params'] == kwargs:
        print('A trained AdvGAN already exists with the following parameters:')
        print(k)
        pprint.pprint(v)
        if input({True:'Retrain ',False:'Test '}[train_gan]+k+'? (y/n)')=='n':
          quit()
        meta_choice_key = k
        meta['advgan'][meta_choice_key]['train_params'].update({
          'batch_size':FLAGS.batch_size,
          'epochs':FLAGS.epochs})
        break
    if not meta_choice_key:
      meta_choice_key = get_new_key('GAN_'+kwargs['g_arch']+'_X',meta)
      
      output_folder = "{5}_{3}_D[{6}]_f[{0}]_alph[{1:.3f}]_beta[{2:.3f}]_c[{4:.1f}]".format(
        kwargs['model_key'],kwargs['alpha'],kwargs['beta'],meta_choice_key,
        kwargs['c_hinge'],kwargs['dataset'],kwargs['d_arch'])
      kwargs.update({
        'output_folder':output_folder,
        'batch_size':FLAGS.batch_size,
        'epochs':FLAGS.epochs
      })
      meta['advgan'].update({meta_choice_key:{
        'tested':False,
        'train_params':kwargs
        }})
  create_new_key = True
  for k,v in meta['attacker'].items():
    if 'meta_key' in v.keys():
      if v["meta_key"]==meta_choice_key:
        create_new_key = False
  if create_new_key:
    new_key=get_new_key('advgan_new',meta)
    meta['attacker'].update({
          new_key: {
            "attack_type": "advgan",
            "active": True,
            "meta_key": meta_choice_key,
            "threat_model": kwargs['threat_model'],
            "attack_params": {
                "eps": 1,
                "ord": np.inf,
            }
                }})
    if kwargs['threat_model']=='black_box':
      meta['attacker']['new_key'].update({
        "attack_strategy": kwargs['distillation_method']+' distillation',
        "target_model": kwargs['black_box_model_key']})

  assert FLAGS.model_key in meta['model'].keys()
  kwargs = meta['advgan'][meta_choice_key]['train_params']
  if not FLAGS.parent_key is None:
    assert kwargs['g_arch']==meta['advgan'][FLAGS.parent_key]['train_params']['g_arch']
    assert kwargs['d_arch']==meta['advgan'][FLAGS.parent_key]['train_params']['d_arch']
    assert kwargs['targeted']==meta['advgan'][FLAGS.parent_key]['train_params']['targeted']
  kwargs.update({'train_gan':train_gan})
  ##################################
  report = train_AdvGAN(**kwargs) ##
  ##################################
  meta['advgan'][meta_choice_key].update(report)
  meta['advgan'][meta_choice_key]['train_params'].update({'train_gan':False})

  write_to_meta(meta,kwargs['dataset'])
  end_time = time.time()
  runtime_m = (end_time-start_time)/60
  print("Program took {0:.2f} minutes".format(runtime_m))

if __name__ == '__main__':

  np.set_printoptions(precision=3)
  flags.DEFINE_string('dataset','CIFAR10',
                      "'CIFAR10' is so far the only dataset supported"
                      "'MNIST', 'CIFAR100', 'IMAGENET', are possible"
                      "future options")
  flags.DEFINE_string('x_loss','L2',
                      "identifier for the loss metric used on the perturbation")
  flags.DEFINE_string('y_loss','CW',
                      "Identifier for the loss metric used on the Y outputs")
  flags.DEFINE_string('g_arch','Dense3',
                      "generator architecture")
  flags.DEFINE_string('d_arch','A',
                      "discriminator architecture")
  flags.DEFINE_string('model_key','model_1_a',
                      "meta key for classifier to train on")
  flags.DEFINE_string('d_loss','least squares',
                      "discriminator loss function descriptor. Options"
                      "are 'least squares' or 'sigmoid cross entropy'")
  flags.DEFINE_float('c_hinge', 4,
                      "hinge loss offset, between 0 and max input loss")
  flags.DEFINE_float('alpha', 0.5,
                      "weight of the discriminator output on GAN loss")
  flags.DEFINE_float('beta', 0.5,
                      "weight of the L_X hinge loss on the overall loss")
  flags.DEFINE_float('rr_alpha',np.inf,
                      "rerank alpha for ATN")
  flags.DEFINE_bool('train_gan', False,"GAN training flag."
                    "True for training, False for testing")
  flags.DEFINE_integer('batch_size',64,"GAN training batch size")
  flags.DEFINE_integer('epochs',40,"GAN training epochs")
  flags.DEFINE_bool('debug',False,'Run in debug mode')
  flags.DEFINE_string('parent_key',None,'Supply meta key of parent' 
                      'if you want to load pretrained weights')
  flags.DEFINE_float('clip_eps',None,'epsilon for clipping before'
                      'classifier')
  flags.DEFINE_bool('targeted',True,'If True, the advgan will be a targeted'
                    'attack. If false, untargeted')
  flags.DEFINE_string('threat_model','white_box',
                      "white_box or black_box")
  flags.DEFINE_string('distillation_method','static',
                      "static or dynamic ignored in white box [Xiao]")
  flags.DEFINE_float('learning_rate',0.0005,'learning_rate for adam optimizer')
  flags.DEFINE_string('black_box_model_key','model_3_0',
                      "meta key for black box query model")
  tf.app.run()