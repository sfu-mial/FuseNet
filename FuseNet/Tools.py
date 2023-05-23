import logging
import os
import tensorflow as tf
import keras
import keras.backend as K
lgr = None

def initlogger(configuration):
    global lgr
    if lgr is None:
        lgr = logging.getLogger('global')
    if 'logdir' in configuration:
        fh = logging.FileHandler(os.path.join(configuration['logdir'], 'FuseNet.log'))
        fh.setLevel(logging.INFO)
        formatter = logging.Formatter("[%(filename)s:%(lineno)s - %(funcName)20s() ] %(message)s")
        fh.setFormatter(formatter)
        lgr.addHandler(fh)
    lgr.setLevel(logging.INFO)
    return lgr


def getlogger():
    global lgr
    if lgr is None:
        return initlogger({})
    return lgr

class PlotLosses(keras.callbacks.Callback):
    def on_train_begin(self, logs={}):
        self.i = 0
        self.x = []
        self.losses_task = []
        self.val_losses_task = []
        self.losses_recons = []
        self.val_losses_recons = []
        self.acc = []
        self.val_acc = []
        self.fig = plt.figure()
        self.logs = []

    def on_epoch_end(self, epoch, logs={}):
        self.logs.append(logs)
        self.x.append(self.i)
        self.losses_recons.append(logs.get('reconstruction_output_fuse_loss'))
        self.val_losses_recons.append(logs.get('val_reconstruction_output_fuse_loss'))
        self.losses_task.append(logs.get('category_output_loss'))
        self.val_losses_task.append(logs.get('val_category_output_loss'))
        self.acc.append(logs.get('category_output_accuracy'))
        self.val_acc.append(logs.get('val_category_output_accuracy'))
        self.i += 1
        
        # clear_output(wait=True)
        plt.plot(self.x, self.losses_task, label="loss")
        plt.plot(self.x, self.val_losses_task, label="val_loss")
        axes = plt.gca()
        # axes.set_ylim([0,1])
        plt.legend()
        # plt.show();
        plt.title('model_task_loss')
        # plt.yscale('log')
        plt.ylabel('loss')
        plt.xlabel('epoch')
        plt.savefig("results/model_task_loss.png", bbox_inches='tight')
        plt.close("all")

        # clear_output(wait=True)
        plt.plot(self.x, self.losses_recons, label="losses_recons")
        plt.plot(self.x, self.val_losses_recons, label="val_losses_recons")
        axes = plt.gca()
        # axes.set_ylim([0,1])
        plt.legend()
        # plt.show();
        plt.title('model_reconst_loss')
        # plt.yscale('log')
        plt.ylabel('loss')
        plt.xlabel('epoch')
        plt.savefig("results/model_recons_loss.png", bbox_inches='tight')
        plt.close("all")
        plt.plot(self.x, self.acc, label="category_output_acc")
        plt.plot(self.x, self.val_acc, label="val_category_output_acc")
        axes = plt.gca()
        # axes.set_ylim([0,1])
        plt.legend()
        # plt.show();
        plt.title('model accuracy')
        # plt.yscale('log')
        plt.ylabel('accuracy')
        plt.xlabel('epoch')
        plt.savefig("results/model accuracy.png", bbox_inches='tight')
        plt.close("all")

plot_losses = PlotLosses()

class MyCallback(keras.callbacks.Callback):
    def __init__(self, alpha, beta):
        self.alpha = alpha
        self.beta = beta
    # customize your behavior
    def on_epoch_end(self, epoch, logs={}):
        if epoch >100 and K.get_value(self.beta)<=0.1:
            # K.set_value(self.beta, K.get_value(self.beta) +0.0001)
            if  K.get_value(self.alpha)<0.3:
                 K.set_value(self.alpha, K.get_value(self.alpha) +0.001)
#            K.set_value(self.alpha, max(0.75, K.get_value(self.alpha) -0.0001))
#                  K.set_value(self.beta,  min(0.7, K.get_value(self.beta) -0.0001))
        logger.info("epoch %s, alpha = %s, beta = %s" % (epoch, K.get_value(self.alpha), K.get_value(self.beta)))

class CustomSchedule(tf.keras.optimizers.schedules.LearningRateSchedule):
  def __init__(self, d_model, warmup_steps=30):
    super(CustomSchedule, self).__init__()

    self.d_model = d_model
    self.d_model = tf.cast(self.d_model, tf.float32)

    self.warmup_steps = warmup_steps

  def __call__(self, step):
    arg1 = tf.math.rsqrt(step)
    arg2 = step * (self.warmup_steps ** -1.5)

    return tf.math.rsqrt(self.d_model) * tf.math.minimum(arg1, arg2) 

class LearningRateReducerCb(tf.keras.callbacks.Callback):

  def on_epoch_end(self, epoch, logs={}):
    old_lr = self.model.optimizer.lr.read_value()# K.get_value(model.optimizer.lr)

    new_lr = old_lr * 0.99
    print("\nEpoch: {}. Reducing Learning Rate from {} to {}".format(epoch, old_lr, new_lr))
    self.model.optimizer.lr.assign(new_lr)
    if epoch >= 1: #and epoch % 2 == 0:
        plot_generated_images(epoch, self.model, True,Tmp_ssimlist)
        plot_confusionmatrix(epoch, self.model)
        plot_roc_curve(self.model)
        # lr schedule callback
        # lr_decay_callback = tf.keras.callbacks.LearningRateScheduler(lr_decay)

    PlotLosses()
    if epoch == 5 or (epoch >= 10 and epoch % 10 == 0):
        self.model.save('./results/deep_spa_mse_only.h5')


class CustomSchedule(tf.keras.optimizers.schedules.LearningRateSchedule):
  def __init__(self, d_model, warmup_steps=30):
    super(CustomSchedule, self).__init__()

    self.d_model = d_model
    self.d_model = tf.cast(self.d_model, tf.float32)

    self.warmup_steps = warmup_steps

  def __call__(self, step):
    arg1 = tf.math.rsqrt(step)
    arg2 = step * (self.warmup_steps ** -1.5)

    return tf.math.rsqrt(self.d_model) * tf.math.minimum(arg1, arg2)  




