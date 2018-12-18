import os
import matplotlib as mpl
if os.environ.get('DISPLAY','') == '':
    print('no display found. Using non-interactive Agg backend')
    mpl.use('Agg')
import numpy as np
import pandas as pd
from PIL import Image
import pickle
from collections import Counter
from keras.applications import Xception
from keras.preprocessing import image
from keras.layers import Dense, GlobalAveragePooling2D, Flatten, Dropout
from keras.models import Model
from keras.applications.xception import preprocess_input
from keras.optimizers import Adam
from keras.preprocessing.image import ImageDataGenerator
from keras.models import Sequential, load_model
from keras import callbacks
from sqlalchemy import create_engine
import matplotlib.pyplot as plt
mpl.style.use('classic')

# for reproducibility
# -----------------------------------------------------------------------------
seed_value = 42

# 1. Set `PYTHONHASHSEED` environment variable at a fixed value
os.environ['PYTHONHASHSEED']=str(seed_value)

# 2. Set `python` built-in pseudo-random generator at a fixed value
import random
random.seed(seed_value)

# 3. Set `numpy` pseudo-random generator at a fixed value
np.random.seed(seed_value)

# 4. Set `tensorflow` pseudo-random generator at a fixed value
import tensorflow as tf
tf.set_random_seed(seed_value)
session_conf = tf.ConfigProto(intra_op_parallelism_threads=1, inter_op_parallelism_threads=1)

# 5. Configure a new global `tensorflow` session
from keras import backend as K
sess = tf.Session(graph=tf.get_default_graph(), config=session_conf)
K.set_session(sess)
# -----------------------------------------------------------------------------

class TransferModel():

    def __init__(self,model=Xception,target_size=(800,800),weights='imagenet',
                batch_size=2,augmentation_strength=0.3,
                preprocessing=preprocess_input,epochs=10):
        self.model = model
        self.target_size = target_size
        self.input_size = self.target_size + (3,)
        self.weights = weights
        self.batch_size = batch_size
        self.train_generator = None
        self.validation_generator = None
        self.augmentation_strength = augmentation_strength
        self.preprocessing = preprocessing
        self.epochs = epochs
        self.class_weights = None

    def make_generators(self,directory):

        train_datagen = ImageDataGenerator(
            preprocessing_function=self.preprocessing,
            rotation_range=15*self.augmentation_strength,
            width_shift_range=self.augmentation_strength,
            height_shift_range=self.augmentation_strength,
            shear_range=self.augmentation_strength,
            zoom_range=self.augmentation_strength,
            vertical_flip=True,
            validation_split=0.15)

        # train_label_df = pd.read_csv('../data/heights.csv')
        engine = create_engine('postgresql://banana:forscale@bananaforscale.ckaldwfguyw5.us-east-2.rds.amazonaws.com:5432/bananaforscale')
        train_label_df = pd.read_sql_table('heights',con=engine)
        print(train_label_df.shape)
        self.train_generator = train_datagen.flow_from_dataframe(
                                            dataframe=train_label_df,
                                            directory=directory,
                                            x_col='image',
                                            y_col='height_inch',
                                            has_ext=True,
                                            class_mode='other',
                                            target_size=self.target_size,
                                            batch_size=self.batch_size,
                                            subset='training')

        self.validation_generator = train_datagen.flow_from_dataframe(
                                            dataframe=train_label_df,
                                            directory=directory,
                                            x_col='image',
                                            y_col='height_inch',
                                            has_ext=True,
                                            class_mode='other',
                                            target_size=self.target_size,
                                            batch_size=self.batch_size,
                                            subset='validation',
                                            shuffle=False)

        return self.train_generator, self.validation_generator

    def _create_transfer_model(self):
        base_model = self.model(weights=self.weights,
                          include_top=False,
                          input_shape=self.input_size)
        x = base_model.output
        x = GlobalAveragePooling2D()(x)
        predictions = Dense(1, activation='linear')(x)
        self.model = Model(inputs=base_model.input, outputs=predictions)
        return self.model

    def _change_trainable_layers(self, trainable_index):
        for layer in self.model.layers[:trainable_index]:
            layer.trainable = False
        for layer in self.model.layers[trainable_index:]:
            layer.trainable = True

    def fit(self,freeze_indices,optimizers,warmup_epochs=5):
        # callbacks
        filepath = 'models/transfer_CNN_reg.h5'
        mc = callbacks.ModelCheckpoint(filepath,
                                            monitor='val_loss',
                                            verbose=1,
                                            save_best_only=True,
                                            save_weights_only=False,
                                            mode='auto',
                                            period=1)
        hist = callbacks.History()
        es = callbacks.EarlyStopping(monitor='val_loss',
                                            min_delta=0,
                                            patience=2,
                                            verbose=1,
                                            mode='auto')

        if not os.path.exists('tensorboard_logs/transfer_CNN_tensorboard_reg'):
            os.makedirs('tensorboard_logs/transfer_CNN_tensorboard_reg')
        tensorboard = callbacks.TensorBoard(
                    log_dir='tensorboard_logs/transfer_CNN_tensorboard_reg',
                    histogram_freq=0,
                    batch_size=self.batch_size,
                    write_graph=True,
                    embeddings_freq=0,
                    write_images=False)

        # change head from default
        self._create_transfer_model()

        # train head, then chunks
        histories = []
        for i, _ in enumerate(freeze_indices):
            if i == 0:
                e = warmup_epochs
                opt = optimizers[0]
            else:
                e = self.epochs
                opt = optimizers[1]
            self._change_trainable_layers(freeze_indices[i])
            self.model.compile(optimizer=opt,
                          loss='mean_squared_error')

            history = self.model.fit_generator(self.train_generator,
                                      steps_per_epoch=len(self.train_generator),
                                      epochs=e,
                                      validation_data=self.validation_generator,
                                      validation_steps=len(self.validation_generator),
                                      callbacks=[mc, tensorboard, hist])
            histories.append(history.history)
        return histories

    def best_training_model(self):
        model = load_model('models/transfer_CNN_reg.h5')
        pred = model.predict_generator(self.validation_generator,
                                                steps=len(self.validation_generator))
        pred = np.array(pred).reshape(-1,1)
        labels = np.array([x.replace('.','_').split('_') for x in self.validation_generator.filenames])
        labels = np.array(['.'.join([row[1],row[2]]) for row in labels])
        names = self.validation_generator.filenames
        data = np.hstack((np.array(names).reshape(-1,1),labels.reshape(-1,1),pred))
        metrics = model.evaluate_generator(self.validation_generator,
                                            steps=len(self.validation_generator),
                                            verbose=1)
        return metrics, data, pred

    def _hstack_histories(self,histories,metric):
        lst = []
        for hist in histories:
            lst.append(hist[metric])
        return tuple(lst)

    def plot_history(self, histories):
        # Plot training & validation accuracy values
        hist_acc = np.hstack(self._hstack_histories(histories,'loss'))
        hist_val_acc = np.hstack(self._hstack_histories(histories,'val_loss'))
        plt.plot(hist_acc)
        plt.plot(hist_val_acc)
        plt.title('Model Mean Squared Error')
        plt.ylabel('MSE')
        plt.ylim(top=max(hist_acc))
        plt.xlabel('Epoch')
        plt.axvline(5,color='k',linestyle='dotted')
        plt.axvline(15,color='k',linestyle='dotted')
        plt.axvline(25,color='k',linestyle='dotted')
        plt.legend(['Train', 'Test'], loc='upper left')
        plt.tight_layout()
        plt.savefig('../graphics/Transfer_CNN_reg_mse_hist.png')
        plt.close()

if __name__ == '__main__':
    dir = '../data/uploads/'

    transfer_CNN = TransferModel()
    transfer_CNN.make_generators(dir)

    freeze_indices = [132, 126, 116 ,106]
    optimizers = [Adam(lr=0.0005), Adam(lr=0.00005)]
    histories = transfer_CNN.fit(freeze_indices,optimizers)
    pickle.dump(histories, open('hist_reg.pkl', 'wb'))

    transfer_CNN.plot_history(histories)

    # # plot model
    # from keras.utils import plot_model
    # plot_model(transfer_CNN.model, to_file='../graphics/transfer_CNN_model.png')

    metrics, data, pred = transfer_CNN.best_training_model()

    print(metrics)
    print(data)
