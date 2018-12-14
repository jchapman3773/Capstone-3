import os
import matplotlib as mpl
if os.environ.get('DISPLAY','') == '':
    print('no display found. Using non-interactive Agg backend')
    mpl.use('Agg')
import numpy as np
from PIL import Image
import pickle
from collections import Counter
from keras.applications import Xception, ResNet50
from keras.preprocessing import image
from keras.layers import Dense, GlobalAveragePooling2D, Flatten, Dropout
from keras.models import Model
from keras.applications.xception import preprocess_input
from keras.optimizers import Adam
from keras.preprocessing.image import ImageDataGenerator
from keras.models import Sequential, load_model
from keras import callbacks
from print_pretty_confusion_matrix import plot_confusion_matrix_from_data
from sklearn.metrics import classification_report
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

    def __init__(self,model=Xception,target_size=(300,300),weights='imagenet',
                n_categories=4,batch_size=8,augmentation_strength=0.3,
                preprocessing=preprocess_input,epochs=10):
        self.model = model
        self.target_size = target_size
        self.input_size = self.target_size + (3,)
        self.weights = weights
        self.n_categories = n_categories
        self.batch_size = batch_size
        self.train_generator = None
        self.validation_generator = None
        self.holdout_generator = None
        self.augmentation_strength = augmentation_strength
        self.preprocessing = preprocessing
        self.epochs = epochs
        self.class_weights = None

    def make_generators(self,directory):
        train_data_dir = directory+'/train'
        holdout_data_dir = directory+'/holdout'

        train_datagen = ImageDataGenerator(
            preprocessing_function=self.preprocessing,
            rotation_range=15*self.augmentation_strength,
            width_shift_range=self.augmentation_strength,
            height_shift_range=self.augmentation_strength,
            shear_range=self.augmentation_strength,
            zoom_range=self.augmentation_strength,
            # horizontal_flip=True,
            vertical_flip=True,
            validation_split=0.15)

        holdout_datagen = ImageDataGenerator(
            preprocessing_function=self.preprocessing)

        self.train_generator = train_datagen.flow_from_directory(
            train_data_dir,
            target_size=self.target_size,
            batch_size=self.batch_size,
            class_mode='categorical',
            subset='training')

        self.validation_generator = train_datagen.flow_from_directory(
            train_data_dir,
            target_size=self.target_size,
            batch_size=self.batch_size,
            class_mode='categorical',
            subset='validation')

        self.holdout_generator = holdout_datagen.flow_from_directory(
            holdout_data_dir,
            target_size=self.target_size,
            batch_size=self.batch_size,
            class_mode='categorical',
            shuffle=False)

        self.holdout_generator_images = holdout_datagen.flow_from_directory(
            holdout_data_dir,
            target_size=self.target_size,
            batch_size=self.holdout_generator.samples,
            class_mode='categorical',
            shuffle=False)

        return self.train_generator, self.validation_generator, self.holdout_generator

    def _create_transfer_model(self):
        base_model = self.model(weights=self.weights,
                          include_top=False,
                          input_shape=self.input_size)
        x = base_model.output
        x = GlobalAveragePooling2D()(x)
        predictions = Dense(self.n_categories, activation='linear')(x)
        self.model = Model(inputs=base_model.input, outputs=predictions)
        return self.model

    def _find_class_weights(self):
        counter = Counter(self.train_generator.classes)
        max_val = float(max(counter.values()))
        self.class_weights = {class_id : max_val/num_images for class_id, num_images in counter.items()}
        return self.class_weights

    def _change_trainable_layers(self, trainable_index):
        for layer in self.model.layers[:trainable_index]:
            layer.trainable = False
        for layer in self.model.layers[trainable_index:]:
            layer.trainable = True

    def fit(self,freeze_indices,optimizers,warmup_epochs=5):
        # callbacks
        filepath = 'models/transfer_CNN.h5'
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

        if not os.path.exists('tensorboard_logs/transfer_CNN_tensorboard_with_weights'):
            os.makedirs('tensorboard_logs/transfer_CNN_tensorboard_with_weights')
        tensorboard = callbacks.TensorBoard(
                    log_dir='tensorboard_logs/transfer_CNN_tensorboard_with_weights',
                    histogram_freq=0,
                    batch_size=self.batch_size,
                    write_graph=True,
                    embeddings_freq=0,
                    write_images=False)

        # change head from default
        self._create_transfer_model()
        # fix class imbalance
        self._find_class_weights()

        # train head, then chunks
        histories = []
        for i, _ in enumerate(freeze_indices):
            if i == 0:
                e = warmup_epochs
            else:
                e = self.epochs
            self._change_trainable_layers(freeze_indices[i])
            self.model.compile(optimizer=optimizers[i],
                          loss='categorical_crossentropy', metrics=['accuracy'])

            history = self.model.fit_generator(self.train_generator,
                                      steps_per_epoch=len(self.train_generator),
                                      epochs=e,
                                      class_weight=self.class_weights,
                                      validation_data=self.validation_generator,
                                      validation_steps=len(self.validation_generator),
                                      callbacks=[mc, tensorboard, hist])
            histories.append(history.history)
        return histories

    def best_training_model(self):
        model = load_model('models/transfer_CNN.h5')
        pred = model.predict_generator(self.holdout_generator,
                                                steps=len(self.holdout_generator))
        predictions = np.argmax(pred, axis=-1)
        label_map = (self.holdout_generator.class_indices)
        label_map = dict((v,k) for k,v in label_map.items()) #flip k,v
        names = np.array([x.split('/') for x in self.holdout_generator.filenames])
        predictions = [label_map[k] for k in predictions]
        predictions = np.array(predictions).reshape(-1,1)
        data = np.hstack((names,predictions))
        metrics = model.evaluate_generator(self.holdout_generator,
                                            steps=len(self.holdout_generator),
                                            verbose=1)
        return metrics, data, pred

    def print_matrix(self,y_true,y_pred):
        columns = self.holdout_generator.class_indices.keys()
        plot_confusion_matrix_from_data(y_true,y_pred,columns=columns)
        return

    def return_failed_images(self,dir,data,pred):
        mpl.rcParams.update({
            'figure.figsize'      : (8,50),
            # 'font.size'           : 20.0,
            # 'axes.titlesize'      : 'medium',
            # 'axes.labelsize'      : 'medium',
            # 'xtick.labelsize'     : 'medium',
            # 'ytick.labelsize'     : 'medium',
            # 'legend.fontsize'     : 'large',
            # 'legend.loc'          : 'upper right'
        })
        failed = data[data[:,0]!=data[:,2]]
        fig, axes = plt.subplots(len(failed),2)

        test_X = self.holdout_generator_images[0][0]
        indices = np.where(data[:,0]!=data[:,2])[0]

        for idx,row in enumerate(failed):
            file_path = dir+'/holdout/'+row[0]+'/'+row[1]
            # plot original image
            axes[idx][0].imshow(Image.open(file_path))
            # plot processed image
            axes[idx][1].imshow((test_X[indices[idx]]/2)+0.5)
            # set titles
            axes[idx][0].set_title(f'Actual: {row[0]}')
            axes[idx][1].set_title(f'Predicted: {row[2]}, {round(pred[indices[idx]].max()*100,2)}%')
            # remove ticks
            for idx2 in range(2):
                axes[idx][idx2].get_xaxis().set_visible(False)
                axes[idx][idx2].get_yaxis().set_visible(False)
                for t in axes[idx][idx2].xaxis.get_major_ticks():
                    t.tick1On = False
                    t.tick2On = False
                for t in axes[idx][idx2].yaxis.get_major_ticks():
                    t.tick1On = False
                    t.tick2On = False
        plt.tight_layout()
        plt.savefig('../graphics/failed_images.png')
        plt.close()
        return

    def _hstack_histories(self,histories,metric):
        lst = []
        for hist in histories:
            lst.append(hist[metric])
        return tuple(lst)

    def plot_history(self, histories):
        # Plot training & validation accuracy values
        hist_acc = np.hstack(self._hstack_histories(histories,'acc'))
        hist_val_acc = np.hstack(self._hstack_histories(histories,'val_acc'))
        plt.plot(hist_acc)
        plt.plot(hist_val_acc)
        plt.title('Model Accuracy')
        plt.ylabel('Accuracy')
        plt.xlabel('Epoch')
        plt.axvline(5,color='k',linestyle='dotted')
        plt.axvline(15,color='k',linestyle='dotted')
        plt.axvline(25,color='k',linestyle='dotted')
        plt.legend(['Train', 'Test'], loc='upper left')
        plt.tight_layout()
        plt.savefig('../graphics/Transfer_CNN_acc_hist.png')
        plt.close()

        # Plot training & validation loss values
        hist_loss = np.hstack(self._hstack_histories(histories,'loss'))
        hist_val_loss = np.hstack(self._hstack_histories(histories,'val_loss'))
        plt.plot(hist_loss)
        plt.plot(hist_val_loss)
        plt.title('Model Loss')
        plt.ylabel('Loss')
        plt.xlabel('Epoch')
        plt.axvline(5,color='k',linestyle='dotted')
        plt.axvline(15,color='k',linestyle='dotted')
        plt.axvline(25,color='k',linestyle='dotted')
        plt.legend(['Train', 'Test'], loc='upper left')
        plt.tight_layout()
        plt.savefig('../graphics/Transfer_CNN_loss_hist.png')
        plt.close()

        # Plot everything
        import pdb; pdb.set_trace()
        fig, ax1 = plt.subplots()
        ax1.plot(hist_acc)
        ax1.plot(hist_val_acc)
        ax1.set_ylabel('Accuracy')

        ax2 = ax1.twinx()
        ax2.plot(hist_loss)
        ax2.plot(hist_val_loss)
        ax2.set_ylabel('Loss')

        ax1.set_title('Model Accuracy and Loss')
        ax1.set_xlabel('Epoch')
        ax1.vlines([5,15,25],color='k',linestyle='dotted')
        ax1.legend(['Train', 'Test'], loc='upper left')
        fig.tight_layout()
        import pdb; pdb.set_trace()
        plt.savefig('../graphics/Transfer_CNN_history.png')
        plt.close()

if __name__ == '__main__':
    dir = '../data/Banana_People_Not/4_Classes'

    transfer_CNN = TransferModel()
    transfer_CNN.make_generators(dir)

    freeze_indices = [132, 126, 116 ,106]
    optimizers = [Adam(lr=0.0005), Adam(lr=0.00005), Adam(lr=0.00005), Adam(lr=0.00005)]
    histories = transfer_CNN.fit(freeze_indices,optimizers)
    pickle.dump(histories, open('hist.pkl', 'wb'))

    transfer_CNN.plot_history(histories)

    # # plot model
    # from keras.utils import plot_model
    # plot_model(transfer_CNN.model, to_file='../graphics/transfer_CNN_model.png')

    metrics, data, pred = transfer_CNN.best_training_model()
    transfer_CNN.print_matrix(data[:,0],data[:,2])
    transfer_CNN.return_failed_images(dir,data,pred)
    print(metrics)
    print(classification_report(data[:,0],data[:,2]))

    with open(f"models/transfer_CNN_report.txt", "w") as text_file:
        print(str(metrics)+'\n\n'+str(classification_report(data[:,0],data[:,2])),file=text_file)
