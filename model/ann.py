import tensorflow as tf
import pandas as pd
import os
import pickle
import matplotlib.pyplot as plt
import matplotlib as mpl
mpl.use('Agg')
import json

## INITIALIZATION OF PARAMETERS
DATASET_PATH = os.path.join(os.path.curdir, 'Titanic-Dataset.csv')
REMOVED_COLS = ['PassengerId', 'Name', 'Ticket', 'Cabin']

## IMPORTING DATASET
dataset = pd.read_csv(DATASET_PATH)
dataset = dataset.drop(REMOVED_COLS, axis=1)

"""Handle missing values, encode categorical features, etc. (example)"""
dataset.dropna(inplace=True)
dataset['Sex'] = dataset['Sex'].map({'male': 0, 'female': 1})
dataset['Embarked'] = dataset['Embarked'].map({'C': 0, 'Q': 1, 'S': 2})

X = dataset.drop(columns=['Survived']).values
y = dataset['Survived'].values

dataset = tf.data.Dataset.from_tensor_slices((X, y))
dataset = dataset.shuffle(buffer_size=len(dataset))  # Shuffle dataset
dataset = dataset.batch(16)
print(dataset.element_spec)

# Add an extra dimension to each element in the dataset
def add_extra_dim(X, y):
    X = tf.expand_dims(X, axis=-1)
    return X, y

# Apply the transformation to the dataset
dataset = dataset.map(add_extra_dim)

## SPLIT
train_size = int(0.8 * len(dataset))
test_size = len(dataset) - train_size

train_dataset = dataset.take(train_size)
test_dataset = dataset.skip(train_size)

class ModelBuilder(tf.keras.Model):
    def __init__(self, name='Classifier', **kwargs):
        super(ModelBuilder, self).__init__(name=name)
        
        self.fir_conv1D = tf.keras.layers.Conv1D(
            filters=32,
            kernel_size = 3,
            data_format='channels_last',
            input_shape = (16,7,1),
            activation='elu',
            name='conv1'
        )
        self.sec_conv1D = tf.keras.layers.Conv1D(
            filters=64,
            kernel_size = 3,
            data_format='channels_last',
            activation='elu',
            name='conv2'
        )
        self.bn = tf.keras.layers.BatchNormalization(name='norm')
        self.flatten = tf.keras.layers.Flatten(name='flattening')
        self.dense1 = tf.keras.layers.Dense(
            units=128,
            name='Dense1',
            activation='relu'
        )
        self.dense2 = tf.keras.layers.Dense(
            units=16,
            name='Dense2',
            activation='relu'
        )
        self.output_layer = tf.keras.layers.Dense(
            units=1,
            name='Output',
            activation='sigmoid'
        )
        
    def call(self, inputs):
        x = self.fir_conv1D(inputs)
        x = self.sec_conv1D(x)
        x = self.bn(x)
        x = self.flatten(x)
        x = self.dense1(x)
        x = self.dense2(x)
        x = self.output_layer(x)
        
        return x

with open('C:\\Users\\ritesh\\Desktop\\CodeSoft\\model\\config.json', 'r') as f:
    data = json.load(f)
files = data[0]['files']
model = ModelBuilder()
model.load_weights(files['model'])
#model = ModelBuilder()
#model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=0.01), loss=tf.keras.losses.BinaryCrossentropy(from_logits=False), metrics=['accuracy'])
for batch_X, batch_y in dataset.take(1):
    model.build(input_shape=batch_X.shape)
print(model.summary())
#history = model.fit(train_dataset, validation_data=test_dataset, epochs=1500)
#model.save_weights(files['model'])
#with open(files["history"], 'wb') as f:
#    pickle.dump(history.history, f)


def plot_curves():
    with open('C:\\Users\\ritesh\\Desktop\\CodeSoft\\model\\config.json', 'r') as f:
        data = json.load(f)
    files = data[0]['files']
    with open(files["history"], 'rb') as f:
        history = pickle.load(f)
        
    plt.style.use('dark_background')
    fig, ax = plt.subplots(1, 2, figsize=(12, 5))
    fig.suptitle('Accuracy and Loss Curves of Deep Learning Model', fontsize=15)
    fig.tight_layout(pad=3.0)
    subtitles = ['Loss Curve', 'Accuracy Curve']

    ax[0].plot(history['loss'], color='crimson', label='loss', linewidth=1.3)
    ax[0].plot(history['val_loss'], color='dodgerblue', label='val_loss', linestyle='--', linewidth=0.44)
    ax[0].set_ylabel('Loss')
    ax[1].plot(history['accuracy'], color='crimson', label='accuracy', linewidth=1.3)
    ax[1].plot(history['val_accuracy'], color='dodgerblue', label='val_accuracy', linestyle='--', linewidth=0.44)
    ax[1].set_ylabel('Accuracy')

    max_ac = max(history['val_accuracy']) 
    max_idx = history['val_accuracy'].index(max_ac)
    min_ls = min(history['val_loss'])
    min_idx = history['val_loss'].index(min_ls)
    
    ax[1].annotate(f'({max_idx},{max_ac:.3f})',
                   xy=(max_idx, max_ac), xytext=(max_idx-150,max_ac-0.025),
                   color='seagreen',
                   textcoords='data')
    ax[0].annotate(f'({min_idx},{min_ls:.3f})',
                   xy=(min_idx, min_ls), xytext=(min_idx-330,min_ls-0.032),
                   color='red',
                   textcoords='data')
    ax[1].plot(max_idx, max_ac, 'o', color='seagreen')
    ax[0].plot(min_idx, min_ls, 'ro')

    for i in range(2):
        ax[i].grid(color='gray', linestyle='--', linewidth=0.5)
        ax[i].set_title(subtitles[i], color='dimgray', fontsize=12)
        ax[i].set_xlabel('Epochs')

    fig.legend(['training curves', 'validation curves'], 
               bbox_to_anchor=(0.68, 0.93),
               ncol=2)
    plt.savefig(files['ann'], transparent=True)
    
plot_curves()
