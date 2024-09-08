# Install necessary packages
!pip install numpy tensorflow matplotlib scikit-learn tqdm

# Import libraries
import os
import shutil
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
from tensorflow.keras.applications import ResNet50
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.layers import Dense, GlobalAveragePooling2D, Dropout
from tensorflow.keras.models import Model
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, mean_absolute_error, mean_squared_error
from tqdm.keras import TqdmCallback
from google.colab import drive
from sklearn.model_selection import train_test_split

# Mount Google Drive
drive.mount('/content/drive')

# Define the paths to the dataset directory
def get_data_paths():
    base_path = '/content/drive/MyDrive/Dataset'
    return base_path

# Split data into training and validation sets
def split_data(base_path, split_ratio=0.2):
    train_dir = os.path.join(base_path, 'train')
    val_dir = os.path.join(base_path, 'val')
    os.makedirs(train_dir, exist_ok=True)
    os.makedirs(val_dir, exist_ok=True)
    
    categories = ['Anemic', 'Non-anemic']
    
    for category in categories:
        category_path = os.path.join(base_path, category)
        train_category_dir = os.path.join(train_dir, category)
        val_category_dir = os.path.join(val_dir, category)
        os.makedirs(train_category_dir, exist_ok=True)
        os.makedirs(val_category_dir, exist_ok=True)

        all_files = os.listdir(category_path)
        all_files = [os.path.join(category_path, file) for file in all_files]

        train_files, val_files = train_test_split(all_files, test_size=split_ratio, random_state=42)

        for file in train_files:
            shutil.copy(file, train_category_dir)
        for file in val_files:
            shutil.copy(file, val_category_dir)

# Load and preprocess the data
def create_data_generators(train_dir, val_dir, img_height=224, img_width=224, batch_size=32):
    train_datagen = ImageDataGenerator(
        rescale=1./255,
        rotation_range=30,
        width_shift_range=0.3,
        height_shift_range=0.3,
        shear_range=0.3,
        zoom_range=0.3,
        horizontal_flip=True,
        fill_mode='nearest'
    )

    val_datagen = ImageDataGenerator(rescale=1./255)

    train_generator = train_datagen.flow_from_directory(
        train_dir,
        target_size=(img_height, img_width),
        batch_size=batch_size,
        class_mode='binary'
    )

    val_generator = val_datagen.flow_from_directory(
        val_dir,
        target_size=(img_height, img_width),
        batch_size=batch_size,
        class_mode='binary'
    )

    return train_generator, val_generator

# Define the ResNet50 model
def create_model(num_classes):
    base_model = ResNet50(weights='imagenet', include_top=False, input_shape=(224, 224, 3))
    x = base_model.output
    x = GlobalAveragePooling2D()(x)
    x = Dense(1024, activation='relu')(x)
    x = Dropout(0.5)(x)  # Add dropout for regularization
    predictions = Dense(num_classes, activation='sigmoid')(x)

    model = Model(inputs=base_model.input, outputs=predictions)

    # Unfreeze some of the top layers
    for layer in base_model.layers[:143]:  # Unfreeze top layers
        layer.trainable = False
    for layer in base_model.layers[143:]:
        layer.trainable = True

    model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=1e-4),  # Adjust learning rate
                  loss='binary_crossentropy', metrics=['accuracy'])

    return model

# Training the model with progress bar
def train_model(model, train_generator, val_generator, epochs=40, model_save_path='/content/drive/MyDrive/model.h5'):
    history = model.fit(
        train_generator,
        epochs=epochs,
        validation_data=val_generator,
        callbacks=[TqdmCallback(verbose=1), tf.keras.callbacks.EarlyStopping(patience=5, restore_best_weights=True)]  # Add EarlyStopping
    )
    
    # Save the model
    model.save(model_save_path)
    print(f'Model saved to {model_save_path}')
    
    return history

# Load a saved model
def load_model(model_path='/content/drive/MyDrive/model.h5'):
    model = tf.keras.models.load_model(model_path)
    print(f'Model loaded from {model_path}')
    return model

# Evaluate the model
def evaluate_model(model, val_generator):
    val_steps = val_generator.samples // val_generator.batch_size
    val_generator.reset()
    predictions = model.predict(val_generator, steps=val_steps, verbose=1)
    y_pred = (predictions > 0.5).astype(int)
    y_true = val_generator.classes

    if len(y_pred) != len(y_true):
        print(f"Inconsistent sample size: Predicted {len(y_pred)}, True {len(y_true)}")
        min_len = min(len(y_pred), len(y_true))
        y_pred = y_pred[:min_len]
        y_true = y_true[:min_len]

    y_true = y_true.flatten()
    y_pred = y_pred.flatten()

    # Metrics
    mae = mean_absolute_error(y_true, y_pred)
    mse = mean_squared_error(y_true, y_pred)
    recall = recall_score(y_true, y_pred)
    precision = precision_score(y_true, y_pred)
    f1 = f1_score(y_true, y_pred)

    print(f'Mean Absolute Error: {mae:.4f}')
    print(f'Mean Squared Error: {mse:.4f}')
    print(f'Recall: {recall:.4f}')
    print(f'Precision: {precision:.4f}')
    print(f'F1 Score: {f1:.4f}')

# Plot training history and save to file
def plot_history(history, plot_save_path='/content/drive/MyDrive/history_plot.png'):
    plt.figure(figsize=(12, 6))
    
    plt.subplot(1, 2, 1)
    plt.plot(history.history['accuracy'])
    plt.plot(history.history['val_accuracy'])
    plt.title('Model Accuracy')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.legend(['Train', 'Validation'], loc='upper left')
    
    plt.subplot(1, 2, 2)
    plt.plot(history.history['loss'])
    plt.plot(history.history['val_loss'])
    plt.title('Model Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend(['Train', 'Validation'], loc='upper left')
    
    plt.tight_layout()
    plt.savefig(plot_save_path)  # Save the plot
    print(f'History plot saved to {plot_save_path}')
    plt.show()

# Main function
def main():
    base_path = get_data_paths()
    split_data(base_path, split_ratio=0.2)  # Split data into train and validation sets
    
    train_dir = os.path.join(base_path, 'train')
    val_dir = os.path.join(base_path, 'val')
    
    train_generator, val_generator = create_data_generators(train_dir, val_dir)

    num_classes = 1  # For binary classification

    model = create_model(num_classes)
    history = train_model(model, train_generator, val_generator)
    evaluate_model(model, val_generator)
    plot_history(history)

    # To load the model and continue training or evaluation
    # model = load_model()  # Uncomment this if you need to load a saved model

if __name__ == '__main__':
    main()
