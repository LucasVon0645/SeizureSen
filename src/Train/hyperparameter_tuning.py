import os
import tensorflow as tf
from keras_tuner import RandomSearch
from ModelTrainer import ModelTrainer
from src.Model.MultiViewConvModel_v2 import MultiViewConvModel_v2

# Paths
config_path = "models/model_without_attention_smote_5s_slices_dogs_1_2_40steps/model_config.json"
data_path = "data/preprocessed/Dog_1_2_5s_slices"

# Load Data
trainer = ModelTrainer(cfg_path=config_path, data_directory=data_path)
trainer.load_data()

# Time Domain
trainer.X_train_time = tf.reshape(
    trainer.X_train_time,
    (-1, trainer.config["channels"] * trainer.config["pca_bins"], trainer.config["model_time_steps"], 1)
)
trainer.X_test_time = tf.reshape(
    trainer.X_test_time,
    (-1, trainer.config["channels"] * trainer.config["pca_bins"], trainer.config["model_time_steps"], 1)
)

# Frequency Domain
trainer.X_train_freq = tf.reshape(
    trainer.X_train_freq,
    (-1, trainer.config["channels"] * trainer.config["fft_bins"], trainer.config["model_time_steps"], 1)
)
trainer.X_test_freq = tf.reshape(
    trainer.X_test_freq,
    (-1, trainer.config["channels"] * trainer.config["fft_bins"], trainer.config["model_time_steps"], 1)
)

# Verify Data Shapes
print("Time Domain Train Shape:", trainer.X_train_time.shape)
print("Freq Domain Train Shape:", trainer.X_train_freq.shape)

def build_model(hp):
    model, _ = MultiViewConvModel_v2.get_model(trainer.config, hp=hp)
    learning_rate = trainer.config["learning_rate"]  # Get learning rate from config

    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=learning_rate),
        loss='binary_crossentropy',
        metrics=['accuracy']
    )
    return model

# Initialize RandomSearch tuner
tuner = RandomSearch(
    build_model,
    objective='val_accuracy',
    max_trials=20,
    executions_per_trial=2,
    directory='hyperparameter_tuning',
    project_name='EEG_Optimization'
)

# Run Hyperparameter Tuning
tuner.search(
    {"time_domain_input": trainer.X_train_time, "freq_domain_input": trainer.X_train_freq},
    trainer.y_train,
    validation_data=(
        {"time_domain_input": trainer.X_test_time, "freq_domain_input": trainer.X_test_freq},
        trainer.y_test
    ),
    epochs=100,
    batch_size=trainer.config['batch_size'],
    verbose=2
)

# Get Best Hyperparameters
best_hps = tuner.get_best_hyperparameters(1)[0]

print(f"""
The best hyperparameters are:
- nb_filter: {best_hps.get('nb_filter')}
- l2: {best_hps.get('l2')}
- dropout: {best_hps.get('dropout')}
- learning_rate: {best_hps.get('learning_rate')}
""")
