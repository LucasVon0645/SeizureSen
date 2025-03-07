import os
import sys

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../../")))

from src.Train.ModelTrainer import ModelTrainer

#? Change the config path to the desired model configuration
config_path = os.path.join("results", "model_without_attention_v2_5s_slices_dogs_1_2_SMOTE", "model_config.json")
#? Change the data path to the desired preprocessed data
data_path = os.path.join("data", "preprocessed", "Dog_1_2_5s_slices_aug")

#? Change the threshold_tunning to True if you want to tune the threshold for the model
threshold_tunning = False

#? Change the model class to the desired model
trainer = ModelTrainer(cfg_path=config_path, data_directory=data_path)

#? Change the preprocessed filenames to the desired preprocessed data

preprocessed_filenames = {
    "freq_train": "freq_domain_train.npz",
    "freq_test": "freq_domain_test.npz",
    "time_train": "time_domain_train.npz",
    "time_test": "time_domain_test.npz",
}

trainer.load_data(preprocessed_filenames)

print("\n\nConfiguration settings")
print(trainer.config)

print("\n\nData for training")
print("X_train_freq: ", trainer.X_train_freq.shape)
print("X_train_time: ", trainer.X_train_time.shape)
print("y_train: ", trainer.y_train.shape)
print("\n\nData for testing")
print("X_test_freq: ", trainer.X_test_freq.shape)
print("X_test_time: ", trainer.X_test_time.shape)
print("y_test: ", trainer.y_test.shape)

trainer.train(threshold_tunning=threshold_tunning)
trainer.evaluate(save_test_pred=True, use_optimal_threshold=threshold_tunning)
trainer.train_with_cross_validation()
