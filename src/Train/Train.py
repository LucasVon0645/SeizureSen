import os
import sys

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../../")))

from src.Train.ModelTrainer import ModelTrainer

config_path = os.path.join("models", "config", "std_time_test_cfg.json")
data_path = os.path.join("data", "preprocessed", "Dog_1")

trainer = ModelTrainer(cfg_path=config_path, data_directory=data_path)

preprocessed_filenames = {
    "freq_train": "freq_domain_train_v2.npz",
    "freq_test": "freq_domain_test_v2.npz",
    "time_train": "time_domain_train_v2.npz",
    "time_test": "time_domain_test_v2.npz",
}

trainer.load_data(file_names_dict=preprocessed_filenames)

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

trainer.train()
trainer.evaluate()
