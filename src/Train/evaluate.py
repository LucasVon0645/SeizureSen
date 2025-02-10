import os
import sys

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../../")))

from src.Train.ModelTrainer import ModelTrainer

#? Change the config path to the desired model configuration
config_path = os.path.join("results", "model_without_attention_v2_5s_slices_dogs_1_2_SMOTE_early_exits", "model_config.json")
#? Change the data path to the desired preprocessed data
data_path = os.path.join("data", "preprocessed", "Dog_1_2_5s_slices_aug")

#? Change the model class to the desired model
trainer = ModelTrainer(cfg_path=config_path, data_directory=data_path)

#? Change the preprocessed filenames to the desired preprocessed data
preprocessed_filenames = {
    "freq_train": "freq_domain_train.npz",
    "freq_test": "freq_domain_test.npz",
    "time_train": "time_domain_train.npz",
    "time_test": "time_domain_test.npz",
}

trainer.load_data(preprocessed_filenames, only_test_data=True)

print("\n\nConfiguration settings")
print(trainer.config)

print("\n\nData for testing")
print("X_test_freq: ", trainer.X_test_freq.shape)
print("X_test_time: ", trainer.X_test_time.shape)
print("y_test: ", trainer.y_test.shape)

# weights_path = os.path.join("models", "model_without_attention", "checkpoint_full_dataset", "best_model.keras")
# scalers_path = os.path.join("models", "model_without_attention", "feature_scalers.pkl")
# trainer.load_model(weights_path, scalers_path)

trainer.load_model()
# trainer.optimal_threshold = 0.7
trainer.evaluate(save_test_pred=False, use_optimal_threshold = False)
# trainer.evaluate(save_test_pred=True, use_optimal_threshold = True)