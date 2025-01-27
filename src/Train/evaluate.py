import os
import sys

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../../")))

from src.Train.ModelTrainer import ModelTrainer
from src.Model.MultiViewConvModel import MultiViewConvModel
from src.Model.MultiViewConvModelAttention import MultiViewConvModelWithAttention

#? Change the config path to the desired model configuration
config_path = os.path.join("models", "model_without_attention_smote_5s_slices", "model_config.json")
#? Change the data path to the desired preprocessed data
data_path = os.path.join("data", "preprocessed", "Dog_1_5s_slices")

#? Change the model class to the desired model
trainer = ModelTrainer(cfg_path=config_path, data_directory=data_path,
                       model_class=MultiViewConvModel)

#? Change the preprocessed filenames to the desired preprocessed data
# preprocessed_filenames = {
#     "freq_train": "freq_domain_train_augmented_preictal.npz",
#     "freq_test": "freq_domain_test_augmented_preictal.npz",
#     "time_train": "time_domain_train_augmented_preictal.npz",
#     "time_test": "time_domain_test_augmented_preictal.npz",
# }

preprocessed_filenames = {
    "freq_train": "freq_domain_train.npz",
    "freq_test": "freq_domain_test.npz",
    "time_train": "time_domain_train.npz",
    "time_test": "time_domain_test.npz",
}

trainer.load_data(preprocessed_filenames)

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

trainer.evaluate(save_test_pred=True)