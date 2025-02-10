from keras.api.callbacks import EarlyStopping
from keras.api.models import Model
from keras.api.layers import Dense, Dropout, Flatten, Input, BatchNormalization, Conv2D, Activation, concatenate
from keras.api.optimizers import Adam
from keras.api.regularizers import L2

class MultiViewConvModelWithBatchNorm:
    """
    A class to build a multi-view convolutional neural network model
    for EEG data analysis.
    """

    @classmethod
    def get_model(cls, config: dict):
        """
        Builds and compiles a convolutional neural network model for EEG data analysis.
        This architecture consists of two input branches, one processing PCA-transformed data and
        the other processing FFT-transformed data. The branches are then merged and passed through
        several dense layers before producing the final output.
        Convolutional layers are followed by batch normalization and dropout layers to improve
        the model's generalization and performance. In total, the model has:
        - 3 convolutional layers for each input branch
        - 2 batch normalization layers for each input branch
        - 3 dropout layers for each input branch
        - 3 dense layers after merging the branches
        
        Args:
            config (dict): A dictionary containing various hyperparameters
                           and settings for the model.
        """
        channels = config["channels"]
        pca_bins = config["pca_bins"]
        fft_bins = config["fft_bins"]
        steps = config["model_time_steps"]

        # First input branch (time domain)
        input_time = Input(shape=(channels * pca_bins, steps, 1), name="time_domain_input")

        seq1 = Conv2D(filters=config["nb_filter"], kernel_size=(channels * pca_bins, 1),
                      kernel_initializer="lecun_uniform", kernel_regularizer=L2(config["l2"]),
                      name="time_conv_layer1")(input_time)
        seq1 = BatchNormalization(name="time_bn_layer1")(seq1)
        seq1 = Activation("relu", name="time_activation_layer1")(seq1)
        seq1 = Dropout(config["dropout"], name="time_dropout_layer1")(seq1)

        seq1 = Conv2D(filters=config["nb_filter"], kernel_size=(1, 16),
                      kernel_initializer="lecun_uniform", kernel_regularizer=L2(config["l2"]),
                      name="time_conv_layer2")(seq1)
        seq1 = BatchNormalization(name="time_bn_layer2")(seq1)
        seq1 = Activation("relu", name="time_activation_layer2")(seq1)
        seq1 = Dropout(config["dropout"], name="time_dropout_layer2")(seq1)

        seq1 = Conv2D(filters=config["nb_filter"], kernel_size=(1, 16),
                      kernel_initializer="lecun_uniform", kernel_regularizer=L2(config["l2"]),
                      name="time_conv_layer3")(seq1)
        seq1 = Activation("relu", name="time_activation_layer3")(seq1)
        seq1 = Dropout(config["dropout"], name="time_dropout_layer3")(seq1)

        seq1 = Flatten(name="time_flatten_layer")(seq1)
        output1 = Dense(config["nn_time_output"], activation="tanh", name="cnn_time_output")(seq1)

        # Second input branch (frequency domain)
        input_freq = Input(shape=(channels * fft_bins, steps, 1), name="freq_domain_input")

        seq2 = Conv2D(filters=config["nb_filter"], kernel_size=(channels * fft_bins, 1),
                      kernel_initializer="lecun_uniform", kernel_regularizer=L2(config["l2"]),
                      name="freq_conv_layer1")(input_freq)
        seq2 = BatchNormalization(name="freq_bn_layer1")(seq2)
        seq2 = Activation("relu", name="freq_activation_layer1")(seq2)
        seq2 = Dropout(config["dropout"], name="freq_dropout_layer1")(seq2)

        seq2 = Conv2D(filters=config["nb_filter"], kernel_size=(1, 16),
                      kernel_initializer="lecun_uniform", kernel_regularizer=L2(config["l2"]),
                      name="freq_conv_layer2")(seq2)
        seq2 = BatchNormalization(name="freq_bn_layer2")(seq2)
        seq2 = Activation("relu", name="freq_activation_layer2")(seq2)
        seq2 = Dropout(config["dropout"], name="freq_dropout_layer2")(seq2)

        seq2 = Conv2D(filters=config["nb_filter"], kernel_size=(1, 16),
                      kernel_initializer="lecun_uniform", kernel_regularizer=L2(config["l2"]),
                      name="freq_conv_layer3")(seq2)
        seq2 = Activation("relu", name="freq_activation_layer3")(seq2)
        seq2 = Dropout(config["dropout"], name="freq_dropout_layer3")(seq2)

        seq2 = Flatten(name="freq_flatten_layer")(seq2)
        output2 = Dense(config["nn_freq_output"], activation="tanh", name="cnn_freq_output")(seq2)

        # Merge both branches
        merged = concatenate([output1, output2])
        merged = Dense(512, activation="tanh", name="fcn_layer1")(merged)
        merged = Dense(256, activation="tanh", name="fcn_layer2")(merged)
        merged = Dense(128, activation="tanh", name="fcn_layer3")(merged)

        # Final classification layer
        output = Dense(2, activation="softmax", name="final_output")(merged)

        # Model definition
        cnn_model = Model(inputs=[input_time, input_freq], outputs=[output])
        cnn_model.compile(
            loss="categorical_crossentropy",
            optimizer=Adam(learning_rate=config["learning_rate"]),
            metrics=["accuracy"]
        )
        
        # Configure EarlyStopping callback
        early_stopping = EarlyStopping(
            monitor="val_loss",  # Metric to monitor
            patience=5,  # Number of epochs to wait for improvement
            restore_best_weights=True,  # Restore model weights from the best epoch
            verbose=1,  # Print messages when early stopping is triggered
        )

        return cnn_model, early_stopping


if __name__ == "__main__":
    # Example configuration
    config_example = {
        "nb_filter": 64,
        "l2": 0.001,
        "dropout": 0.2,
        "learning_rate": 0.01,
        "nn_time_output": 64,
        "nn_freq_output": 64,
        "batch_size": 128,
        "nb_epoch": 100,
        "name": "first_test",
        "model_path": "models/first_test",
        "channels": 16,
        "fft_bins": 16,
        "pca_bins": 16,
        "model_time_steps": 20,
    }

    model, _ = MultiViewConvModelWithBatchNorm.get_model(config_example)
    model.summary()
