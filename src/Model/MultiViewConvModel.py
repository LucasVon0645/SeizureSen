from keras.api.callbacks import EarlyStopping
from keras.api.models import Model
from keras.api.layers import Dense, Dropout, Flatten, Input, concatenate, Conv2D
from keras.api.optimizers import SGD
from keras.api.regularizers import L2
from src.Model.losses import focal_loss, f1_loss


class MultiViewConvModel:
    """
    A class to build a multi-view convolutional neural network model
    for EEG data analysis.
    This model consists of two input branches, each processing
    different features of the EEG data.
    The first branch processes PCA-transformed data, while the
    second branch processes FFT-transformed data.
    The branches are then merged and passed through several dense layers
    before producing the final output.
    """

    @classmethod
    def get_model(cls, config: dict):
        """
        Builds and compiles a convolutional neural network model for EEG data analysis.
        Args:
            config (dict): A dictionary containing various hyperparameters
                              and settings for the model.
        Returns:
            model (tf.keras.Model): The compiled Keras model.
            early_stopping (tf.keras.callbacks.EarlyStopping): EarlyStopping callback configured
            for the model.
        Model Inputs:
            - input_time: PCA-transformed EEG data with shape (channels * pca_bins, steps, 1).
            - input_freq: FFT-transformed EEG data with shape (channels * fft_bins, steps, 1).
        Model Outputs:
            - final_output: Final classification output with softmax
                            activation for multi-class classification.
            - early_exit1: Early exit classification output from the
                           first branch.
            - early_exit2: Early exit classification output from the
                           first branch after additional convolution.
            - early_exit3: Early exit classification output from
                           the second branch.
            - early_exit4: Early exit classification output from the second
                           branch after additional convolution.
        """

        # Define the input shapes for the model
        channels = config["channels"]
        pca_bins = config["pca_bins"]
        fft_bins = config["fft_bins"]
        steps = config["model_time_steps"]
        
        use_early_exits = config.get("use_early_exits", False)

        # Define first input branch for the time domain
        # The last dimension of the input shape is 1 because the input data
        # is single-channel (like grayscale image)
        input_time = Input(shape=(channels * pca_bins, steps, 1), name="time_domain_input")

        seq1 = Conv2D(
            filters=config["nb_filter"],
            kernel_size=(channels * pca_bins, 1),
            kernel_initializer="lecun_uniform",
            kernel_regularizer=L2(config["l2"]),
            activation="relu",
            name="time_conv_layer1")(input_time)

        seq1 = Dropout(config["dropout"], name="time_dropout_layer1")(seq1)

        early_exit1 = Dense(2, activation="softmax", name="early_exit1")(Flatten()(seq1))

        seq1 = Conv2D(
            filters=config["nb_filter"],
            kernel_size=(1, 3),
            kernel_regularizer=L2(config["l2"]),
            kernel_initializer="lecun_uniform",
            activation="relu",
            name="time_conv_layer2")(seq1)

        seq1 = Dropout(config["dropout"], name="time_dropout_layer2")(seq1)

        early_exit2 = Dense(2, activation="softmax", name="early_exit2")(Flatten()(seq1))

        seq1 = Flatten(name="time_flatten_layer")(seq1)

        output1 = Dense(config["nn_time_output"], activation="tanh", name="cnn_time_output")(seq1)

        # Define second input branch for the frequency domain
        # The last dimension of the input shape is 1 because the input data
        # is single-channel (like grayscale image).
        input_freq = Input(shape=(channels * fft_bins, steps, 1), name="freq_domain_input")
        seq2 = Conv2D(
            filters=config["nb_filter"],
            kernel_size=(channels * fft_bins, 1),
            kernel_regularizer=L2(config["l2"]),
            kernel_initializer="lecun_uniform",
            activation="relu",
            name="freq_conv_layer1")(input_freq)

        seq2 = Dropout(config["dropout"], name="freq_dropout_layer1")(seq2)

        early_exit3 = Dense(2, activation="softmax", name="early_exit3")(Flatten()(seq2))

        seq2 = Conv2D(
            filters=config["nb_filter"],
            kernel_size=(1, 3),
            kernel_regularizer=L2(config["l2"]),
            kernel_initializer="lecun_uniform",
            activation="relu",
            name="freq_conv_layer2")(seq2)

        seq2 = Dropout(config["dropout"], name="freq_dropout_layer2")(seq2)

        early_exit4 = Dense(2, activation="softmax", name="early_exit4")(Flatten()(seq2))

        seq2 = Flatten(name="freq_flatten_layer")(seq2)

        output2 = Dense(config["nn_freq_output"], activation="tanh", name="cnn_freq_output")(seq2)

        # Merge both branches
        merged = concatenate([output1, output2])
        merged = Dense(512, activation="tanh", name="fcn_layer1")(merged)
        merged = Dense(256, activation="tanh", name="fcn_layer2")(merged)
        merged = Dense(128, activation="tanh", name="fcn_layer3")(merged)

        # Add final classification layer with softmax activation for multi-class classification
        output = Dense(2, activation="softmax", name="final_output")(merged)

        loss_function = config.get("loss", "categorical_crossentropy")  # Default to categorical_crossentropy

        if loss_function == "focal":
            loss_fn = focal_loss(gamma=2.0, alpha=0.25)
        elif loss_function == "f1":
            loss_fn = f1_loss
        else:
            loss_fn = "categorical_crossentropy"

        # Model definition
        if use_early_exits:
            # Create a model with early exits
            cnn_model = Model(inputs=[input_time, input_freq], outputs=[output, early_exit1, early_exit2, early_exit3, early_exit4])
            cnn_model.compile(
                optimizer=SGD(learning_rate=config["learning_rate"]),
                loss={
                    "final_output": loss_fn,
                    "early_exit1": loss_fn,
                    "early_exit2": loss_fn,
                    "early_exit3": loss_fn,
                    "early_exit4": loss_fn,
                },
                # Weights for the losses of the different outputs
                loss_weights={
                    "final_output": 1.0,
                    "early_exit1": 0.3,
                    "early_exit2": 0.4,
                    "early_exit3": 0.8,
                    "early_exit4": 1.0,
                },
                metrics={
                    "final_output": ["accuracy"],
                    "early_exit1": ["accuracy"],
                    "early_exit2": ["accuracy"],
                    "early_exit3": ["accuracy"],
                    "early_exit4": ["accuracy"],
                },
            )
        else:
            cnn_model = Model(inputs=[input_time, input_freq], outputs=[output])
            cnn_model.compile(
                loss=loss_fn,
                optimizer=SGD(learning_rate=config["learning_rate"]),
                metrics=["accuracy"]
            )

        # Configure EarlyStopping callback
        early_stopping = EarlyStopping(
            monitor="val_loss",  # Metric to monitor
            patience=5,  # Number of epochs to wait for improvement
            restore_best_weights=True,  # Restore model weights from the best epoch
            verbose=1,  # Print messages when early stopping is triggered
        )

        print(f"Using Loss Function: {loss_function}")

        # Return the model and the early stopping callback
        return cnn_model, early_stopping


if __name__ == "__main__":
    # Example configuration for the model
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
        "fft_bins": 8,
        "pca_bins": 16,
        "model_time_steps": 4,
        "preictal_class_weight": 5.0,
    }

    # Initialize the model
    model, _ = MultiViewConvModel.get_model(config_example)

    # Display the model summary
    model.summary()
