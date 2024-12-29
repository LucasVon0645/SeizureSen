from keras.callbacks import EarlyStopping
from keras.models import Model
from keras.layers import Dense, Dropout, Flatten, Input, concatenate, Conv2D
from keras.optimizers import SGD
from keras.regularizers import L2


class MultiViewConvModel:

    @classmethod
    def get_model(cls, config, channels, fft_bins, pca_bins, steps):
        """
        Builds and compiles a convolutional neural network model
        for EEG data analysis.
        This model consists of two input branches, each processing
        different features of the EEG data.
        The first branch processes PCA-transformed data, while the
        second branch processes FFT-transformed data.
        The branches are then merged and passed through several
        dense layers before producing the final output.
        Args:
            config (dict): A dictionary containing various hyperparameters
                              and settings for the model.
            channels (int): Number of EEG channels.
            fft_bins (int): Number of FFT bins.
            pca_bins (int): Number of PCA bins.
            steps (int): Number of consecutive slices of EEG data to be
                         analyzed by the neural network.
        Returns:
            model (tf.keras.Model): The compiled Keras model.
            early_stopping (tf.keras.callbacks.EarlyStopping): EarlyStopping
                                                               callback configured for the model.
        Model Inputs:
            - input1: PCA-transformed EEG data with shape (channels * pca_bins, steps, 1).
            - input2: FFT-transformed EEG data with shape (channels * fft_bins, steps, 1).
        Model Outputs:
            - final_output: Final classification output with sigmoid
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

        # Define first input branch for the time domain
        # The last dimension of the input shape is 1 because the input data is single-channel (like grayscale image).
        input1 = Input(shape=(channels * pca_bins, steps, 1), name="time_domain_input")
        seq1 = Conv2D(
            filters=config["nb_filter"],
            kernel_size=(channels * pca_bins, 1),
            kernel_initializer="lecun_uniform",
            kernel_regularizer=L2(config["l2"]),
            activation="relu",
        )(input1)
        seq1 = Dropout(config["dropout"])(seq1)
        early_exit1 = Dense(2, activation="sigmoid", name="early_exit1")(
            Flatten()(seq1)
        )

        seq1 = Conv2D(
            filters=config["nb_filter"],
            kernel_size=(1, 3),
            kernel_regularizer=L2(config["l2"]),
            kernel_initializer="lecun_uniform",
            activation="relu",
        )(seq1)
        seq1 = Dropout(config["dropout"])(seq1)
        early_exit2 = Dense(2, activation="sigmoid", name="early_exit2")(
            Flatten()(seq1)
        )

        seq1 = Flatten()(seq1)
        output1 = Dense(config["nn_time_output"], activation="tanh")(seq1)

        # Define second input branch for the frequency domain
        # The last dimension of the input shape is 1 because the input data is single-channel (like grayscale image).
        input2 = Input(shape=(channels * fft_bins, steps, 1), name="freq_domain_input")
        seq2 = Conv2D(
            filters=config["nb_filter"],
            kernel_size=(channels * fft_bins, 1),
            kernel_regularizer=L2(config["l2"]),
            kernel_initializer="lecun_uniform",
            activation="relu",
        )(input2)
        seq2 = Dropout(config["dropout"])(seq2)
        early_exit3 = Dense(2, activation="sigmoid", name="early_exit3")(
            Flatten()(seq2)
        )

        seq2 = Conv2D(
            filters=config["nb_filter"],
            kernel_size=(1, 3),
            kernel_regularizer=L2(config["l2"]),
            kernel_initializer="lecun_uniform",
            activation="relu",
        )(seq2)
        seq2 = Dropout(config["dropout"])(seq2)
        early_exit4 = Dense(2, activation="sigmoid", name="early_exit4")(
            Flatten()(seq2)
        )

        seq2 = Flatten()(seq2)
        output2 = Dense(config["nn_freq_output"], activation="tanh")(seq2)

        # Merge both branches
        merged = concatenate([output1, output2])
        merged = Dense(512, activation="tanh")(merged)
        merged = Dense(256, activation="tanh")(merged)
        merged = Dense(128, activation="tanh")(merged)

        # Add final classification layer with sigmoid activation for multi-class classification
        output = Dense(2, activation="sigmoid", name="final_output")(merged)

        # Define the model
        cnn_model = Model(
            inputs=[input1, input2],
            outputs=[output, early_exit1, early_exit2, early_exit3, early_exit4],
        )

        # Choose optimizer based on the setting
        sgd = SGD(learning_rate=config["lr"])
        cnn_model.compile(loss="binary_crossentropy", optimizer=sgd, metrics=["accuracy"])

        # Configure EarlyStopping callback
        early_stopping = EarlyStopping(
            monitor="val_loss",  # Metric to monitor
            patience=5,  # Number of epochs to wait for improvement
            restore_best_weights=True,  # Restore model weights from the best epoch
            verbose=1,  # Print messages when early stopping is triggered
        )

        # Return the model and the early stopping callback
        return cnn_model, early_stopping


if __name__ == "__main__":
    
    # Initialize the model
    model, _ = MultiViewConvModel.get_model(
        setting=Setting(), channels=16, fft_bins=8, pca_bins=16, steps=4
    )

    # Display the model summary
    model.summary()
