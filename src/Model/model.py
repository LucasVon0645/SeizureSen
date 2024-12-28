from keras.callbacks import EarlyStopping
from keras.models import Model
from keras.layers import Dense, Dropout, Flatten, Input, concatenate, Conv2D
from keras.regularizers import l2
from keras.optimizers import SGD


class MultiViewConvModel:

    @classmethod
    def get_model(cls, setting, channels, fft_bins, pca_bins, steps):
        """
        Builds and compiles a convolutional neural network model for EEG data analysis.
        This model consists of two input branches, each processing different features of the EEG data.
        The first branch processes PCA-transformed data, while the second branch processes FFT-transformed data.
        The branches are then merged and passed through several dense layers before producing the final output.
        Args:
            setting (object): An object containing various hyperparameters and settings for the model.
            channels (int): Number of EEG channels.
            fft_bins (int): Number of FFT bins.
            pca_bins (int): Number of PCA bins.
            steps (int): Number of consecutive slices of EEG data to be analyzed by the neural network.
        Returns:
            model (tf.keras.Model): The compiled Keras model.
            early_stopping (tf.keras.callbacks.EarlyStopping): EarlyStopping callback configured for the model.
        Model Inputs:
            - input1: PCA-transformed EEG data with shape (channels * pca_bins, steps, 1).
            - input2: FFT-transformed EEG data with shape (channels * fft_bins, steps, 1).
        Model Outputs:
            - final_output: Final classification output with softmax activation for multi-class classification.
            - early_exit1: Early exit classification output from the first branch.
            - early_exit2: Early exit classification output from the first branch after additional convolution.
            - early_exit3: Early exit classification output from the second branch.
            - early_exit4: Early exit classification output from the second branch after additional convolution.
        """

        # Define first input branch
        input1 = Input(shape=(channels * pca_bins, steps, 1), name="input1")
        seq1 = Conv2D(
            filters=setting.nb_filter,
            kernel_size=(channels * pca_bins, 1),
            kernel_initializer="lecun_uniform",
            kernel_regularizer=l2(l=setting.l2),
            activation="relu",
        )(input1)
        seq1 = Dropout(setting.dropout)(seq1)
        early_exit1 = Dense(2, activation="softmax", name="early_exit1")(
            Flatten()(seq1)
        )

        seq1 = Conv2D(
            filters=setting.nb_filter,
            kernel_size=(1, 3),
            kernel_regularizer=l2(l=setting.l2),
            kernel_initializer="lecun_uniform",
            activation="relu",
        )(seq1)
        seq1 = Dropout(setting.dropout)(seq1)
        early_exit2 = Dense(2, activation="softmax", name="early_exit2")(
            Flatten()(seq1)
        )

        seq1 = Flatten()(seq1)
        output1 = Dense(setting.output1, activation="tanh")(seq1)

        # Define second input branch
        input2 = Input(shape=(channels * fft_bins, steps, 1), name="input2")
        seq2 = Conv2D(
            filters=setting.nb_filter,
            kernel_size=(channels * fft_bins, 1),
            kernel_regularizer=l2(l=setting.l2),
            kernel_initializer="lecun_uniform",
            activation="relu",
        )(input2)
        seq2 = Dropout(setting.dropout)(seq2)
        early_exit3 = Dense(2, activation="softmax", name="early_exit3")(
            Flatten()(seq2)
        )

        seq2 = Conv2D(
            filters=setting.nb_filter,
            kernel_size=(1, 3),
            kernel_regularizer=l2(l=setting.l2),
            kernel_initializer="lecun_uniform",
            activation="relu",
        )(seq2)
        seq2 = Dropout(setting.dropout)(seq2)
        early_exit4 = Dense(2, activation="softmax", name="early_exit4")(
            Flatten()(seq2)
        )

        seq2 = Flatten()(seq2)
        output2 = Dense(setting.output2, activation="tanh")(seq2)

        # Merge both branches
        merged = concatenate([output1, output2])
        merged = Dense(512, activation="tanh")(merged)
        merged = Dense(256, activation="tanh")(merged)
        merged = Dense(128, activation="tanh")(merged)

        # Add final classification layer with softmax activation for multi-class classification
        output = Dense(2, activation="softmax", name="final_output")(merged)

        # Define the model
        model = Model(
            inputs=[input1, input2],
            outputs=[output, early_exit1, early_exit2, early_exit3, early_exit4],
        )

        # Choose optimizer based on the setting
        sgd = SGD(lr=setting.lr)
        if setting.name == "Patient_1":
            model.compile(
                loss="categorical_crossentropy", optimizer="adam", metrics=["accuracy"]
            )
        else:
            model.compile(
                loss="categorical_crossentropy", optimizer=sgd, metrics=["accuracy"]
            )

        # Configure EarlyStopping callback
        early_stopping = EarlyStopping(
            monitor="val_loss",  # Metric to monitor
            patience=5,  # Number of epochs to wait for improvement
            restore_best_weights=True,  # Restore model weights from the best epoch
            verbose=1,  # Print messages when early stopping is triggered
        )

        # Return the model and the early stopping callback
        return model, early_stopping
