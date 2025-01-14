from keras.api.callbacks import EarlyStopping
from keras.api.models import Model
from keras.api.layers import Dense, Dropout, Flatten, Input, concatenate, Conv2D
from keras.api.optimizers import SGD
from keras.api.regularizers import L2
from keras.api.layers import GlobalAveragePooling2D, GlobalMaxPooling2D, Reshape, multiply, add, Activation


def channel_attention(input_feature, ratio=8):
    """
    Implements a channel attention mechanism.
    Args:
        input_feature (tf.Tensor): Input feature map (must be 4D).
        ratio (int): Reduction ratio for the dense layer.
    Returns:
        tf.Tensor: Output feature map after applying channel attention.
    """
    # Ensure the input is 4D
    if len(input_feature.shape) != 4:
        raise ValueError(f"Input feature must be 4D, but got shape {input_feature.shape}")

    channel = input_feature.shape[-1]

    shared_dense_one = Dense(channel // ratio, activation='relu', use_bias=True, kernel_initializer='he_normal')
    shared_dense_two = Dense(channel, activation='sigmoid', use_bias=True, kernel_initializer='he_normal')

    avg_pool = GlobalAveragePooling2D()(input_feature)
    avg_pool = Reshape((1, 1, channel))(avg_pool)
    avg_pool = shared_dense_one(avg_pool)
    avg_pool = shared_dense_two(avg_pool)

    max_pool = GlobalMaxPooling2D()(input_feature)
    max_pool = Reshape((1, 1, channel))(max_pool)
    max_pool = shared_dense_one(max_pool)
    max_pool = shared_dense_two(max_pool)

    cbam_feature = add([avg_pool, max_pool])
    cbam_feature = Activation('sigmoid')(cbam_feature)

    return multiply([input_feature, cbam_feature])


class MultiViewConvModelWithAttention:
    """
    Extends the MultiViewConvModel by adding channel attention mechanisms.
    """

    @classmethod
    def get_model(cls, config: dict):
        channels = config["channels"]
        pca_bins = config["pca_bins"]
        fft_bins = config["fft_bins"]
        steps = config["model_time_steps"]

        # Time domain input branch
        input_time = Input(shape=(channels * pca_bins, steps, 1), name="time_domain_input")
        seq1 = Conv2D(filters=config["nb_filter"], kernel_size=(channels * pca_bins, 1),
                      kernel_initializer="lecun_uniform",
                      kernel_regularizer=L2(config["l2"]),
                      activation="relu", name="time_conv_layer1")(input_time)
        seq1 = channel_attention(seq1)
        seq1 = Dropout(config["dropout"], name="time_dropout_layer1")(seq1)

        early_exit1 = Dense(2, activation="sigmoid", name="early_exit1")(Flatten()(seq1))

        seq1 = Conv2D(filters=config["nb_filter"], kernel_size=(1, 3),
                      kernel_regularizer=L2(config["l2"]),
                      kernel_initializer="lecun_uniform",
                      activation="relu", name="time_conv_layer2")(seq1)
        seq1 = Dropout(config["dropout"], name="time_dropout_layer2")(seq1)
        early_exit2 = Dense(2, activation="sigmoid", name="early_exit2")(Flatten()(seq1))

        seq1 = Flatten(name="time_flatten_layer")(seq1)
        output1 = Dense(config["nn_time_output"], activation="tanh", name="cnn_time_output")(seq1)

        # Frequency domain input branch
        input_freq = Input(shape=(channels * fft_bins, steps, 1), name="freq_domain_input")
        seq2 = Conv2D(filters=config["nb_filter"], kernel_size=(channels * fft_bins, 1),
                      kernel_regularizer=L2(config["l2"]),
                      kernel_initializer="lecun_uniform",
                      activation="relu", name="freq_conv_layer1")(input_freq)
        seq2 = channel_attention(seq2)
        seq2 = Dropout(config["dropout"], name="freq_dropout_layer1")(seq2)

        early_exit3 = Dense(2, activation="sigmoid", name="early_exit3")(Flatten()(seq2))

        seq2 = Conv2D(filters=config["nb_filter"], kernel_size=(1, 3),
                      kernel_regularizer=L2(config["l2"]),
                      kernel_initializer="lecun_uniform",
                      activation="relu", name="freq_conv_layer2")(seq2)
        seq2 = Dropout(config["dropout"], name="freq_dropout_layer2")(seq2)
        early_exit4 = Dense(2, activation="sigmoid", name="early_exit4")(Flatten()(seq2))

        seq2 = Flatten(name="freq_flatten_layer")(seq2)
        output2 = Dense(config["nn_freq_output"], activation="tanh", name="cnn_freq_output")(seq2)

        # Merge and apply attention
        # After merging the two branches
        merged = concatenate([output1, output2])
        merged = Dense(512, activation="tanh", name="fcn_layer1")(merged)
        merged = Dense(256, activation="tanh", name="fcn_layer2")(merged)
        merged = Dense(128, activation="tanh", name="fcn_layer3")(merged)

        # Reshape merged to 4D before applying channel_attention
        merged = Reshape((1, 1, 128), name="reshape_for_attention")(merged)
        merged = channel_attention(merged)

        # Final output
        output = Dense(2, activation="softmax", name="final_output")(Flatten()(merged))

        # Model definition
        cnn_model = Model(inputs=[input_time, input_freq], outputs=[output, early_exit1, early_exit2, early_exit3, early_exit4])

        cnn_model.compile(
            loss="binary_crossentropy",
            optimizer=SGD(learning_rate=config["learning_rate"]),
            metrics=["accuracy"]
        )

        early_stopping = EarlyStopping(
            monitor="val_loss",
            patience=5,
            restore_best_weights=True,
            verbose=1,
        )

        return cnn_model, early_stopping
