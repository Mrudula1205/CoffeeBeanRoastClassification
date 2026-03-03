import tensorflow as tf
from tensorflow.keras import applications, layers, models, Input
from .utils import read_params


class CoffeeModelEngine:
    def __init__(self):
        self.config = read_params()
        self.model_cfg = self.config['model']
        self.data_cfg = self.config['data']
        self.model = None

    def build_inception_model(self):
        """
        Recreates the architecture from your notebook (Cell 15/16).
        """
        img_shape = (*self.data_cfg['image_size'], 3)

        # 1. Load Base InceptionV3
        base_model = applications.InceptionV3(
            weights=self.model_cfg['weights'],
            include_top=False,
            input_shape=img_shape
        )
        base_model.trainable = False  # Freeze base layers as per your notebook

        # 2. Add Custom Head
        inputs = Input(shape=img_shape)
        x = base_model(inputs, training=False)
        x = layers.GlobalAveragePooling2D()(x)
        x = layers.Dense(self.model_cfg['dense_units'], activation='relu')(x)
        x = layers.Dropout(self.model_cfg['dropout_rate'])(x)

        # 4 outputs for ['Dark', 'Green', 'Light', 'Medium']
        outputs = layers.Dense(len(self.data_cfg['class_names']), activation='sigmoid')(x)

        self.model = models.Model(inputs, outputs)

        # 3. Compile
        self.model.compile(
            optimizer=tf.keras.optimizers.Adam(learning_rate=self.model_cfg['learning_rate']),
            loss=tf.keras.losses.CategoricalCrossentropy(from_logits=True),
            metrics=[tf.keras.metrics.CategoricalAccuracy()]
        )
        return self.model

    def save_model(self, path="inception.hdf5"):
        if self.model:
            self.model.save(path)

    def load_existing_model(self, path="inception.hdf5"):
        # Rebuild architecture to avoid Keras version config deserialization issues,
        # then load only the saved weights from the .hdf5 file.
        self.build_inception_model()
        self.model.load_weights(path, by_name=True, skip_mismatch=True)
        return self.model
