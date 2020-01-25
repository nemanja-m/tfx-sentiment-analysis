import tensorflow as tf
import tensorflow_transform as tft
from tfx.utils.logging_utils import LoggerConfig
from tfx.utils.logging_utils import get_logger

logger = get_logger(LoggerConfig(pipeline_name="amazon_reviews_sentiment"))

_VOCAB_SIZE = 25000
_OOV_BUCKETS = 10


def preprocessing_fn(inputs):
    """tf.transform's callback function for preprocessing inputs.

    Raw text reviews are used to create vocabulary with top k words. Each word have an
    unique int value. Transformed labels have value 1 for positives and 0 for
    negatives.

        Args:
        inputs: map from feature keys to raw not-yet-transformed features.

      Returns:
        Map from string feature key to transformed feature operations.
    """
    outputs = {}

    reviews = inputs["review"]

    vocabulary = tft.compute_and_apply_vocabulary(
        reviews, top_k=_VOCAB_SIZE, num_oov_buckets=_OOV_BUCKETS
    )
    outputs["vocabulary"] = vocabulary

    labels = inputs["label"]
    dense_labels = tf.sparse.to_dense(
        tf.SparseTensor(labels.indices, labels.values, [labels.dense_shape[0], 1])
    )

    # In the raw dataset positive labels have value 2 and negative have 1.
    outputs["label"] = dense_labels - 1

    return outputs
