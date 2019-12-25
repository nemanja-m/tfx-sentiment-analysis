from tfx.utils.logging_utils import get_logger, LoggerConfig
import tensorflow as tf
import tensorflow_transform as tft

logger = get_logger(LoggerConfig(pipeline_name="amazon_reviews_sentiment"))

VOCAB_SIZE = 20000


@tf.function
def log_tensor(tensor):
    logger.info(tensor.values)


@tf.function
def concat_titles_and_reviews(titles_sparse_tensor, reviews_sparse_tensor):
    titles_tensor = tft.sparse_tensor_to_dense_with_shape(
        titles_sparse_tensor, shape=titles_sparse_tensor.dense_shape, default_value=""
    )
    reviews_tensor = tft.sparse_tensor_to_dense_with_shape(
        reviews_sparse_tensor, shape=reviews_sparse_tensor.dense_shape, default_value=""
    )
    return titles_tensor + " " + reviews_tensor


def preprocessing_fn(inputs):
    """tf.transform's callback function for preprocessing inputs.

        Args:
        inputs: map from feature keys to raw not-yet-transformed features.

      Returns:
        Map from string feature key to transformed feature operations.
    """
    outputs = {}

    reviews = inputs["review"]

    vocab = tft.compute_and_apply_vocabulary(reviews, top_k=VOCAB_SIZE)

    outputs["vocab"] = vocab
    outputs["label"] = inputs["label"]

    return outputs
