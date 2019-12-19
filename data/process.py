import glob
import os
import re
from typing import Tuple

import pandas as pd
from langua import Predict
from langua.lang_detect_exception import LangDetectException
from pandarallel import pandarallel

pandarallel.initialize(verbose=0)

LANGUAGE_DETECTOR = Predict()


def _is_valid(row: pd.Series) -> bool:
    try:
        language = LANGUAGE_DETECTOR.get_lang(row.review)
    except LangDetectException:
        return False

    if language != "en":
        return False

    if not isinstance(row.review, str):
        return False

    if not row.review.strip():
        return False

    return True


def _clean_row(row: pd.Series) -> Tuple[int, str, str]:
    label = row.label - 1
    title = _clean_text(row.title)
    review = _clean_text(row.review)
    return label, title, review


def _clean_text(text: str) -> str:
    lowercase = text.lower()
    clean_text = re.sub("[^a-zA-Z\\s]", "", lowercase)
    return clean_text


def _process(filename: str) -> pd.DataFrame:
    df = pd.read_csv(filename, header=None, names=["label", "title", "review"])
    non_spam_indexes = df.parallel_apply(_is_valid, axis=1)
    non_spam_df = df[non_spam_indexes]
    non_spam_df = non_spam_df.parallel_apply(_clean_row, result_type="broadcast", axis=1)
    return non_spam_df


if __name__ == "__main__":
    data_dir = os.path.dirname(os.path.abspath(__file__))
    raw_data_paths = glob.glob(os.path.join(data_dir, "raw", "*.csv"))

    for file_path in raw_data_paths:
        basename = os.path.basename(file_path)
        output_path = os.path.join(data_dir, "processed", basename)

        print(f"\nProcessing '{basename}'")

        processed_df = _process(file_path)
        processed_df.to_csv(output_path, index=False)

        print(f"Saving results to '{output_path}'")
        print("Done\n")
