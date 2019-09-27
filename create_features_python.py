import math
import numpy as np
from collections import Counter

import pandas as pd
import dask.dataframe as dd
import dask.array as da


NUMBER_OF_TOP_POPULAR_PRODUCTS = 6000
MOVING_WINDOW_SIZE = 4
REMEMBER_WINDOW_SIZE = 4
MOVING_WINDOW_SKIP_SIZE = 1


def load_data():
    return pd.read_csv(
        "./data/input.csv", encoding="utf-8", dtype=str, na_values="null"
    )


def extract_raw_sessions(raw_data):
    raw_sessions = []
    for x in raw_data["product_sequence"].iteritems():
        raw_sessions.append(x[1].split(","))
    return raw_sessions


def get_popular_products(raw_sessions):
    products_all_occurrences = [x.strip() for y in raw_sessions for x in y if x.strip()]
    products_ordered_by_occurrences = sorted(
        Counter(products_all_occurrences).most_common(NUMBER_OF_TOP_POPULAR_PRODUCTS),
        key=lambda x: (x[1], x[0]),
        reverse=True,
    )

    popular_products = [x[0] for x in products_ordered_by_occurrences]
    return popular_products


def create_product_ids_masking(popular_products):
    return {
        product: index for index, product in enumerate(popular_products)
    }


def create_product_ids_masking_reversed(popular_products):
    return {
        index: product for index, product in enumerate(popular_products)
    }


def encode_sessions(product_ids_masking, raw_sessions):
    encoded_sessions = [
        [
            product_ids_masking.get(x)
            for x in item
            if product_ids_masking.get(x) is not None
        ]
        for item in raw_sessions
    ]
    return encoded_sessions


def get_valid_sessions(encoded_sessions):
    # Sequence of 1 doesn't have a target ;)
    return [x for x in encoded_sessions if len(x) > 1]


def extract_sequences_vs_targets(encoded_sessions):
    # Extract sequences and targets
    sequences_vs_targets = []

    # Only keep the x latest items that can fit in the REMEMBER_WINDOW_SIZE
    number_of_latest_items_to_fetch = MOVING_WINDOW_SIZE + (
            (REMEMBER_WINDOW_SIZE - 1) * MOVING_WINDOW_SKIP_SIZE
    )
    for session in encoded_sessions:
        targets = session[1:]
        intervals = []
        for index, target in enumerate(targets):
            base = session[: index + 1][-number_of_latest_items_to_fetch:]
            len_base_minus_window_size = len(base) - REMEMBER_WINDOW_SIZE
            number_of_active_remember_windows = (
                    math.ceil(len_base_minus_window_size / MOVING_WINDOW_SKIP_SIZE) + 1
            )
            if number_of_active_remember_windows < 1:
                number_of_active_remember_windows = 1
            sequences = [
                base[i: i + MOVING_WINDOW_SIZE]
                for i in range(number_of_active_remember_windows)
            ]
            sequences = ([[]] * (REMEMBER_WINDOW_SIZE - len(sequences))) + sequences
            intervals.append(sequences)
        sequences_vs_targets.append([intervals, targets])

    # print(sequences_vs_targets)
    return sequences_vs_targets


def encode_binary(sequences_vs_targets):
    # binary encoding
    OUTPUT_PROCESSING_CHUNK_SIZE = 10000
    number_of_sequences = len(sequences_vs_targets)
    chunks = [
        sequences_vs_targets[x: x + OUTPUT_PROCESSING_CHUNK_SIZE]
        for x in range(0, number_of_sequences, OUTPUT_PROCESSING_CHUNK_SIZE)
    ]

    for count, sequences_targets_sliced in enumerate(chunks):

        combined_X = []
        combined_y = []

        for X, y in sequences_targets_sliced:
            number_items = len(X)
            X_ = np.zeros(
                (number_items, MOVING_WINDOW_SIZE, NUMBER_OF_TOP_POPULAR_PRODUCTS),
                dtype=bool,
            )
            y_ = np.zeros((number_items, NUMBER_OF_TOP_POPULAR_PRODUCTS), dtype=bool)
            for index, (x_indexes, y_index) in enumerate(zip(X, y)):
                y_[index][y_index] = 1
                for interval_index in range(MOVING_WINDOW_SIZE):
                    X_[index][interval_index][x_indexes[interval_index]] = 1

            combined_X.append(np.copy(X_))
            combined_y.append(np.copy(y_))

            del X_
            del y_

        dataset_X = np.vstack(combined_X)
        dataset_y = np.vstack(combined_y)
    return dataset_X, dataset_y


def main():
    raw_data = load_data()
    print(raw_data.head())
    raw_sessions = extract_raw_sessions(raw_data)

    popular_products = get_popular_products(raw_sessions)
    popular_products_encoded = list(range(NUMBER_OF_TOP_POPULAR_PRODUCTS))

    product_ids_masking = create_product_ids_masking(popular_products)
    product_ids_masking_reversed = create_product_ids_masking_reversed(popular_products)

    # encoding
    encoded_sessions = encode_sessions(product_ids_masking, raw_sessions)
    encoded_sessions = get_valid_sessions(encoded_sessions)

    sequences_vs_targets = extract_sequences_vs_targets(encoded_sessions)

    X, y = encode_binary(sequences_vs_targets)
    # import pickle
    # with open('data.pickle', 'wb') as f:
    #     # Pickle the 'data' dictionary using the highest protocol available.
    #     pickle.dump((X, y), f, pickle.HIGHEST_PROTOCOL)

    print(f"length of X and y: {len(X)}, {len(y)}")
    print("end")
    return X, y

if __name__ == "__main__":
    main()
