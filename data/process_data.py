import pickle
import json
from collections import defaultdict

import pandas as pd
from sklearn.model_selection import train_test_split


def rating_to_label(r):
    if r <= 2:
        return 0
    else:
        return 1


def main():

    user_times = defaultdict(list)
    with open('amazon_fashion_dataset_review.json', 'r') as f:
        data = json.load(f)

    df = pd.DataFrame.from_dict(data)

    df = df[df.text.apply(lambda x: len(x.strip().split())) >= 10]
    df['text'] = df.text.apply(lambda x: x.lower())
    df['label'] = df['rating'].apply(rating_to_label)
    df.drop_duplicates(subset=['text'], inplace=True)
    df.dropna(inplace=True)

    df.reset_index(inplace=True, drop=True)

    data = df[['product', 'time', 'year', 'month', 'day', 'text', 'rating', 'label']]

    train_dev, test = train_test_split(data, test_size=0.2, random_state=123, stratify=data[['rating']])
    train, dev = train_test_split(train_dev, test_size=0.125, random_state=123, stratify=train_dev[['rating']])

    train.to_csv('amazon_fashion_train.csv', index=False)
    dev.to_csv('amazon_fashion_dev.csv', index=False)
    test.to_csv('amazon_fashion_test.csv', index=False)

    edge_set = set()
    users = set(data.user)

