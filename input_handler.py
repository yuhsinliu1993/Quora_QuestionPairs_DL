import numpy as np
import csv
from utils import to_categorical


def get_input_from_csv(file_path):
    question_1 = []
    question_2 = []
    labels = []

    with open(file_path, 'r') as f:
        reader = csv.DictReader(f, fieldnames=['id', 'qid1', 'qid2', 'question1', 'question2', 'is_duplicate'])
        for i, row in enumerate(reader):
            if i > 0:
                # v_1 = get_vectorized_text(get_cleaned_text(row['question1']))
                # v_2 = get_vectorized_text(get_cleaned_text(row['question2']))
                # question_pairs[i] = {'question1': v_1, 'question2': v_2}
                # labels[i] = {'y_true': int(row['is_duplicate'])}
                question_1.append(row['question1'].decode('utf-8', 'ignore'))
                question_2.append(row['question2'].decode('utf-8', 'ignore'))
                labels.append(row['is_duplicate'])

    return question_1, question_2, labels
