import argparse
import csv
import os

from utils import get_cleaned_text


def main():
    argparser = argparse.ArgumentParser(description=("Clean the Quora dataset "
                                                     "by removing newlines in "
                                                     "the data."))
    argparser.add_argument("dataset_input_path", type=str,
                           help=("The path to the raw Quora "
                                 "dataset to clean."))
    argparser.add_argument("dataset_output_path", type=str,
                           help=("The *folder* to write the "
                                 "cleaned file to. The name will just have "
                                 "_cleaned appended to it, before the "
                                 "extension"))
    config = argparser.parse_args()

    clean_rows = []
    with open(config.dataset_input_path, 'r') as f:
        reader = csv.DictReader(f, fieldnames=['id', 'qid1', 'qid2', 'question1', 'question2', 'is_duplicate'])
        for i, row in enumerate(reader):
            if i > 0:
                question1 = get_cleaned_text(row['question1'])
                question2 = get_cleaned_text(row['question2'])

                clean_rows.append([row['id'], row['qid1'], row['qid2'], question1, question2, row['is_duplicate']])

    input_filename_full = os.path.basename(config.dataset_input_path)
    input_filename, input_ext = os.path.splitext(input_filename_full)
    out_path = os.path.join(config.dataset_output_path,
                            input_filename + "_cleaned" + input_ext)

    with open(out_path, "w") as f:
        writer = csv.DictWriter(f, fieldnames=['id', 'qid1', 'qid2', 'question1', 'question2', 'is_duplicate'])
        writer.writeheader()

        for row in clean_rows:
            writer.writerow({'id': row[0],
                             'qid1': row[1],
                             'qid2': row[2],
                             'question1': row[3],
                             'question2': row[4],
                             'is_duplicate': row[5]
                             })


if __name__ == "__main__":
    main()
