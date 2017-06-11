import argparse
import csv
import os
import random


def main():
    argparser = argparse.ArgumentParser(
        description=("Split a file from the Kaggle Quora dataset "
                     "into train and validation files, given a validation "
                     "proportion"))
    argparser.add_argument("validation_proportion", type=float,
                           help=("Proportion of data in the input file "
                                 "to randomly split into a separate "
                                 "validation file."))
    argparser.add_argument("dataset_input_path", type=str,
                           help=("The path to the cleaned Quora "
                                 "dataset file to split."))
    argparser.add_argument("dataset_output_path", type=str,
                           help=("The *folder* to write the "
                                 "split files to. The name will just have "
                                 "_{split}_split appended to it, before "
                                 "the extension"))
    config = argparser.parse_args()

    # Get the data

    with open(config.dataset_input_path) as f:
        reader = csv.reader(f)
        csv_rows = list(reader)[1:]

    # For reproducibility
    random.seed(0)
    # Shuffle csv_rows deterministically in place
    random.shuffle(csv_rows)

    num_validation_lines = int(len(csv_rows) * config.validation_proportion)

    input_filename_full = os.path.basename(config.dataset_input_path)
    input_filename, input_ext = os.path.splitext(input_filename_full)
    train_out_path = os.path.join(config.dataset_output_path, "train_split" + input_ext)
    val_out_path = os.path.join(config.dataset_output_path, "val_split" + input_ext)

    with open(train_out_path, "w") as f:
        writer = csv.DictWriter(f, fieldnames=['id', 'qid1', 'qid2', 'question1', 'question2', 'is_duplicate'])
        writer.writeheader()
        rows = csv_rows[num_validation_lines:]

        for row in rows:
            writer.writerow({'id': row[0],
                             'qid1': row[1],
                             'qid2': row[2],
                             'question1': row[3],
                             'question2': row[4],
                             'is_duplicate': row[5]
                             })

    with open(val_out_path, "w") as f:
        # writer = csv.writer(f, quoting=csv.QUOTE_ALL)
        # writer.writerows(csv_rows[:num_validation_lines])
        writer = csv.DictWriter(f, fieldnames=['id', 'qid1', 'qid2', 'question1', 'question2', 'is_duplicate'])
        writer.writeheader()

        rows = csv_rows[:num_validation_lines]
        for row in rows:
            writer.writerow({'id': row[0],
                             'qid1': row[1],
                             'qid2': row[2],
                             'question1': row[3],
                             'question2': row[4],
                             'is_duplicate': row[5]
                             })


if __name__ == "__main__":
    main()
