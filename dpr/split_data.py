import csv
from sklearn.model_selection import train_test_split


def split_dataset(
    input_file,
    train_file,
    test_file,
    valid_file,
    test_size=0.2,
    valid_size=0.1,
    random_seed=42,
):
    with open(input_file, "r", newline="", encoding="utf-8-sig") as csvfile:
        reader = csv.reader(csvfile)
        headers = next(reader)

        # Remove quotes from header names
        headers = [col.strip('"') for col in headers]

        # Read the remaining rows
        rows = list(reader)

        # Separate features and labels
        features = [
            {
                "id": str(idx),
                "question": row[headers.index("question")],
                "answer": row[headers.index("answer1")],
            }
            for idx, row in enumerate(rows)
        ]

        # Extract labels or any other information needed for your model training

        # Split the data into train, test, and validation sets
        train_data, test_valid_data = train_test_split(
            features, test_size=(test_size + valid_size), random_state=random_seed
        )
        test_data, valid_data = train_test_split(
            test_valid_data,
            test_size=valid_size / (test_size + valid_size),
            random_state=random_seed,
        )

    # Write the datasets to separate CSV files
    write_to_csv(train_data, headers, train_file)
    write_to_csv(test_data, headers, test_file)
    write_to_csv(valid_data, headers, valid_file)


def write_to_csv(data, headers, output_file):
    with open(output_file, "w", newline="", encoding="utf-8") as csvfile:
        fieldnames = ["id", "question", "answer"]
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(data)


# Example usage
input_csv_file = "ts_hust_crawl_final_tokenized.csv"
train_csv_file = "train.csv"
test_csv_file = "test.csv"
valid_csv_file = "valid.csv"

split_dataset(input_csv_file, train_csv_file, test_csv_file, valid_csv_file)
