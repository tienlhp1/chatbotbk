import csv
from preprocess import tokenise, preprocess_question
from pyvi.ViTokenizer import tokenize


def filter_and_format_csv(input_file, output_file):
    filtered_rows = []
    i = 0
    with open(input_file, "r", newline="", encoding="utf-8-sig") as csvfile:
        reader = csv.reader(csvfile)
        headers = next(reader)

        # Remove quotes from header names
        headers = [col.strip('"') for col in headers]

        # Check if required columns exist
        required_columns = ["question", "answer1"]
        for col in required_columns:
            if col not in headers:
                raise ValueError(f"Column '{col}' not found in the CSV file.")

        # Read the remaining rows
        rows = list(reader)
        # Filter rows with content in "question," "answer1," or "answer2" columns
        for idx, row in enumerate(rows):
            if row[headers.index("question")] and row[headers.index("answer1")]:
                filtered_rows.append(
                    {
                        "id": str(i),
                        "question": tokenise(
                            preprocess_question(
                                row[headers.index("question")], remove_end_phrase=False
                            ),
                            tokenize,
                        ),
                        "answer1": tokenise(
                            preprocess_question(
                                row[headers.index("answer1")], remove_end_phrase=False
                            ),
                            tokenize,
                        ),
                    }
                )
                i = i + 1

    # # Write the filtered and formatted rows to a new CSV file
    with open(output_file, "w", newline="", encoding="utf-8") as csvfile:
        fieldnames = ["id", "question", "answer1"]
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(filtered_rows)


# Example usage
input_csv_file = "ts_hust_crawl_final.csv"
output_csv_file = "ts_hust_crawl_final_tokenized.csv"
filter_and_format_csv(input_csv_file, output_csv_file)
