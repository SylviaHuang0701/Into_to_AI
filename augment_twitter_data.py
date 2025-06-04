import csv
import os

def process_twitter_file(input_path, output_writer):
    """Process a single twitter data file and write to output"""
    with open(input_path, 'r', encoding='utf-8') as f:
        for line in f:
            parts = line.strip().split('\t')
            if len(parts) != 3:
                continue
            
            tweet_id, text, label = parts
            if label not in ['true', 'false']:
                continue  # Skip unverified/non-rumor
            
            # Convert label to 1/0
            new_label = '1' if label == 'true' else '0'
            output_writer.writerow([tweet_id, text, new_label, '0'])

def main():
    # Output file
    output_path = 'data/train_new.csv'
    
    # Get existing data if file exists
    existing_data = []
    if os.path.exists(output_path):
        with open(output_path, 'r', encoding='utf-8') as f:
            reader = csv.reader(f)
            header = next(reader)  # Skip header
            existing_data = list(reader)
    
    # Process all twitter files
    twitter_dirs = ['data/twitter15', 'data/twitter16']
    file_types = ['.train', '.test', '.dev']
    
    with open(output_path, 'w', newline='', encoding='utf-8') as f:
        writer = csv.writer(f)
        writer.writerow(['id', 'text', 'label', 'event'])  # Write header
        
        # Write existing data first
        for row in existing_data:
            writer.writerow(row)
            
        # Process new twitter files
        for dir_path in twitter_dirs:
            for file_type in file_types:
                file_path = os.path.join(dir_path, f"twitter{dir_path[-2:]}{file_type}")
                if os.path.exists(file_path):
                    process_twitter_file(file_path, writer)
                    print(f"Processed: {file_path}")
                else:
                    print(f"File not found: {file_path}")

if __name__ == '__main__':
    main()
