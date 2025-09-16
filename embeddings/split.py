import os
import pandas as pd
import csv
from pathlib import Path

def split_csv_file(input_file, max_size_mb=45, output_dir="split_files"):
    """
    Split a large CSV file into smaller chunks for Bedrock Knowledge Base
    
    Args:
        input_file (str): Path to the input CSV file
        max_size_mb (int): Maximum size per chunk in MB (default 45MB, under 50MB limit)
        output_dir (str): Directory to save split files
    """
    
    max_size_bytes = max_size_mb * 1024 * 1024  # Convert MB to bytes
    
    # Create output directory if it doesn't exist
    Path(output_dir).mkdir(exist_ok=True)
    
    # Get base filename without extension
    base_name = Path(input_file).stem
    
    print(f"Splitting {input_file} into chunks of max {max_size_mb}MB...")
    print(f"Original file size: {os.path.getsize(input_file) / (1024*1024):.2f}MB")
    
    # Read CSV in chunks to handle large files efficiently
    chunk_num = 1
    current_rows = []
    header_row = None
    
    # First, get the header
    with open(input_file, 'r', encoding='utf-8') as infile:
        csv_reader = csv.reader(infile)
        header_row = next(csv_reader)
        header_size = len(','.join(header_row).encode('utf-8')) + 1
    
    # Process file row by row
    with open(input_file, 'r', encoding='utf-8') as infile:
        csv_reader = csv.reader(infile)
        next(csv_reader)  # Skip header (already read)
        
        current_size = header_size  # Start with header size
        
        for row_num, row in enumerate(csv_reader, 1):
            # Calculate size of this row
            row_text = ','.join(f'"{cell}"' if ',' in cell or '"' in cell else cell for cell in row)
            row_size = len(row_text.encode('utf-8')) + 1  # +1 for newline
            
            # Check if adding this row would exceed the size limit
            if current_size + row_size > max_size_bytes and current_rows:
                # Write current chunk
                write_csv_chunk(current_rows, output_dir, base_name, chunk_num, header_row, current_size)
                
                # Start new chunk
                chunk_num += 1
                current_rows = [row]
                current_size = header_size + row_size
            else:
                # Add row to current chunk
                current_rows.append(row)
                current_size += row_size
            
            # Progress indicator
            if row_num % 1000 == 0:
                print(f"Processed {row_num} rows...")
    
    # Write the final chunk if there are remaining rows
    if current_rows:
        write_csv_chunk(current_rows, output_dir, base_name, chunk_num, header_row, current_size)
    
    print(f"\nSplitting complete! Created {chunk_num} files in '{output_dir}' directory")
    
    # Verify all chunks are under the size limit
    verify_chunks(output_dir, max_size_bytes)

def write_csv_chunk(rows, output_dir, base_name, chunk_num, header_row, size):
    """Write a chunk of rows to a new CSV file"""
    
    output_file = f"{output_dir}/{base_name}_part_{chunk_num:03d}.csv"
    
    with open(output_file, 'w', encoding='utf-8', newline='') as outfile:
        csv_writer = csv.writer(outfile)
        
        # Write header first
        csv_writer.writerow(header_row)
        
        # Write all rows
        for row in rows:
            csv_writer.writerow(row)
    
    size_mb = size / (1024 * 1024)
    print(f"Created {output_file} - {len(rows)} records, {size_mb:.2f}MB")

def split_csv_by_rows(input_file, rows_per_chunk=1000, output_dir="split_files"):
    """
    Split CSV by number of rows (simpler approach for consistent-sized records)
    """
    Path(output_dir).mkdir(exist_ok=True)
    base_name = Path(input_file).stem
    
    print(f"Splitting {input_file} by {rows_per_chunk} rows per chunk...")
    
    # Read and split using pandas (more efficient for large CSVs)
    chunk_num = 1
    
    # Read CSV in chunks
    for chunk_df in pd.read_csv(input_file, chunksize=rows_per_chunk):
        output_file = f"{output_dir}/{base_name}_part_{chunk_num:03d}.csv"
        
        # Write chunk to CSV
        chunk_df.to_csv(output_file, index=False)
        
        # Get file size
        size_mb = os.path.getsize(output_file) / (1024 * 1024)
        print(f"Created {output_file} - {len(chunk_df)} records, {size_mb:.2f}MB")
        
        chunk_num += 1
    
    print(f"\nCreated {chunk_num-1} files by row count")
    
    # Verify sizes
    verify_chunks(output_dir, 50 * 1024 * 1024)  # 50MB limit

def verify_chunks(output_dir, max_size_bytes):
    """Verify that all chunks are under the size limit"""
    
    print("\nVerifying chunk sizes...")
    
    csv_files = [f for f in os.listdir(output_dir) if f.endswith('.csv')]
    csv_files.sort()
    
    total_size = 0
    all_valid = True
    
    for file in csv_files:
        file_path = os.path.join(output_dir, file)
        file_size = os.path.getsize(file_path)
        file_size_mb = file_size / (1024 * 1024)
        total_size += file_size
        
        status = "✓" if file_size < max_size_bytes else "✗ TOO LARGE"
        print(f"{file}: {file_size_mb:.2f}MB {status}")
        
        if file_size >= max_size_bytes:
            all_valid = False
    
    print(f"\nTotal size of all chunks: {total_size / (1024*1024):.2f}MB")
    
    if all_valid:
        print("✅ All chunks are under the size limit!")
    else:
        print("❌ Some chunks are too large. Consider reducing max_size_mb parameter.")

def analyze_csv_structure(input_file, sample_size=5):
    """
    Analyze the structure of your CSV file to understand the data
    """
    print(f"Analyzing structure of {input_file}...")
    
    # Read first few rows to understand structure
    df_sample = pd.read_csv(input_file, nrows=sample_size)
    
    print(f"\nCSV Structure:")
    print(f"Columns ({len(df_sample.columns)}): {list(df_sample.columns)}")
    print(f"Sample data shape: {df_sample.shape}")
    
    # Show field sizes for each column
    print(f"\nColumn analysis:")
    for col in df_sample.columns:
        if df_sample[col].dtype == 'object':  # String columns
            avg_length = df_sample[col].astype(str).str.len().mean()
            max_length = df_sample[col].astype(str).str.len().max()
            print(f"  {col}: avg {avg_length:.0f} chars, max {max_length} chars")
        else:
            print(f"  {col}: {df_sample[col].dtype}")
    
    # Estimate average row size
    sample_csv = df_sample.to_csv(index=False)
    avg_row_size = len(sample_csv.encode('utf-8')) / len(df_sample)
    print(f"\nEstimated average row size: {avg_row_size:.0f} bytes")
    
    # Calculate total file info
    total_rows = sum(1 for _ in open(input_file, 'r')) - 1  # -1 for header
    estimated_rows_per_45mb = (45 * 1024 * 1024) // avg_row_size
    estimated_chunks = max(1, total_rows // estimated_rows_per_45mb)
    
    print(f"Total rows in file: {total_rows:,}")
    print(f"Estimated rows per 45MB chunk: {estimated_rows_per_45mb:,}")
    print(f"Estimated number of chunks needed: {estimated_chunks}")

# Usage examples
if __name__ == "__main__":
    # Replace with your actual CSV file path
    input_csv = "/Users/shubhan/MacroRecipe/MacroRecipeLLM/bedrock_kb_recipes.csv"
    
    # First, analyze the structure (optional)
    if os.path.exists(input_csv):
        analyze_csv_structure(input_csv, sample_size=3)
        print("\n" + "="*50 + "\n")
        
        # Choose your splitting method:
        
        # Method 1: Split by file size (recommended)
        split_csv_file(
            input_file=input_csv,
            max_size_mb=20,  # 45MB per chunk (safely under 50MB limit)
            output_dir="recipe_csv_chunks"
        )
        
        # Method 2: Split by row count (alternative - uncomment to use)
        # split_csv_by_rows(
        #     input_file=input_csv,
        #     rows_per_chunk=1000,  # 1000 recipes per file
        #     output_dir="recipe_row_chunks"
        # )
        
    else:
        print(f"File {input_csv} not found!")
        print("\nTo use this script:")
        print("1. Update 'input_csv' variable with your CSV file path")
        print("2. Run the script")
        print("3. Upload all CSV files from output directory to your S3 bucket")