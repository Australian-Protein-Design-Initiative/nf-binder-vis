#!/usr/bin/env python3
# /// script
# requires-python = ">=3.12"
# dependencies = [
#     "pandas",
# ]
# ///

import argparse
import os
import sys
import shutil
import pandas as pd
import logging
from io import StringIO
import hashlib
import base64

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s', stream=sys.stderr)

def seguid(seq: str) -> str:
    """
    Return the SEGUID v1 (SHA-1 hash Base64 encoded) of the amino acid sequence.
    See: https://www.seguid.org/
    """
    m = hashlib.sha1()
    seq = seq.upper()
    m.update(seq.encode('utf-8'))
    return base64.b64encode(m.digest()).rstrip(b"=").decode('ascii')

def write_fasta_output(df, output_path):
    """Write sequences from the dataframe to a FASTA file."""
    # Find the first sequence column
    sequence_column = None
    for col in df.columns:
        if col in ['Sequence', 'sequence']:
            sequence_column = col
            break
    
    if not sequence_column:
        logging.warning("No sequence columns found in the data. Looking for columns named 'Sequence' or 'sequence'.")
        return 0
    
    sequences_written = 0
    
    try:
        if output_path == '-':
            output_file = sys.stdout
        else:
            output_file = open(output_path, 'w')
        
        for idx, row in df.iterrows():
            sequence = str(row[sequence_column]).strip()

            # Create a unique identifier for each sequence
            design_id = row.get('Design', '')
            if design_id:
                base_id = str(design_id).strip()
            else:
                base_id = f"design_{seguid(sequence)}"
            
            if sequence and sequence.lower() not in ['nan', 'none', '']:
                header = f">{base_id}"
                
                output_file.write(f"{header}\n")
                # Write sequence with line wrapping at 80 characters
                for i in range(0, len(sequence), 80):
                    output_file.write(f"{sequence[i:i+80]}\n")
                sequences_written += 1
        
        if output_path != '-':
            output_file.close()
            logging.info(f"FASTA output written to {output_path}")
        
    except Exception as e:
        logging.error(f"Error writing FASTA output to {output_path}: {e}")
        if output_path != '-' and 'output_file' in locals():
            output_file.close()
        return 0
    
    return sequences_written

def main():
    parser = argparse.ArgumentParser(description="Collect PDB files listed in a TSV file, with optional filtering.")
    parser.add_argument("tsv_file", help="Path to the input TSV file.")
    parser.add_argument("--output-dir", default="good_designs", help="Directory to copy PDB files to (default: good_designs).")
    parser.add_argument("--all", action="store_true", help="Process all designs, ignoring the 'good' column.")
    parser.add_argument("--top-n", type=int, help="Process only the top N designs, sorted by the score column (see --score-column).")
    parser.add_argument("--score-column", default="Average_i_pTM", help="The column to use for sorting when --top-n is used (default: Average_i_pTM).")
    parser.add_argument("--output-tsv", default="-", help="Path to output the selected rows of the input TSV. Use '-' for stdout (default). To disable, use an empty string.")
    parser.add_argument("--fasta-output", help="Path to output sequences in FASTA format. Use '-' for stdout. Automatically detects sequence columns.")
    
    args = parser.parse_args()

    # Create output directory if it doesn't exist
    if not os.path.exists(args.output_dir):
        try:
            os.makedirs(args.output_dir)
            logging.info(f"Created output directory: {args.output_dir}")
        except OSError as e:
            logging.error(f"Error creating output directory {args.output_dir}: {e}")
            sys.exit(1)

    # Read the TSV file
    try:
        df = pd.read_csv(args.tsv_file, sep='\t', keep_default_na=False, na_filter=False)
        # Log initial NaN count in pdb_file
        if 'pdb_file' in df.columns:
            # With keep_default_na=False and na_filter=False, actual NaN values are less likely from read_csv
            # unless they are explicitly 'NaN' strings or similar that pandas still parses as NaN even with these flags.
            # Empty strings will not be NaN.
            initial_pdb_nan_count = df['pdb_file'].isna().sum()
            empty_string_count = (df['pdb_file'] == '').sum()
            if initial_pdb_nan_count > 0:
                logging.info(f"INITIAL DIAGNOSTIC: Found {initial_pdb_nan_count} actual NaN values in 'pdb_file' column after reading TSV ('{args.tsv_file}') with keep_default_na=False.")
            if empty_string_count > 0:
                logging.info(f"INITIAL DIAGNOSTIC: Found {empty_string_count} empty strings in 'pdb_file' column after reading TSV ('{args.tsv_file}') with na_filter=False.")
            if initial_pdb_nan_count == 0 and empty_string_count == 0:
                logging.info(f"INITIAL DIAGNOSTIC: No NaN values or empty strings found in 'pdb_file' column immediately after reading the TSV '{args.tsv_file}'.")
        # Note: The main check for 'pdb_file' column existence is still further down,
        # this is just for early diagnostic on NaNs if the column does exist at this point.
    except FileNotFoundError:
        logging.error(f"Input TSV file not found: {args.tsv_file}")
        sys.exit(1)
    except pd.errors.EmptyDataError:
        logging.error(f"Input TSV file is empty: {args.tsv_file}")
        sys.exit(1)
    except Exception as e:
        logging.error(f"Error reading TSV file {args.tsv_file}: {e}")
        sys.exit(1)

    if 'pdb_file' not in df.columns:
        logging.error(f"Column 'pdb_file' not found in {args.tsv_file}")
        sys.exit(1)

    # Apply filtering
    if not args.all:
        if 'good' not in df.columns:
            logging.error(f"Column 'good' not found in {args.tsv_file}. Use --all to ignore this column.")
            sys.exit(1)
        # Ensure 'good' column is boolean or can be safely converted
        try:
            # Attempt to convert to boolean if it's not already
            if df['good'].dtype != bool:
                # Handle common string representations of boolean
                if df['good'].isin([True, False, 'True', 'False', 'true', 'false', 1, 0]).all():
                     df['good'] = df['good'].replace({'True': True, 'False': False, 'true': True, 'false': False, 1: True, 0: False}).astype(bool)
                else:
                    logging.warning(f"Column 'good' contains values that cannot be reliably converted to boolean. Trying direct conversion.")
                    df['good'] = df['good'].astype(bool)
            df = df[df['good'] == True]
            logging.info("Filtered designs by 'good == True'.")
        except Exception as e:
            logging.error(f"Error processing 'good' column for filtering: {e}. Ensure it contains boolean-like values or use --all.")
            sys.exit(1)


    if args.top_n is not None:
        if args.top_n <= 0:
            logging.error("--top-n must be a positive integer.")
            sys.exit(1)
        if args.score_column not in df.columns:
            logging.error(f"Column '{args.score_column}' not found in {args.tsv_file}, which is required for --top-n.")
            sys.exit(1)
        try:
            # Ensure the score column is numeric
            df[args.score_column] = pd.to_numeric(df[args.score_column], errors='coerce')
            df = df.dropna(subset=[args.score_column]) # Remove rows where conversion failed
            df = df.sort_values(by=args.score_column, ascending=False).head(args.top_n)
            logging.info(f"Selected top {args.top_n} designs based on '{args.score_column}'.")
        except Exception as e:
            logging.error(f"Error processing '{args.score_column}' for sorting: {e}")
            sys.exit(1)
            
    # Output the filtered TSV
    if args.output_tsv:
        try:
            if args.output_tsv == '-':
                # Writing to stdout
                df.to_csv(sys.stdout, sep='	', index=False)
            else:
                # Writing to a file
                df.to_csv(args.output_tsv, sep='	', index=False)
                logging.info(f"Filtered data written to {args.output_tsv}")
        except Exception as e:
            logging.error(f"Error writing to {args.output_tsv}: {e}")
            sys.exit(1)

    # Output FASTA if requested
    if args.fasta_output:
        sequences_written = write_fasta_output(df, args.fasta_output)
        if sequences_written > 0:
            logging.info(f"Wrote {sequences_written} sequences to FASTA output.")
        else:
            logging.warning("No sequences were written to FASTA output.")

    if df.empty:
        logging.info("No designs to process after filtering.")
        sys.exit(0)

    copied_count = 0
    not_found_count = 0

    for pdb_path in df['pdb_file']:
        if not pdb_path:
            logging.warning("Found a missing value in 'pdb_file' column after filtering, skipping.")
            continue
        
        pdb_path = str(pdb_path).strip() # Ensure it's a string and remove whitespace
        if not pdb_path:
            logging.warning("Found an empty path in 'pdb_file' column after filtering, skipping.")
            continue

        if os.path.isfile(pdb_path):
            pdb_filename = os.path.basename(pdb_path)
            destination_path = os.path.join(args.output_dir, pdb_filename)
            try:
                shutil.copy(pdb_path, destination_path)
                logging.info(f"Copied {pdb_path} to {destination_path}")
                copied_count += 1
            except shutil.Error as e:
                logging.error(f"Error copying {pdb_path} to {destination_path}: {e}")
            except Exception as e:
                logging.error(f"An unexpected error occurred while copying {pdb_path}: {e}")
        else:
            logging.warning(f"PDB file not found: {pdb_path}")
            not_found_count += 1
    
    logging.info(f"Finished processing. Copied {copied_count} files. {not_found_count} files not found (after filtering).")

if __name__ == "__main__":
    main()