#!/usr/bin/env python
# /// script
# requires-python = ">=3.12"
# dependencies = [
#     "pandas",
# ]
# ///

import argparse
import logging
import sys
import io
from typing import Set

import pandas as pd

## TODO:
# - Add a molecular weight column (in kDa)
# - Theoretical isoelectric point pI might be interesting also ?

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s', stream=sys.stderr)

def main():
    """Main function to process sequences."""
    parser = argparse.ArgumentParser(
        description="Add N- or C-terminal tags to sequences based on a tag column."
    )
    parser.add_argument(
        "input_file",
        help="Input TSV or CSV file with 'Design' and 'Sequence' columns.",
    )
    parser.add_argument(
        "-o",
        "--output",
        help="Output TSV file path. Defaults to stdout.",
        default="-",
    )
    parser.add_argument(
        "--tag-table",
        help="Optional TSV/CSV file with 'Design' and 'tag' columns. Overrides tag column in input file.",
    )
    parser.add_argument(
        "--n-seq",
        default="HHHHHHSG",
        help="Sequence to add to the N-terminus. Default: HHHHHHSG",
    )
    parser.add_argument(
        "--c-seq",
        default="GSHHHHHH",
        help="Sequence to add to the C-terminus. Default: GSHHHHHH",
    )
    parser.add_argument(
        "--add-m",
        action="store_true",
        help="Add 'M' (start) to the beginning of the sequence. Default: False",
    )
    parser.add_argument(
        "--must-contain-aas",
        default="Y,W",
        help="Comma-separated list of amino acids that must be present in the sequence. Default: Y,W",
    )
    parser.add_argument(
        "--must-not-contain-aas",
        default="C",
        help="Comma-separated list of amino acids that must not be present in the sequence. Default: C",
    )
    parser.add_argument(
        "--no-stops",
        action="store_true",
        help="Don't add stop (**) codons to the end of the sequence. Default: False",
    )
    parser.add_argument(
        "--new-design-prefix",
        default="",
        help="Change the name of design prefix - if set, everything before the first underscore will be replaced with this string. Default: ''",
    )

    parser.add_argument(
        "--fasta-output",
        help="Path for FASTA output file. Header will be >{Design}.",
    )

    args = parser.parse_args()

    # Read input file
    try:
        # Determine separator
        sep = '\t' if args.input_file.endswith('.tsv') else ','
        df = pd.read_csv(args.input_file, sep=sep)
    except FileNotFoundError:
        logging.error(f"Input file not found: {args.input_file}")
        sys.exit(1)

    # Check for required columns
    required_cols = {"Design", "Sequence"}
    if not required_cols.issubset(df.columns):
        logging.error(f"Input file must contain 'Design' and 'Sequence' columns. Found: {list(df.columns)}")
        sys.exit(1)

    # Handle tag column
    if args.tag_table:
        try:
            tag_sep = '\t' if args.tag_table.endswith('.tsv') else ','
            tag_df = pd.read_csv(args.tag_table, sep=tag_sep)
            if "tag" not in tag_df.columns or "Design" not in tag_df.columns:
                 logging.error(f"--tag-table must contain 'Design' and 'tag' columns. Found: {list(tag_df.columns)}")
                 sys.exit(1)
            
            # keep only Design and tag from tag_df to avoid other column conflicts
            tag_df = tag_df[['Design', 'tag']]

            # If the input file has a 'tag' column, it will be overridden by the tag-table.
            if 'tag' in df.columns:
                df = df.drop(columns=['tag'])

            df = pd.merge(df, tag_df, on="Design", how="left")
        except FileNotFoundError:
            logging.error(f"Tag table file not found: {args.tag_table}")
            sys.exit(1)

    if "tag" not in df.columns:
        logging.error("No 'tag' column found in the input file or provided tag table.")
        sys.exit(1)
    
    if df['tag'].isnull().any():
        designs_no_tag = df[df['tag'].isnull()]['Design'].tolist()
        logging.error(f"{len(designs_no_tag)} designs have no tag and will not be modified: {', '.join(designs_no_tag)}")
        sys.exit(1)

    # Validate sequences
    must_contain_aas: Set[str] = set(args.must_contain_aas.split(',')) if args.must_contain_aas else set()
    must_not_contain_aas: Set[str] = set(args.must_not_contain_aas.split(',')) if args.must_not_contain_aas else set()

    def check_must_contain(sequence: str) -> bool:
        return any(aa in sequence for aa in must_contain_aas)

    def check_must_not_contain(sequence: str) -> bool:
        return any(aa in sequence for aa in must_not_contain_aas)

    df['has_YW'] = df['Sequence'].apply(check_must_contain)
    df['has_C'] = df['Sequence'].apply(check_must_not_contain)

    for _, row in df.iterrows():
        sequence = row["Sequence"]
        design = row["Design"]
        
        # Must contain check
        if must_contain_aas and not row['has_YW']:
            logging.warning(
                f"Design '{design}': Sequence does not contain any of the required amino acids ({', '.join(must_contain_aas)})."
            )

        # Must not contain check
        if must_not_contain_aas and row['has_C']:
            logging.warning(
                f"Design '{design}': Sequence contains a forbidden amino acid ({', '.join(sorted(list(must_not_contain_aas.intersection(set(sequence)))))})."
            )

    # Modify sequences
    def modify_sequence(row):
        tag = row["tag"]
        sequence = row["Sequence"]
        # remove any stop codons, we will add them back later
        sequence = sequence.strip("*")
        new_seq = None
        if tag == "N":
            new_seq = f"{args.n_seq}{sequence}"
        elif tag == "C":
            new_seq = f"{sequence}{args.c_seq}"
        else:
            new_seq = sequence
        if args.add_m and not new_seq.startswith('M'):
            new_seq = f"M{new_seq}"
        # add double stop codon
        if not args.no_stops:
            new_seq = f"{new_seq}**"
        return new_seq

    df["Sequence"] = df.apply(modify_sequence, axis=1)

    # Prepare output
    output_df = df[["Design", "Sequence", "tag", "has_YW", "has_C"]]

    if args.new_design_prefix:
        output_df["Design"] = output_df["Design"].str.replace(
            r"^[^_]+_", 
            args.new_design_prefix + "_", 
            regex=True)

    output_df = output_df.sort_values(by="Design")

    # Write FASTA output if requested
    if args.fasta_output:
        with open(args.fasta_output, "w") as f:
            for _, row in output_df.iterrows():
                design = row["Design"]
                sequence = row["Sequence"].strip("*")
                f.write(f">{design}\n{sequence}\n")
        logging.info(f"Wrote {len(output_df)} sequences to FASTA file: {args.fasta_output}")

    # Write output
    output_buffer = io.StringIO()
    output_df.to_csv(output_buffer, sep="\t", index=False)
    output_content = output_buffer.getvalue()

    if args.output == "-":
        sys.stdout.write(output_content)
    else:
        with open(args.output, "w") as f:
            f.write(output_content)
    logging.info(f"Processed {len(df)} sequences.")


if __name__ == "__main__":
    main()
