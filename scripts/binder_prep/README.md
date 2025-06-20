# Preparing binders for ordering

This is a set of scripts to streamline preparing binder sequences to order for synthetic gene production (eg via Twist Bioscience).

> Currently these instructions and scripts are intended for BindCraft output, based on the column names used, however they will be adapted later to also accomodate RFDiffusion and generic binder design pipelines

## Step 0 - Install `uv`

[`uv`](https://docs.astral.sh/uv/) makes it easier to run Python scripts without thinking too much about environments and dependencies. We will use it here, which is why every command starts with `uv run`. You can create a virtualenv or something else if you prefer.

To install, run:
```bash
curl -LsSf https://astral.sh/uv/install.sh | sh
```

## Step 1 - Triage designs

Use the `nf-binder-vis/bindcraft.py` web app to 'triage' your binder designs, visualizing the best scoring designs and marking them as 'good' (thumbs up) or 'bad' (thumbs down). This generates `bindcraft_summary.tsv` in the folder with all your BindCraft runs.

## Step 2 - Collect best designs

Use `collect_designs.py` to copy the designs marked as 'good', and all the designs. The `bindcraft_summary.tsv` table should contain columns 'good' (boolean), 'pdb_file' (absolute path) and possibly 'Average_i_pTM' (float) (or other score indicated by `--score-column`)

```bash
uv run ./collect_designs.py --help
```

```bash
# Take the top 96 good designs
uv run ./collect_designs.py bindcraft_summary.tsv \
  --top-n 96 \
  --output-dir good_designs \
  --output-tsv good_designs/good_designs.tsv

# Collect all designs (flagged good and bad)
uv run ./collect_designs.py bindcraft_summary.tsv \
  --all \
  --output-dir all_designs \
  --output-tsv all_designs/all_designs.tsv
```

## Step 3 - Placing terminal tags

Now we need to decide which terminii to place the purification/detection tag on. We will assume a 6xHis-tag here, which is the default.

```bash
uv run ./tag_placement.py --help
```

This script is used to suggest N- or C-terminal His-tag placement for a designed protein binder based on the following criteria:

- The termini with the tag should be solvent accessible
- The termini with the tag should not be making contact with target residues
- (Optional) The preferred termini should be distant from the target to allow anti-His antibody binding

```bash
uv run ./tag_placement.py \
  --distant-from A258,A175,A165 \
  --output-pdb-path tagged_pdbs \
  --output tag_positions.tsv \
  ./good_designs/
```

The output file `tag_positions.tsv` contains `Design`, `pdb_file`, a number of columns with calculated values used to decide on the tag terminii, and a `tag` column which will be `N` or `C` (or None/empty is neither terminii appeared suitable).

The `--output-pdb-path` will create a directory with copies of the PDB file with a `TAG` residue added (heavy atom HG) that allows quick visualization of the tag placement. In ChimeraX you can highlight this using:

```
# Align 100 models to the first, chain A
matchmaker #2-96/A to #1/A

# Make the tag HG atom large and red
color /Z@HG red
size /Z@HG atomRadius 4
```

> Tip: If you right-click on the ChimeraX "Models" panel and "Show Sequential Display Controls" you get "Next" and "Previous" buttons to step through models.

You can verify the chosen tags positions look sensible - if not you may need to tune the thresholds to suit your target/binder set (see `--help` for options). If all else failsn, manually change the few that are incorrect in `tag_positions.tsv`.


## Step 4 - Preparing sequences for codon optimization

```bash
uv run ./prepare_sequences.py --help
```

We extract each of the sequences in the `good_designs.tsv` table (or any table with a `Design`,`Sequence` and optionally `tag` column) and modify sequence like:
 
 - add `HHHHHHSG` or `GSHHHHHH` tags to the N- or C-terminal end
 - add a double stop `**` to the end of every sequence
 - ***does not*** add any start (`M`) residue (use `--add-m` if you vector requires this)

This script also adds columns to the table indicating if the sequence contains `YW` or `C` residues. A `--fasta-output` can be optionally generated.

```bash
uv run ./prepare_sequences.py \
  --tag-table tag_postions.tsv  \
  --fasta-output tagged_sequences.fasta \
  good_designs/good_designs.tsv \
  >tagged_sequences.tsv
```

`tagged_sequences.tsv` is the amino acid sequence ready to be codon optimised and ordered.

 > If you feel you need to obfusicate the design name, use `--new-design-prefix mynewname` to replace the Design name before the first `_` - but ensure the new name can be uniquely traced back to the original design settings and PDB later !

## Step 5 - codon optimization

Use the [Twist Biosciences Codon Optimization tool](https://www.twistbioscience.com/resources/digital-tools/codon-optimization-tool) with your `tagged_sequences.tsv`.

> TODO: Twist requires inserts to be larger than a specific size (300 bp or larger ?). We need to insert a padding sequence after the ** stop codons if this is the case.

## Step 6 - Verify plasmids

One the plasmids have been designed using Twist's tools, you can verify that the insert sequences are what we expect.

We can convert the Genbank files to fasta format using `seqret` from EMBOSS:

```bash
mkdir -p plasmids_fasta plasmid_orfs

for gb in plasmids/*.gb; do seqret ${gb} -osformat fasta plasmids_fasta/$(basename ${gb} .gb).fasta; done

for gb in plasmids/*.gb; do getorf -sequence ${gb} -outseq plasmid_orfs/$(basename ${gb} .gb).fasta -circular Y -find 1; done
```

Make a BLAST database with translated ORFs from each plasmid, and search each of our original tagged sequences against it:

```bash
cat plasmid_orfs/*.fasta | makeblastdb -title plasmid_orfs -dbtype prot -out blast/plasmid_orfs_db

# Default BLAST output shows the alignments
blastp -query tagged_sequences.fasta \
       -db blast/plasmid_orfs_db \
       -max_target_seqs 1 \
       -out blast_results.txt

blastp -query tagged_sequences.fasta \
       -db blast/plasmid_orfs_db \
       -max_target_seqs 1 \
       -out blast_results.tsv \
       -outfmt 6
```

`blast_results.txt` shows the alignments, and `blast_results.tsv` shows tabulated BLAST results (with columns `qseqid sseqid pident length mismatch gapopen qstart qend sstart send evalue bitscore`).

We want to see `pident` = 100.0 for each row, and `mismatch` = 0 - if not, these sequences in the plasmid are identical to the original tagged designs.

>This procedure assmume the vector is generally sensible, and _only_ helps check that the expected binder coding sequences are present - it won't check for ribosome binding sites, antibiotic resistance markers etc.

If all looks okay, order your sequences. Good luck !

----

# His-tag placement script - additional details

## Output

```
Design  pdb_file        n_aa_type       c_aa_type       n_sasa  c_sasa  n_percent_sasa  c_percent_sasa        n_c_dist        n_dist_target   c_dist_target   n_target_contacts       c_target_contacts     tag
cxcr2_l128_s899440_mpnn2_model2 cxcr2_l128_s899440_mpnn2_model2.pdb     SER     ALA     93.61108.83   81.40   94.63   65.18   33.56   40.34   False   False   C
cxcr2_l125_s32063_mpnn4_model2  cxcr2_l125_s32063_mpnn4_model2.pdb      MET     SER     161.55113.53  87.32   98.72   32.89   39.05   22.50   False   False   N
cxcr2_l97_s817728_mpnn1_model2  cxcr2_l97_s817728_mpnn1_model2.pdb      MET     GLU     101.71176.51  54.98   92.90   18.22   29.92   28.07   False   False   C
```

- `n_percent_sasa` and `c_percent_sasa` is the percentage of solvent accessible surface area of the N- and C-terminal of the protein (these values are not exactly within the range 0 - 100 % due to the difficulty in defining a maximum possible SASA for a terminal residue).
- `n_c_dist` is the distance between the N- and C-terminal of the protein.
- `n_dist_target` and `c_dist_target` is the distance between the N- or C- terminal of the binder and the (centroid) of the specified `--distant-from` target residues.
- `tag` is the suggested termini for the His-tag.

## Tag selection logic

- If the termini is in contact with the target, exclude that termini from consideration
- If the termini has < 30 % SASA, exclude that termini from consideration (`--sasa-threshold 30`)
- Choose the termini most distant from the target, if it is significantly more distant from the target than the other termini (threshold of difference abs(n_dist_target - c_dist_target) > 5A `--more-distant-threshold 5`)
  - If both termini are of similar distance from the target (abs(n_dist_target - c_dist_target) < 5A), choose the termini with the highest SASA
- If no termini is suitable based on the above criteria, `tag` is None and will require manual inspection, possible exclusion of that design
