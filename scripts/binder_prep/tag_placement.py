#!/usr/bin/env python
# /// script
# requires-python = ">=3.12"
# dependencies = [
#     "biopython>=1.75",
#     "numpy",
# ]
# ///

import sys
import logging
import argparse
import warnings
import re
from pathlib import Path
from typing import Tuple, Optional, List, Dict, Any
import numpy as np # For distance calculation

from Bio.PDB.PDBParser import PDBParser
from Bio.PDB import Structure, Residue, Atom
from Bio.PDB.PDBIO import PDBIO
from Bio.PDB.SASA import ShrakeRupley
from Bio.PDB.PDBExceptions import PDBConstructionWarning
from Bio.PDB.Polypeptide import is_aa, PPBuilder


logging.basicConfig(
    stream=sys.stderr,
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)

# Suppress PDBConstructionWarning, typically about discontinuous chains or missing atoms.
warnings.filterwarnings("ignore", category=PDBConstructionWarning)
# Also suppress UserWarning from ShrakeRupley for atoms with unknown radii (e.g. H, or other non C,N,O,S atoms)
warnings.filterwarnings("ignore", message="WARNING: Unrecognized atom type")
warnings.filterwarnings("ignore", message="WARNING: Negative sasa result!") # Can happen with odd geometries

# Maximum residue accessible surface area in tripeptide (Chothia, 1976)
# https://www.genome.jp/entry/aaindex:CHOC760101
AA_SASA_CHOC760101: Dict[str, float] = {
    'ALA': 115.0, 'LEU': 170.0,
    'ARG': 225.0, 'LYS': 200.0,
    'ASN': 160.0, 'MET': 185.0,
    'ASP': 150.0, 'PHE': 210.0,
    'CYS': 135.0, 'PRO': 145.0,
    'GLN': 180.0, 'SER': 115.0,
    'GLU': 190.0, 'THR': 140.0,
    'GLY': 75.0,  'TRP': 255.0,
    'HIS': 195.0, 'TYR': 230.0,
    'ILE': 175.0, 'VAL': 155.0
}

# MAximum residue accessible surface area (theoretical) from Tien et al., 2013
# https://doi.org/10.1371/journal.pone.0080635
TIEN_2023_THEORETICAL: Dict[str, float] = {
    'ALA': 129.0, 'ARG': 274.0, 'ASN': 195.0, 'ASP': 193.0,
    'CYS': 167.0, 'GLU': 223.0, 'GLN': 225.0, 'GLY': 104.0,
    'HIS': 224.0, 'ILE': 197.0, 'LEU': 201.0, 'LYS': 236.0,
    'MET': 224.0, 'PHE': 240.0, 'PRO': 159.0, 'SER': 155.0,
    'THR': 172.0, 'TRP': 285.0, 'TYR': 263.0, 'VAL': 174.0
}

def parse_distant_from_string(distant_from_str: Optional[str]) -> Optional[List[Tuple[str, int]]]:
    """Parses a string like 'A118,B20' into [('A', 118), ('B', 20)]."""
    if not distant_from_str:
        return None
    
    parsed_residues: List[Tuple[str, int]] = []
    parts = distant_from_str.split(',')
    for part in parts:
        part = part.strip()
        if not part:
            continue
        try:
            chain_id = part[0]
            res_num_str = part[1:]
            if not chain_id.isalpha() or not res_num_str.isdigit():
                raise ValueError(f"Residue identifier '{part}' must be a letter followed by numbers.")
            res_num = int(res_num_str)
            parsed_residues.append((chain_id.upper(), res_num))
        except (IndexError, ValueError) as e:
            logger.warning(f"Invalid format for distant-from residue '{part}': {e}. Skipping this entry.")
            continue
    
    return parsed_residues if parsed_residues else None

#
# Helper functions for get_terminal_residue_sasa_from_computed_structure
#

def _get_sasa_and_percent_sasa(
    residue: Residue.Residue, 
    pdb_id: str, 
    chain_id_str: str, 
    terminal_type: str
) -> Tuple[Optional[float], Optional[float], Optional[str]]:
    """Calculates SASA, percent SASA, and AA type for a given residue."""
    sasa: Optional[float] = None
    percent_sasa: Optional[float] = None
    aa_type: Optional[str] = residue.get_resname().upper()

    if not hasattr(residue, 'sasa'):
        logger.error(
            f"SASA attribute not found on {terminal_type}-terminal residue for chain '{chain_id_str}' in PDB ID {pdb_id}."
        )
        return None, None, aa_type
    
    sasa = round(float(getattr(residue, 'sasa')), 2)

    try:
        if aa_type:
            standard_sasa = TIEN_2023_THEORETICAL.get(aa_type)
            if standard_sasa and standard_sasa > 0:
                percent_sasa = round((sasa / standard_sasa) * 100, 2)
            else:
                logger.debug(f"Standard SASA for {terminal_type}-terminal residue {aa_type} not found or is zero in {pdb_id}, chain {chain_id_str}.")
        else:
            logger.warning(f"{terminal_type}-terminal residue type was None for {pdb_id}, chain {chain_id_str}. Cannot calculate percent SASA.")
    except Exception as e:
        logger.warning(f"Error calculating {terminal_type}-terminal percent SASA for {aa_type} in {pdb_id}, chain {chain_id_str}: {e}")
    
    return sasa, percent_sasa, aa_type

def _get_ca_coord(residue: Residue.Residue, terminal_type: str, chain_id_str: str, pdb_id: str) -> Optional[np.ndarray]:
    """Safely retrieves the C-alpha coordinate of a residue."""
    if 'CA' in residue:
        return residue['CA'].get_coord()
    else:
        logger.warning(f"CA atom not found in {terminal_type}-terminal residue of chain {chain_id_str} in {pdb_id}.")
        return None

def _calculate_distance_to_target_center(
    terminal_ca_coord: Optional[np.ndarray],
    target_ca_coords: List[np.ndarray],
    terminal_type: str,
    pdb_id: str
) -> Optional[float]:
    """Calculates distance from a terminal CA to the geometric center of target CAs."""
    if terminal_ca_coord is None or not target_ca_coords:
        return None
    try:
        geometric_center = np.mean(target_ca_coords, axis=0)
        distance = round(float(np.linalg.norm(terminal_ca_coord - geometric_center)), 2)
        return distance
    except Exception as e:
        logger.warning(f"Error calculating {terminal_type}-terminal distance to target center in {pdb_id}: {e}")
        return None

def _check_terminal_target_contacts(
    terminal_residue: Residue.Residue,
    all_target_residues_for_contacts: List[Residue.Residue],
    terminal_type: str,
    pdb_id: str,
    contact_distance_threshold: float = 6.0
) -> bool:
    """Checks for atomic contacts between a terminal residue and target residues."""
    if not all_target_residues_for_contacts:
        logger.info(f"No target residues found for {terminal_type}-terminal contact calculation in {pdb_id}.")
        return False

    for term_atom in terminal_residue:
        for target_res in all_target_residues_for_contacts:
            for target_atom in target_res:
                try:
                    distance = np.linalg.norm(term_atom.get_coord() - target_atom.get_coord())
                    if distance < contact_distance_threshold:
                        return True  # Contact found
                except Exception as e:
                    logger.debug(f"Could not calculate distance between {term_atom} (in {terminal_type}-term) and {target_atom} in {pdb_id}: {e}")
                    continue # Skip this atom pair
    return False # No contact found

#
# TODO: Consider using https://github.com/kalininalab/spherecon for relative SASA calculation instead
#
def compute_terminii_stats(
    structure: Structure.Structure, 
    chain_id: str,
    target_residues_spec: Optional[List[Tuple[str, int]]] = None
) -> Tuple[Optional[float], Optional[float], Optional[float], Optional[float], 
           Optional[str], Optional[str], Optional[float],
           Optional[float], Optional[float], Optional[bool], Optional[bool], Optional[str]]:
    """
    Retrieves SASA, type, inter-terminal C-alpha distance, and distances to target residues.
    Uses helper functions for modularity.
    Returns: (n_sasa, c_sasa, n_percent_sasa, c_percent_sasa, 
              n_aa_type, c_aa_type, n_c_dist,
              n_dist_target, c_dist_target, n_target_contacts, c_target_contacts, sequence)
    """
    pdb_id = structure.id
    # pdb_file_path = structure.xtra.get('pdb_path', 'N/A') if hasattr(structure, 'xtra') else 'N/A' # Not directly used in this refined version but good for context

    n_sasa: Optional[float] = None
    c_sasa: Optional[float] = None
    n_percent_sasa: Optional[float] = None
    c_percent_sasa: Optional[float] = None
    n_aa_type: Optional[str] = None
    c_aa_type: Optional[str] = None
    n_c_dist: Optional[float] = None
    n_dist_target: Optional[float] = None
    c_dist_target: Optional[float] = None
    n_target_contacts: Optional[bool] = None # Initialize to None, helper will return bool
    c_target_contacts: Optional[bool] = None # Initialize to None, helper will return bool
    sequence: Optional[str] = None

    try:
        model = structure[0]
        if chain_id not in model:
            logger.warning(f"Chain '{chain_id}' not found in PDB ID {pdb_id}.")
            return None, None, None, None, None, None, None, None, None, None, None, None

        chain = model[chain_id]

        peptides = []
        try:
            ppb = PPBuilder()
            peptides = list(ppb.build_peptides(chain))
        except Exception as e:
            logger.warning(f"PPBuilder failed for chain '{chain_id}' in {pdb_id}: {e}")

        if not peptides:
            logger.warning(f"No polypeptides found in chain '{chain_id}' of PDB ID {pdb_id}.")
            return None, None, None, None, None, None, None, None, None, None, None, None

        sequence = "".join([str(p.get_sequence()) for p in peptides])
        if not sequence:
            sequence = None # Ensure it is None if empty string
            
        aa_residues: List[Residue.Residue] = [res for p in peptides for res in p]

        if not aa_residues:
            logger.warning(f"No standard amino acid residues found in chain '{chain_id}' of PDB ID {pdb_id}.")
            return None, None, None, None, None, None, None, None, None, None, None, sequence

        n_terminal_residue = aa_residues[0]
        c_terminal_residue = aa_residues[-1]

        n_sasa, n_percent_sasa, n_aa_type = _get_sasa_and_percent_sasa(n_terminal_residue, pdb_id, chain_id, "N")
        c_sasa, c_percent_sasa, c_aa_type = _get_sasa_and_percent_sasa(c_terminal_residue, pdb_id, chain_id, "C")

        ca_n_coord = _get_ca_coord(n_terminal_residue, "N", chain_id, pdb_id)
        ca_c_coord = _get_ca_coord(c_terminal_residue, "C", chain_id, pdb_id)
        
        if ca_n_coord is not None and ca_c_coord is not None:
            try:
                n_c_dist = round(float(np.linalg.norm(ca_n_coord - ca_c_coord)), 2)
            except Exception as e:
                logger.warning(f"Error calculating N-C distance for chain {chain_id} in {pdb_id}: {e}")
        # else: # Implicitly n_c_dist remains None if CAs are missing, log handled in _get_ca_coord
            # logger.warning(f"Cannot calculate N-C distance due to missing CA atom(s) in chain {chain_id} of {pdb_id}.")

        if target_residues_spec:
            target_ca_coords_list: List[np.ndarray] = []
            all_target_residues_for_contacts_list: List[Residue.Residue] = []
            
            # Process specified target residues to get their CA coordinates and full residue objects
            for target_chain_id_spec, target_res_num_spec in target_residues_spec:
                try:
                    if target_chain_id_spec not in model:
                        logger.warning(f"Target chain '{target_chain_id_spec}' not found in {pdb_id}. Skipping target {target_chain_id_spec}{target_res_num_spec}.")
                        continue
                    target_chain_obj = model[target_chain_id_spec]
                    res_id_tuple = (' ', target_res_num_spec, ' ') # Standard HETATM flag, resseq, and icode
                    if res_id_tuple not in target_chain_obj:
                        logger.warning(f"Target residue {target_chain_id_spec}{target_res_num_spec} (ID: {res_id_tuple}) not found in {pdb_id}. Skipping.")
                        continue
                    
                    target_residue_obj = target_chain_obj[res_id_tuple]
                    
                    target_ca = _get_ca_coord(target_residue_obj, f"target {target_chain_id_spec}{target_res_num_spec}", pdb_id, chain_id) # pdb_id, chain_id for logging context
                    if target_ca is not None:
                        target_ca_coords_list.append(target_ca)
                    
                    all_target_residues_for_contacts_list.append(target_residue_obj)
                except KeyError:
                    logger.warning(f"KeyError accessing target residue {target_chain_id_spec}{target_res_num_spec} in {pdb_id}. Skipping.")
                except Exception as e:
                    logger.warning(f"Error processing target residue {target_chain_id_spec}{target_res_num_spec} in {pdb_id}: {e}. Skipping.")

            # Calculate distances to the geometric center of target CAs
            if target_ca_coords_list:
                n_dist_target = _calculate_distance_to_target_center(ca_n_coord, target_ca_coords_list, "N", pdb_id)
                c_dist_target = _calculate_distance_to_target_center(ca_c_coord, target_ca_coords_list, "C", pdb_id)
            elif target_residues_spec: # Log only if targets were specified but no CAs were collected
                logger.info(f"No valid target CA atoms collected from {target_residues_spec} for distance calculation in {pdb_id}.")

            # Check for contacts between terminal residues and target residues
            if all_target_residues_for_contacts_list:
                n_target_contacts = _check_terminal_target_contacts(n_terminal_residue, all_target_residues_for_contacts_list, "N", pdb_id)
                c_target_contacts = _check_terminal_target_contacts(c_terminal_residue, all_target_residues_for_contacts_list, "C", pdb_id)
            elif target_residues_spec: # Log only if targets were specified but no residues were collected for contacts
                 logger.info(f"No target residues collected from {target_residues_spec} for contact calculation in {pdb_id}.")
        
        return (n_sasa, c_sasa, n_percent_sasa, c_percent_sasa, 
                n_aa_type, c_aa_type, n_c_dist, 
                n_dist_target, c_dist_target, 
                n_target_contacts, c_target_contacts, sequence)

    except KeyError as e: # Should be less likely if initial chain check passes
        logger.warning(f"Chain '{chain_id}' caused KeyError in PDB ID {pdb_id}: {e}.")
        return None, None, None, None, None, None, None, None, None, None, None, None
    except IndexError: # Should be less likely if aa_residues check passes
        logger.warning(f"Chain '{chain_id}' caused IndexError (e.g., no residues) in PDB ID {pdb_id}.")
        return None, None, None, None, None, None, None, None, None, None, None, None
    except Exception as e:
        logger.error(f"Unexpected error processing chain '{chain_id}' in PDB ID {pdb_id}: {e}", exc_info=True)
        return None, None, None, None, None, None, None, None, None, None, None, None

def create_pdb_with_tag_pseudoatom(
    structure: Structure.Structure,
    chain_id: str,
    terminus: str,
    output_path: Path,
    atomtype: str = 'HG',
    restype: str = 'HG',
    chain_id_tag: str = 'Z',
    element: str = 'HG'
) -> bool:
    """
    Creates a copy of the PDB structure with a pseudoatom added at the CA position
    of the specified terminus (N or C).
    
    Args:
        structure: The original PDB structure
        chain_id: The chain identifier 
        terminus: Either "N" or "C" for the terminus
        output_path: Path to write the modified PDB
        atomtype: Atom name for the pseudoatom (default: 'HG')
        restype: Residue name for the pseudoatom (default: 'HG')
        element: Element type for the pseudoatom (default: 'HG')
        
    Returns:
        True if successful, False otherwise
    """
    try:
        # Make a copy of the structure to avoid modifying the original
        import copy
        modified_structure = copy.deepcopy(structure)
        
        model = modified_structure[0]
        if chain_id not in model:
            logger.warning(f"Chain '{chain_id}' not found in structure {structure.id}")
            return False
            
        chain = model[chain_id]
        aa_residues = [res for res in chain.get_residues() if is_aa(res, standard=True)]
        
        if not aa_residues:
            logger.warning(f"No standard amino acid residues found in chain '{chain_id}'")
            return False
            
        # Get the appropriate terminal residue
        if terminus == "N":
            terminal_residue = aa_residues[0]
        elif terminus == "C":
            terminal_residue = aa_residues[-1]
        else:
            logger.warning(f"Invalid terminus '{terminus}'. Must be 'N' or 'C'")
            return False
            
        # Get CA coordinate
        if 'CA' not in terminal_residue:
            logger.warning(f"CA atom not found in {terminus}-terminal residue")
            return False
            
        ca_coord = terminal_residue['CA'].get_coord()
        
        # Create pseudoatom as a HETATM record with chain ID Z
        # Format atom name with proper spacing
        atom_fullname = f' {atomtype} ' if len(atomtype) == 2 else f' {atomtype}'
        if len(atomtype) == 1:
            atom_fullname = f' {atomtype}  '
        
        pseudoatom = Atom.Atom(
            name=atomtype,
            coord=ca_coord,
            bfactor=100.0,
            occupancy=1.0,
            altloc=' ',
            fullname=atom_fullname,
            serial_number=None,
            element=element
        )
        
        # Create a new HG residue in chain Z
        from Bio.PDB import Chain, Residue
        
        # Get or create chain Z
        if chain_id_tag not in model:
            chain_z = Chain.Chain(chain_id_tag)
            model.add(chain_z)
        else:
            chain_z = model[chain_id_tag]
        
        # Create pseudoatom residue (HETATM)
        pseudoatom_residue = Residue.Residue(
            id=(f'{restype}', 1, ' '),  # ('H_' prefix indicates HETATM, resseq=1, icode=' ')
            resname=restype,
            segid=' '
        )
        
        # Add the pseudoatom to the residue
        pseudoatom_residue.add(pseudoatom)
        
        # Add the residue to chain Z
        chain_z.add(pseudoatom_residue)
        
        # Write the modified structure
        output_path.parent.mkdir(parents=True, exist_ok=True)
        pdb_io = PDBIO()
        pdb_io.set_structure(modified_structure)
        pdb_io.save(str(output_path))
        
        logger.info(f"Created modified PDB with {atomtype} pseudoatom at {terminus}-terminus: {output_path}")
        return True
        
    except Exception as e:
        logger.error(f"Error creating PDB with {atomtype} pseudoatom: {e}", exc_info=True)
        return False

# TODO: Logic here probably need to de-emphasise the SASA threshold
#       and weight distance from target as more important
def determine_his_tag_placement(
    n_sasa: Optional[float],
    c_sasa: Optional[float],
    n_percent_sasa: Optional[float],
    c_percent_sasa: Optional[float],
    n_dist_target: Optional[float],
    c_dist_target: Optional[float],
    n_target_contacts: Optional[bool],
    c_target_contacts: Optional[bool],
    sasa_threshold_percent: float,
    more_distant_threshold_angstrom: float
) -> Optional[str]:
    """
    Determines the preferred terminus for His-tag placement based on SASA,
    distance to target, and contacts.

    Rules:
    1. Exclude termini in contact with the target.
    2. Exclude termini with percent SASA < sasa_threshold_percent.
    3. If both eligible:
        a. If one is significantly more distant (> more_distant_threshold_angstrom), choose it.
        b. Else (similar distance), choose the one with higher absolute SASA.
        c. If SASAs are equal (and distances similar), return None (ambiguous).
    4. If only one terminus is eligible, choose it.
    5. If neither is eligible, return None.
    """

    n_eligible = True
    # Rule 1: Contact with target
    if n_target_contacts is True:
        n_eligible = False
    # Rule 2: SASA threshold (only if still eligible)
    if n_eligible and (n_percent_sasa is None or n_percent_sasa < sasa_threshold_percent):
        n_eligible = False

    c_eligible = True
    # Rule 1: Contact with target
    if c_target_contacts is True:
        c_eligible = False
    # Rule 2: SASA threshold (only if still eligible)
    if c_eligible and (c_percent_sasa is None or c_percent_sasa < sasa_threshold_percent):
        c_eligible = False

    if n_eligible and c_eligible:
        # Both eligible. n_sasa, c_sasa, n_percent_sasa, c_percent_sasa are guaranteed non-None here.
        # This is because eligibility requires n_percent_sasa (or c_percent_sasa) to be non-None and above threshold,
        # and _get_sasa_and_percent_sasa ensures that if percent_sasa is non-None, absolute sasa is also non-None.
        
        # Ensure type checker understands these are floats now if eligible.
        # This is more for logical flow than Python's dynamic typing.
        current_n_sasa: float = n_sasa # type: ignore 
        current_c_sasa: float = c_sasa # type: ignore

        can_compare_distances = n_dist_target is not None and c_dist_target is not None

        if can_compare_distances:
            # Ensure type checker understands these are floats for comparison if can_compare_distances is True
            current_n_dist_target: float = n_dist_target # type: ignore
            current_c_dist_target: float = c_dist_target # type: ignore
            dist_diff = abs(current_n_dist_target - current_c_dist_target)

            # Rule 3a: Significantly more distant
            if dist_diff > more_distant_threshold_angstrom:
                return "N" if current_n_dist_target > current_c_dist_target else "C"
            else: # Rule 3b: Similar distance, choose by highest absolute SASA
                if current_n_sasa > current_c_sasa:
                    return "N"
                elif current_c_sasa > current_n_sasa:
                    return "C"
                else: # Rule 3c: Equal SASA, similar distance
                    return None # Ambiguous
        else: # Cannot compare distances (e.g., target not specified or CA coords missing)
              # Fallback to highest absolute SASA
            if current_n_sasa > current_c_sasa:
                return "N"
            elif current_c_sasa > current_n_sasa:
                return "C"
            else: # Equal SASA, no distance info to differentiate
                return None # Ambiguous
                
    elif n_eligible: # Rule 4: Only N is eligible
        return "N"
    elif c_eligible: # Rule 4: Only C is eligible
        return "C"
    else: # Rule 5: Neither is eligible
        return None

def main():
    parser = argparse.ArgumentParser(
        description="Calculate Solvent Accessible Surface Area (SASA) for N and C terminal "
                    "amino acid residues, their types, N-C distance, and distance to specified target residues in PDB files.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    parser.add_argument(
        "input_pdbs",
        type=str,
        nargs='+', # Allow one or more arguments
        help="Path(s) to PDB files or directories containing PDB files (scanned recursively). "
             "Can be used with shell globs like /path/*/*.pdb.",
    )
    parser.add_argument(
        "--binder-chain",
        type=str,
        default="B",
        help="Identifier of the binder chain.",
    )
    parser.add_argument(
        "--output",
        type=str,
        default="-",
        help="Path to the output TSV file. Use '-' for stdout.",
    )
    parser.add_argument(
        "--sasa-probe-radius",
        type=float,
        default=1.4,
        help="Radius of the probe for SASA calculation (in Angstroms).",
    )
    parser.add_argument(
        "--sasa-n-points",
        type=int,
        default=100,
        help="Number of points for SASA sphere resolution (higher is more precise but slower).",
    )
    parser.add_argument(
        "--distant-from",
        type=str,
        default=None,
        help="Comma-separated list of target residues for distance calculation, e.g., 'A118,A142,C50'. "
             "Distances from N/C termini to the geometric center of these residues' CA atoms will be calculated."
    )
    parser.add_argument(
        "--sasa-threshold",
        type=float,
        default=30.0,
        help="Minimum percent SASA for a terminus to be considered suitable (e.g., 30 for 30%%)."
    )
    parser.add_argument(
        "--more-distant-threshold",
        type=float,
        default=5.0,
        help="Minimum difference in Angstroms for one terminus to be considered 'significantly more distant' from the target."
    )
    parser.add_argument(
        "--output-pdb-path",
        type=str,
        default=None,
        help="Directory path to output modified PDB files with tag marker at chosen terminus. "
             "Files will be named {original_name}_tag_placement.pdb"
    )

    args = parser.parse_args()

    pdb_files_to_process: List[Path] = []
    if args.input_pdbs:
        for path_str in args.input_pdbs:
            current_path = Path(path_str)
            if current_path.is_file():
                # Check suffix explicitly for .pdb
                if current_path.suffix.lower() == '.pdb':
                    pdb_files_to_process.append(current_path.resolve()) # Resolve to get absolute path
                else:
                    logger.warning(f"Input path '{path_str}' is a file but not a PDB file (expected .pdb extension). Skipping.")
            elif current_path.is_dir():
                found_in_dir = list(current_path.rglob("*.pdb"))
                if found_in_dir:
                    pdb_files_to_process.extend([p.resolve() for p in found_in_dir]) # Resolve to get absolute paths
                else:
                    logger.info(f"No PDB files found in directory '{path_str}'.")
            else:
                # This handles cases where path_str is not a file or directory (e.g., broken symlink, or non-existent path from glob)
                logger.warning(f"Input path '{path_str}' is not a valid PDB file or directory, or it does not exist. Skipping.")
    
    # Remove duplicates while attempting to preserve order (dict.fromkeys is a common way)
    # Using resolve() above helps in making duplicates more apparent if symlinks/relative paths point to same file.
    pdb_files_found: List[Path] = []
    if pdb_files_to_process:
        pdb_files_found = list(dict.fromkeys(pdb_files_to_process))


    if not pdb_files_found:
        logger.warning(f"No PDB files were found based on the input arguments: {args.input_pdbs}. Exiting.")
        sys.exit(0) # Graceful exit if no files, as this might be expected with globs

    logger.info(f"Found {len(pdb_files_found)} unique PDB file(s) to process.")

    target_residues_for_distance = parse_distant_from_string(args.distant_from)
    if args.distant_from and not target_residues_for_distance:
        logger.warning(f"Specified --distant-from '{args.distant_from}' but no valid residues could be parsed. Proceeding without distance calculation to targets.")
    elif target_residues_for_distance:
        logger.info(f"Target residues for distance calculation: {target_residues_for_distance}")

    if args.output_pdb_path:
        logger.info(f"Modified PDB files with tag markers will be written to: {args.output_pdb_path}")

    pdb_parser = PDBParser(QUIET=True) 
    sasa_calculator = ShrakeRupley(probe_radius=args.sasa_probe_radius, n_points=args.sasa_n_points)

    output_lines = ["Design\tpdb_file\tSequence\tn_aa_type\tc_aa_type\tn_sasa\tc_sasa\tn_percent_sasa\tc_percent_sasa\tn_c_dist\tn_dist_target\tc_dist_target\tn_target_contacts\tc_target_contacts\ttag"] # Added new columns

    processed_count = 0
    for pdb_file in pdb_files_found:
        design_name = pdb_file.stem
        design_name = re.sub(r"_model\d+$", "", design_name)
        logger.info(f"Processing {pdb_file.name} (path: {pdb_file})...")

        try:
            structure: Structure.Structure = pdb_parser.get_structure(design_name, str(pdb_file))
            if not hasattr(structure, 'xtra'):
                structure.xtra = {}
            structure.xtra['pdb_path'] = str(pdb_file)
        except Exception as e:
            logger.error(f"Could not parse PDB file {pdb_file}: {e}", exc_info=False)
            logger.debug(f"Detailed parsing error for {pdb_file}:", exc_info=True)
            continue

        try:
            sasa_calculator.compute(structure, level='R') 
        except Exception as e:
            logger.error(f"SASA computation failed for {pdb_file}: {e}. Skipping.", exc_info=False)
            logger.debug(f"Detailed SASA error for {pdb_file}:", exc_info=True)
            continue
        
        n_sasa, c_sasa, n_percent_sasa, c_percent_sasa, \
        n_aa_type, c_aa_type, n_c_dist, \
        n_dist_target, c_dist_target, \
        n_target_contacts, c_target_contacts, sequence = compute_terminii_stats(
            structure, args.binder_chain, target_residues_for_distance
        )

        if n_sasa is not None and c_sasa is not None:
            # Determine tag using the new logic
            tag = determine_his_tag_placement(
                n_sasa, c_sasa, n_percent_sasa, c_percent_sasa,
                n_dist_target, c_dist_target,
                n_target_contacts, c_target_contacts,
                args.sasa_threshold, args.more_distant_threshold
            )
            
            n_perc_sasa_str = f"{n_percent_sasa:.2f}" if n_percent_sasa is not None else ""
            c_perc_sasa_str = f"{c_percent_sasa:.2f}" if c_percent_sasa is not None else ""
            n_aa_type_str = n_aa_type if n_aa_type is not None else ""
            c_aa_type_str = c_aa_type if c_aa_type is not None else ""
            n_c_dist_str = f"{n_c_dist:.2f}" if n_c_dist is not None else ""
            n_dist_target_str = f"{n_dist_target:.2f}" if n_dist_target is not None else ""
            c_dist_target_str = f"{c_dist_target:.2f}" if c_dist_target is not None else ""
            n_target_contacts_str = str(n_target_contacts) if n_target_contacts is not None else ""
            c_target_contacts_str = str(c_target_contacts) if c_target_contacts is not None else ""
            # Convert tag to empty string for output if it's None
            tag_str = tag if tag is not None else "-"
            sequence_str = sequence if sequence is not None else ""

            # Create modified PDB with Hg marker if output path is specified and tag is determined
            if args.output_pdb_path and tag is not None:
                output_pdb_dir = Path(args.output_pdb_path)
                output_pdb_filename = f"{design_name}_tag_placement.pdb"
                output_pdb_path = output_pdb_dir / output_pdb_filename
                
                success = create_pdb_with_tag_pseudoatom(structure, args.binder_chain, tag, output_pdb_path)
                if not success:
                    logger.warning(f"Failed to create modified PDB for {design_name}")

            output_lines.append(
                f"{design_name}\t{pdb_file.name}\t{sequence_str}\t{n_aa_type_str}\t{c_aa_type_str}\t"
                f"{n_sasa:.2f}\t{c_sasa:.2f}\t{n_perc_sasa_str}\t{c_perc_sasa_str}\t{n_c_dist_str}\t"
                f"{n_dist_target_str}\t{c_dist_target_str}\t{n_target_contacts_str}\t{c_target_contacts_str}\t{tag_str}"
            )
            processed_count +=1
        else:
            logger.warning(
                f"Could not retrieve valid terminal SASA (or other critical data like AA types) for chain '{args.binder_chain}' "
                f"in {pdb_file}. Skipping this file in output."
            )
    
    logger.info(f"Processed {processed_count} PDB files successfully out of {len(pdb_files_found)} found.")

    if args.output == "-":
        if sys.stdout.isatty() and not output_lines[1:]: # Check if there's data beyond the header
             logger.info("No data to print to stdout.")
        else:
            for line in output_lines:
                print(line)
    else:
        output_file_path = Path(args.output)
        try:
            output_file_path.parent.mkdir(parents=True, exist_ok=True)
            with open(output_file_path, "w") as f:
                for line in output_lines:
                    f.write(line + "\n") # Corrected to write newline character
            logger.info(f"Output written to {output_file_path.resolve()}")
        except IOError as e:
            logger.error(f"Could not write to output file {output_file_path}: {e}", exc_info=True)
            sys.exit(1)
        except Exception as e:
            logger.error(f"An unexpected error occurred while writing output to {output_file_path}: {e}", exc_info=True)
            sys.exit(1)

if __name__ == "__main__":
    main() 