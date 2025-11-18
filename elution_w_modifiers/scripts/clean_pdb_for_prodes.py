#!/usr/bin/env python3
"""
Remove ACE/NMA capping groups from PDB files for ProDes compatibility
"""

import sys

def clean_pdb(input_pdb, output_pdb):
    """Remove ACE and NMA residues from PDB file"""
    
    excluded_residues = {'ACE', 'NMA', 'NME'}  # Common capping groups
    
    with open(input_pdb, 'r') as f:
        lines = f.readlines()
    
    cleaned_lines = []
    removed_count = 0
    
    for line in lines:
        # Keep non-ATOM/HETATM lines
        if not (line.startswith('ATOM') or line.startswith('HETATM')):
            cleaned_lines.append(line)
            continue
        
        # Check residue name (columns 18-20 in PDB format)
        if len(line) >= 20:
            res_name = line[17:20].strip()
            if res_name in excluded_residues:
                removed_count += 1
                continue
        
        cleaned_lines.append(line)
    
    # Write cleaned PDB
    with open(output_pdb, 'w') as f:
        f.writelines(cleaned_lines)
    
    print(f"Removed {removed_count} atoms from capping groups (ACE/NMA/NME)")
    print(f"Cleaned PDB saved to: {output_pdb}")

if __name__ == '__main__':
    if len(sys.argv) != 3:
        print("Usage: python clean_pdb_for_prodes.py input.pdb output.pdb")
        sys.exit(1)
    
    input_pdb = sys.argv[1]
    output_pdb = sys.argv[2]
    
    clean_pdb(input_pdb, output_pdb)
