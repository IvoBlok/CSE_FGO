import sys
import numpy as np
from ase import Atoms
from ase.visualize import view

def parse_cluster_file(file_path):
    """
    Parse the cluster data file and return atomic positions.

    Args:
        file_path (str): Path to the input file.

    Returns:
        positions (list): List of atomic positions as [x, y, z].
    """
    positions = []

    with open(file_path, 'r') as file:
        for line in file:
            coords = line.strip().split("::")
            x, y, z = float(coords[0]), float(coords[1]), float(coords[2])
            positions.append([x, y, z])

    return np.array(positions)

def compute_bonds(positions, cutoff=2.0):
    """
    Generate bonds based on a distance cutoff.

    Args:
        positions (array): Array of atomic positions.
        cutoff (float): Distance cutoff for bonding.

    Returns:
        list: List of tuples representing bonds (atom1, atom2).
    """
    bonds = []
    num_atoms = len(positions)

    for i in range(num_atoms):
        for j in range(i + 1, num_atoms):
            distance = np.linalg.norm(positions[i] - positions[j])
            if distance <= cutoff:
                bonds.append((i, j))

    return bonds

def save_to_xyz(file_path, positions, atom_types):
    """
    Save atomic positions and types to an XYZ file.
    
    Args:
        file_path (str): Path to output the XYZ file.
        positions (array): Array of atomic positions.
        atom_types (list): List of atomic symbols.
    """
    num_atoms = len(positions)

    with open(file_path, 'w') as file:
        # Write the number of atoms (first line of XYZ file)
        file.write(f"{num_atoms}\n")
        file.write("Atoms\n")  # Optional header line
        for i in range(num_atoms):
            # Write each atom and its position
            file.write(f"{atom_types[i]} {positions[i][0]:.6f} {positions[i][1]:.6f} {positions[i][2]:.6f}\n")

def main():
    if len(sys.argv) < 2:
        print("Usage: python generate_mol_with_atoms.py <path_to_cluster_file>")
        sys.exit(1)

    # Parse the cluster file
    file_path = sys.argv[1]
    positions = parse_cluster_file(file_path)

    # Assuming all atoms are carbon for simplicity
    atom_types = ["F"] * len(positions)

    # Generate bonds with a cutoff of 2.0 Å
    bonds = compute_bonds(positions, cutoff=1.0)

    # Create ASE Atoms object
    atoms = Atoms(atom_types, positions=positions)

    # Save atomic positions to an XYZ file
    output_xyz_file = "results/output_structure.xyz"
    save_to_xyz(output_xyz_file, positions, atom_types)

    # Inform the user that the XYZ file has been saved
    print(f"XYZ file with atomic positions saved to: {output_xyz_file}")

    # Visualize the structure with manually computed bonds (optional)
    view(atoms, bonds=bonds)

if __name__ == "__main__":
    main()
