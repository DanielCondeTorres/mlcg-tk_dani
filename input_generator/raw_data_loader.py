import numpy as np
import os
from natsort import natsorted
from glob import glob
import h5py
from typing import Tuple, Optional, List, Generator
import mdtraj as md
import warnings
from pathlib import Path
from tqdm import tqdm
from .utils import chunker

class DatasetLoader:
    r"""
    Base loader object for a dataset. Defines the interface that all loader should have.

    As there is no standard way of saving the output of molecular dynamics simulations,
    the objective of this is class is to be an interface which will load the MD data
    from a given dataset stored in a local path. It will further process the data to
    remove thing that are not relevant to our system (for example, solvent coordinates
    and forces) and returned the clean coordinate and force as np.ndarrays and an mdtraj
    object that carries the topology

    The loader should be flexible enough to handle datasets with different molecules,
    multiple independent trajectories of different length and other things.


    """

    def get_traj_top(self, name: str, pdb_fn: str):
        r"""
        Function to get the topology associated with a trajectory in the dataset.

        PDB files represent molecules by putting the position of every atom in cartesian space.
        From the distances between atoms, covalent bonds and other things can be deduced.

        Note that, by convention, raw PDB files use angstroms as the unit of the coordinates.
        Engines such as Pymol and mdtraj convert this values to nanometers

        Parameters
        ----------
        name:
            Name of input sample (i.e. the molecule or object in the data)
        pdb_fn:
            String representing a path to a PDB file which has the topolgy for
        """
        raise NotImplementedError(f"Base class {self.__class__} has no implementation")

    def load_coords_forces(
        self,
        base_dir: str,
        name: str,
        stride: int = 1,
        batch: Optional[int] = None,
        n_batches: Optional[int] = 1,
    ) -> Tuple[np.ndarray, np.ndarray]:
        r"""
        Method to load the coordinate and force data

        This requires the base directory where files can be saved, the name of the molecule
        we are trying to load and a stride parameter. The stride is useful for cases where
        the data is saved too frequently.

        Parameters
        ----------
        base_dir:
            Path to coordinate and force files.
        name:
            Name of input sample
        stride : int
            Interval by which to stride loaded data
        batch: int or None
            if trajectories are loaded by batch, indicates the batch index to load
            must be set if n_batches > 1
        n_batches: int
            if greater than 1, divide the total trajectories to load into n_batches chunks
        """
        raise NotImplementedError(f"Base class {self.__class__} has no implementation")


class CATH_loader(DatasetLoader):
    """
    Loader object for original 50 CATH domain proteins
    """

    def get_traj_top(self, name: str, pdb_fn: str):
        """
        For a given CATH domain name, returns a loaded MDTraj object at the input resolution
        (generally atomistic) as well as the dataframe associated with its topology.

        Parameters
        ----------
        name:
            Name of input sample
        pdb_fn:
            Path to pdb structure file
        """
        pdb = md.load(pdb_fn.format(name))
        aa_traj = pdb.atom_slice(
            [a.index for a in pdb.topology.atoms if a.residue.is_protein]
        )
        top_dataframe = aa_traj.topology.to_dataframe()[0]
        return aa_traj, top_dataframe

    def load_coords_forces(
        self,
        base_dir: str,
        name: str,
        stride: int = 1,
        batch: Optional[int] = None,
        n_batches: Optional[int] = 1,
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        For a given CATH domain name, returns np.ndarray's of its coordinates and forces at
        the input resolution (generally atomistic)

        Parameters
        ----------
        base_dir:
            Path to coordinate and force files
        name:
            Name of input sample
        stride : int
            Interval by which to stride loaded data
        batch: int or None
            if trajectories are loaded by batch, indicates the batch index to load
            must be set if n_batches > 1
        n_batches: int
            if greater than 1, divide the total trajectories to load into n_batches chunks
        """

        outputs_fns = natsorted(glob(os.path.join(base_dir, f"output/{name}/*_part_*")))

        if n_batches > 1:
            raise NotImplementedError(
                "mol_num_batches can only be used for single-protein datasets for now"
            )

        aa_coord_list = []
        aa_force_list = []
        # load the files, checking against the mol dictionary
        for fn in outputs_fns:
            output = np.load(fn)
            coord = output["coords"]
            coord = 10.0 * coord  # convert nm to angstroms
            force = output["Fs"]
            force = force / 41.84  # convert to from kJ/mol/nm to kcal/mol/ang
            assert coord.shape == force.shape
            aa_coord_list.append(coord)
            aa_force_list.append(force)
        aa_coords = np.concatenate(aa_coord_list)[::stride]
        aa_forces = np.concatenate(aa_force_list)[::stride]
        return aa_coords, aa_forces


class CATH_ext_loader(DatasetLoader):
    """
    Loader object for extended dataset of CATH domain proteins
    """

    def get_traj_top(self, name: str, pdb_fn: str):
        """
        For a given CATH domain name, returns a loaded MDTraj object at the input resolution
        (generally atomistic) as well as the dataframe associated with its topology.

        Parameters
        ----------
        name:
            Name of input sample
        pdb_fn:
            Path to pdb structure file
        """
        pdb_fns = glob(pdb_fn.format(name))
        pdb = md.load(pdb_fns[0])
        aa_traj = pdb.atom_slice(
            [a.index for a in pdb.topology.atoms if a.residue.is_protein]
        )
        top_dataframe = aa_traj.topology.to_dataframe()[0]
        return aa_traj, top_dataframe

    def load_coords_forces(
        self,
        base_dir: str,
        name: str,
        stride: int = 1,
        batch: Optional[int] = None,
        n_batches: Optional[int] = 1,
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        For a given CATH domain name, returns np.ndarray's of its coordinates and forces at
        the input resolution (generally atomistic)

        Parameters
        ----------
        base_dir:
            Path to coordinate and force files
        name:
            Name of input sample
        """
        if n_batches > 1:
            raise NotImplementedError(
                "mol_num_batches can only be used for single-protein datasets for now"
            )
        traj_dirs = glob(os.path.join(base_dir, f"group_*/{name}_*/"))
        all_coords = []
        all_forces = []
        for traj_dir in tqdm(traj_dirs):
            traj_coords = []
            traj_forces = []
            fns = glob(os.path.join(traj_dir, "prod_out_full_output/*.npz"))
            fns.sort(key=lambda file: int(file.split("_")[-2]))
            last_parent_id = None
            for fn in fns:
                np_dict = np.load(fn, allow_pickle=True)
                current_id = np_dict["id"]
                parent_id = np_dict["parent_id"]
                if parent_id is not None:
                    assert parent_id == last_parent_id
                traj_coords.append(np_dict["coords"])
                traj_forces.append(np_dict["Fs"])
                last_parent_id = current_id
            traj_full_coords = np.concatenate(traj_coords)
            traj_full_forces = np.concatenate(traj_forces)
            if traj_full_coords.shape[0] != 25000:
                continue
            else:
                all_coords.append(traj_full_coords[::stride])
                all_forces.append(traj_full_forces[::stride])
        full_coords = np.concatenate(all_coords)
        full_forces = np.concatenate(all_forces)
        return full_coords, full_forces


class DIMER_loader(DatasetLoader):
    """
    Loader object for original dataset of mono- and dipeptide pairwise umbrella sampling simulations
    """

    def get_traj_top(self, name: str, pdb_fn: str):
        """
        For a given DIMER pair name, returns a loaded MDTraj object at the input resolution
        (generally atomistic) as well as the dataframe associated with its topology.

        Parameters
        ----------
        name:
            Name of input sample
        pdb_fn:
            Path to pdb structure file
        """
        pdb = md.load(pdb_fn.format(name))
        aa_traj = pdb.atom_slice(
            [a.index for a in pdb.topology.atoms if a.residue.is_protein]
        )
        top_dataframe = aa_traj.topology.to_dataframe()[0]
        return aa_traj, top_dataframe

    def load_coords_forces(
        self,
        base_dir: str,
        name: str,
        stride: int = 1,
        batch: Optional[int] = None,
        n_batches: Optional[int] = 1,
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        For a given DIMER pair name, returns np.ndarray's of its coordinates and forces at
        the input resolution (generally atomistic)

        Parameters
        ----------
        base_dir:
            Path to coordinate and force files
        name:
            Name of input sample
        stride : int
            Interval by which to stride loaded data
        batch: int or None
            if trajectories are loaded by batch, indicates the batch index to load
            must be set if n_batches > 1
        n_batches: int
            if greater than 1, divide the total trajectories to load into n_batches chunks
        """
        if n_batches > 1:
            raise NotImplementedError(
                "mol_num_batches can only be used for single-protein datasets for now"
            )

        with h5py.File(os.path.join(base_dir, "allatom.h5"), "r") as data:
            coord = data["MINI"][name]["aa_coords"][:][::stride]
            force = data["MINI"][name]["aa_forces"][:][::stride]

        # convert to kcal/mol/angstrom and angstrom
        # from kJ/mol/nm and nm
        coord = coord * 10
        force = force / 41.84

        return coord, force


class DIMER_ext_loader(DatasetLoader):
    """
    Loader object for extended dataset of mono- and dipeptide pairwise umbrella sampling simulations
    """

    def get_traj_top(self, name: str, pdb_fn: str):
        """
        For a given DIMER pair name, returns a loaded MDTraj object at the input resolution
        (generally atomistic) as well as the dataframe associated with its topology.

        Parameters
        ----------
        name:
            Name of input sample
        pdb_fn:
            Path to pdb structure file
        """
        pdb_fns = glob(pdb_fn.format(name))
        if len(pdb_fns) == 1:
            pdb = md.load(pdb_fns[0])
        elif len(pdb_fns) > 1:
            warnings.warn(
                f"Pattern {pdb_fn.format(name)} has more than one result",
                category=UserWarning,
            )
        else:
            return None, None
        aa_traj = pdb.atom_slice(
            [a.index for a in pdb.topology.atoms if a.residue.is_protein]
        )
        top_dataframe = aa_traj.topology.to_dataframe()[0]
        return aa_traj, top_dataframe

    def load_coords_forces(
        self,
        base_dir: str,
        name: str,
        stride: int = 1,
        batch: Optional[int] = None,
        n_batches: Optional[int] = 1,
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        For a given DIMER pair name, returns np.ndarray's of its coordinates and forces at
        the input resolution (generally atomistic)

        Parameters
        ----------
        base_dir:
            Path to coordinate and force files
        name:
            Name of input sample
        """
        if n_batches > 1:
            raise NotImplementedError(
                "mol_num_batches can only be used for single-protein datasets for now"
            )
        coord = np.load(
            glob(os.path.join(base_dir, f"dip_dimers_*/data/{name}_coord.npy"))[0],
            allow_pickle=True,
        )[::stride]
        force = np.load(
            glob(os.path.join(base_dir, f"dip_dimers_*/data/{name}_force.npy"))[0],
            allow_pickle=True,
        )[::stride]

        # convert to kcal/mol/angstrom and angstrom
        # from kJ/mol/nm and nm
        coord = coord * 10
        force = force / 41.84

        return coord, force


class Trpcage_loader(DatasetLoader):
    """
    Loader object for Trpcage simulation dataset
    """

    def get_traj_top(self, name: str, pdb_fn: str):
        """
        For a given name, returns a loaded MDTraj object at the input resolution
        (generally atomistic) as well as the dataframe associated with its topology.

        Parameters
        ----------
        name:
            Name of input sample
        pdb_fn:
            Path to pdb structure file
        """
        pdb = md.load(pdb_fn.format(name))
        aa_traj = pdb.atom_slice(
            [a.index for a in pdb.topology.atoms if a.residue.is_protein]
        )
        top_dataframe = aa_traj.topology.to_dataframe()[0]
        return aa_traj, top_dataframe

    def load_coords_forces(
        self,
        base_dir: str,
        name: str,
        stride: int = 1,
        batch: Optional[int] = None,
        n_batches: Optional[int] = 1,
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        For a given name, returns np.ndarray's of its coordinates and forces at
        the input resolution (generally atomistic)

        Parameters
        ----------
        base_dir:
            Path to coordinate and force files
        name:
            Name of input sample
        stride : int
            Interval by which to stride loaded data
        batch: int or None
            if trajectories are loaded by batch, indicates the batch index to load
            must be set if n_batches > 1
        n_batches: int
            if greater than 1, divide the total trajectories to load into n_batches chunks
        """
        coords_fns = natsorted(
            glob(
                os.path.join(base_dir, f"coords_nowater/trp_coor_folding-trpcage_*.npy")
            )
        )

        forces_fns = [
            fn.replace(
                "coords_nowater/trp_coor_folding", "forces_nowater/trp_force_folding"
            )
            for fn in coords_fns
        ]

        coords_fns = np.array(coords_fns)
        forces_fns = np.array(forces_fns)

        if n_batches > 1:
            assert batch is not None, "batch id must be set if more than 1 batch"
            chunk_ids = chunker(
                [i for i in range(len(coords_fns))], n_batches=n_batches
            )
            coords_fns = coords_fns[np.array(chunk_ids[batch])]
            forces_fns = forces_fns[np.array(chunk_ids[batch])]

        aa_coord_list = []
        aa_force_list = []
        # load the files, checking against the mol dictionary
        for cfn, ffn in tqdm(zip(coords_fns, forces_fns), total=len(coords_fns)):
            force = np.load(ffn)  # in AA
            coord = np.load(cfn)  # in kcal/mol/AA

            assert coord.shape == force.shape
            aa_coord_list.append(coord[::stride])
            aa_force_list.append(force[::stride])
        aa_coords = np.concatenate(aa_coord_list)
        aa_forces = np.concatenate(aa_force_list)
        return aa_coords, aa_forces


class Cln_loader(DatasetLoader):
    def get_traj_top(self, name: str, pdb_fn: str):
        pdb = md.load(pdb_fn.format(name))
        aa_traj = pdb.atom_slice(
            [a.index for a in pdb.topology.atoms if a.residue.is_protein]
        )
        top_dataframe = aa_traj.topology.to_dataframe()[0]
        return aa_traj, top_dataframe

    def load_coords_forces(
        self,
        base_dir: str,
        name: str,
        stride: int = 1,
        batch: Optional[int] = None,
        n_batches: Optional[int] = 1,
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        For a given name, returns np.ndarray's of its coordinates and forces at
        the input resolution (generally atomistic)

        Parameters
        ----------
        base_dir:
            Path to coordinate and force files
        name:
            Name of input sample
        stride : int
            Interval by which to stride loaded data
        batch: int or None
            if trajectories are loaded by batch, indicates the batch index to load
            must be set if n_batches > 1
        n_batches: int
            if greater than 1, divide the total trajectories to load into n_batches chunks
        """
        coords_fns = natsorted(
            glob(os.path.join(base_dir, f"coords_nowater/chig_coor_*.npy"))
        )

        forces_fns = [
            fn.replace("coords_nowater/chig_coor_", "forces_nowater/chig_force_")
            for fn in coords_fns
        ]

        coords_fns = np.array(coords_fns)
        forces_fns = np.array(forces_fns)

        if n_batches > 1:
            assert batch is not None, "batch id must be set if more than 1 batch"
            chunk_ids = chunker(
                [i for i in range(len(coords_fns))], n_batches=n_batches
            )
            coords_fns = coords_fns[np.array(chunk_ids[batch])]
            forces_fns = forces_fns[np.array(chunk_ids[batch])]

        aa_coord_list = []
        aa_force_list = []
        # load the files, checking against the mol dictionary
        for cfn, ffn in tqdm(zip(coords_fns, forces_fns), total=len(coords_fns)):
            force = np.load(ffn)  # in AA
            coord = np.load(cfn)  # in kcal/mol/AA

            assert coord.shape == force.shape
            aa_coord_list.append(coord[::stride])
            aa_force_list.append(force[::stride])
        aa_coords = np.concatenate(aa_coord_list)
        aa_forces = np.concatenate(aa_force_list)
        return aa_coords, aa_forces


class BBA_loader(DatasetLoader):
    """
    Loader object for CHARMM22* BBA simulation dataset
    """

    def get_traj_top(self, name: str, pdb_fn: str):
        """
        For a given name, returns a loaded MDTraj object at the input resolution
        (generally atomistic) as well as the dataframe associated with its topology.

        Parameters
        ----------
        name:
            Name of input sample
        pdb_fn:
            Path to pdb structure file
        """
        pdb = md.load(pdb_fn.format(name))
        aa_traj = pdb.atom_slice(
            [a.index for a in pdb.topology.atoms if a.residue.is_protein]
        )
        top_dataframe = aa_traj.topology.to_dataframe()[0]
        return aa_traj, top_dataframe

    def load_coords_forces(
        self,
        base_dir: str,
        name: str,
        stride: int = 1,
        batch: Optional[int] = None,
        n_batches: Optional[int] = 1,
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        For a given name, returns np.ndarray's of its coordinates and forces at
        the input resolution (generally atomistic)

        Parameters
        ----------
        base_dir:
            Path to coordinate and force files
        name:
            Name of input sample
        stride : int
            Interval by which to stride loaded data
        batch: int or None
            if trajectories are loaded by batch, indicates the batch index to load
            must be set if n_batches > 1
        n_batches: int
            if greater than 1, divide the total trajectories to load into n_batches chunks
        """
        coords_fns = natsorted(
            glob(os.path.join(base_dir, f"coords_nowater/bba_coor_folding-bba_*.npy"))
        )

        forces_fns = [
            fn.replace(
                "coords_nowater/bba_coor_folding", "forces_nowater/bba_force_folding"
            )
            for fn in coords_fns
        ]

        coords_fns = np.array(coords_fns)
        forces_fns = np.array(forces_fns)

        if n_batches > 1:
            assert batch is not None, "batch id must be set if more than 1 batch"
            chunk_ids = chunker(
                [i for i in range(len(coords_fns))], n_batches=n_batches
            )
            coords_fns = coords_fns[np.array(chunk_ids[batch])]
            forces_fns = forces_fns[np.array(chunk_ids[batch])]

        aa_coord_list = []
        aa_force_list = []
        # load the files, checking against the mol dictionary
        for cfn, ffn in tqdm(zip(coords_fns, forces_fns), total=len(coords_fns)):
            force = np.load(ffn)  # in AA
            coord = np.load(cfn)  # in kcal/mol/AA

            assert coord.shape == force.shape
            aa_coord_list.append(coord[::stride])
            aa_force_list.append(force[::stride])
        aa_coords = np.concatenate(aa_coord_list)
        aa_forces = np.concatenate(aa_force_list)
        return aa_coords, aa_forces


class Villin_loader(DatasetLoader):
    def get_traj_top(self, name: str, pdb_fn: str):
        pdb = md.load(pdb_fn.format(name))
        aa_traj = pdb.atom_slice(
            [a.index for a in pdb.topology.atoms if a.residue.is_protein]
        )
        top_dataframe = aa_traj.topology.to_dataframe()[0]
        return aa_traj, top_dataframe

    def load_coords_forces(
        self,
        base_dir: str,
        name: str,
        stride: int = 1,
        batch: Optional[int] = None,
        n_batches: Optional[int] = 1,
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        For a given name, returns np.ndarray's of its coordinates and forces at
        the input resolution (generally atomistic)

        Parameters
        ----------
        base_dir:
            Path to coordinate and force files
        name:
            Name of input sample
        stride : int
            Interval by which to stride loaded data
        batch: int or None
            if trajectories are loaded by batch, indicates the batch index to load
            must be set if n_batches > 1
        n_batches: int
            if greater than 1, divide the total trajectories to load into n_batches chunks
        """
        coords_fns = sorted(
            glob(os.path.join(base_dir, f"{name}/*_coords.npy"))
        )  # combining all trajectories from single starting structure

        forces_fns = sorted(glob(os.path.join(base_dir, f"{name}/*_forces.npy")))

        coords_fns = np.array(coords_fns)
        forces_fns = np.array(forces_fns)

        if n_batches > 1:
            assert batch is not None, "batch id must be set if more than 1 batch"
            chunk_ids = chunker(
                [i for i in range(len(coords_fns))], n_batches=n_batches
            )
            coords_fns = coords_fns[np.array(chunk_ids[batch])]
            forces_fns = forces_fns[np.array(chunk_ids[batch])]

        aa_coord_list = []
        aa_force_list = []
        for c, f in tqdm(zip(coords_fns, forces_fns), total=len(coords_fns)):
            coords = np.load(c)
            forces = np.load(f)
            assert coords.shape == forces.shape

            aa_coord_list.append(coords[::stride])
            aa_force_list.append(forces[::stride])

        aa_coords = np.concatenate(aa_coord_list)
        aa_forces = np.concatenate(aa_force_list)
        return aa_coords, aa_forces


class NTL9_loader(DatasetLoader):
    r"""
    Loader object for CHARMM22* NTL9 simulation dataset
    """

    def get_traj_top(self, name: str, pdb_fn: str):
        r"""
        For a given name, returns a loaded MDTraj object at the input resolution
        (generally atomistic) as well as the dataframe associated with its topology.

        Parameters
        ----------
        name:
            Name of input sample
        pdb_fn:
            Path to pdb structure file
        """
        pdb = md.load(pdb_fn.format(name))
        aa_traj = pdb.atom_slice(
            [a.index for a in pdb.topology.atoms if a.residue.is_protein]
        )
        top_dataframe = aa_traj.topology.to_dataframe()[0]
        return aa_traj, top_dataframe

    def load_coords_forces(
        self,
        base_dir: str,
        name: str,
        stride: int = 1,
        batch: Optional[int] = None,
        n_batches: Optional[int] = 1,
        chop: int = 5,
    ) -> Tuple[np.ndarray, np.ndarray]:
        r"""
        For a given name, returns np.ndarray's of its coordinates and forces at
        the input resolution (generally atomistic)

        Parameters
        ----------
        base_dir:
            Path to coordinate and force files
        name:
            Name of input sample
        stride : int
            Interval by which to stride loaded data
        batch: int or None
            if trajectories are loaded by batch, indicates the batch index to load
            must be set if n_batches > 1
        n_batches: int
            if greater than 1, divide the total trajectories to load into n_batches chunks
        chop: int
            initial frames to discard from every simulation. Necessary due to some problem frames.
        """
        traj_name_pat = "ntl9_coor_"
        coord_pattern = Path(base_dir) / "coords_nowater" / f"{traj_name_pat}*.npy"
        coords_fns = natsorted(glob(str(coord_pattern)))

        forces_fns = [
            fn.replace("coords_nowater", "forces_nowater").replace("coor", "force")
            for fn in coords_fns
        ]

        coords_fns = np.array(coords_fns)
        forces_fns = np.array(forces_fns)

        if n_batches > 1:
            assert batch is not None, "batch id must be set if more than 1 batch"
            chunk_ids = chunker(
                [i for i in range(len(coords_fns))], n_batches=n_batches
            )
            coords_fns = coords_fns[np.array(chunk_ids[batch])]
            forces_fns = forces_fns[np.array(chunk_ids[batch])]

        aa_coord_list = []
        aa_force_list = []
        # load the files, checking against the mol dictionary
        for cfn, ffn in tqdm(zip(coords_fns, forces_fns), total=len(coords_fns)):
            force = np.load(ffn)[chop::stride]  # in AA
            coord = np.load(cfn)[chop::stride]  # in kcal/mol/AA

            assert coord.shape == force.shape
            aa_coord_list.append(coord)
            aa_force_list.append(force)
        aa_coords = np.concatenate(aa_coord_list)
        aa_forces = np.concatenate(aa_force_list)
        return aa_coords, aa_forces


class ProteinG_loader(DatasetLoader):
    """
    Loader object for CHARMM22* Protein G simulation dataset
    """

    def get_traj_top(self, name: str, pdb_fn: str):
        """
        For a given name, returns a loaded MDTraj object at the input resolution
        (generally atomistic) as well as the dataframe associated with its topology.

        Parameters
        ----------
        name:
            Name of input sample
        pdb_fn:
            Path to pdb structure file
        """
        pdb = md.load(pdb_fn.format(name))
        aa_traj = pdb.atom_slice(
            [a.index for a in pdb.topology.atoms if a.residue.is_protein]
        )
        top_dataframe = aa_traj.topology.to_dataframe()[0]
        return aa_traj, top_dataframe

    def load_coords_forces(
        self,
        base_dir: str,
        name: str,
        stride: int = 1,
        batch: Optional[int] = None,
        n_batches: Optional[int] = 1,
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        For a given name, returns np.ndarray's of its coordinates and forces at
        the input resolution (generally atomistic)

        Parameters
        ----------
        base_dir:
            Path to coordinate and force files
        name:
            Name of input sample
        stride : int
            Interval by which to stride loaded data
        batch: int or None
            if trajectories are loaded by batch, indicates the batch index to load
            must be set if n_batches > 1
        n_batches: int
            if greater than 1, divide the total trajectories to load into n_batches chunks
        """
        coords_fns = natsorted(
            glob(
                os.path.join(
                    base_dir, f"coords_3_nowater/proteing_coor_folding-proteing-*.npy"
                )
            )
        )

        forces_fns = [
            fn.replace(
                "coords_3_nowater/proteing_coor_folding",
                "forces_3_nowater/proteing_force_folding",
            )
            for fn in coords_fns
        ]

        coords_fns = np.array(coords_fns)
        forces_fns = np.array(forces_fns)

        if n_batches > 1:
            assert batch is not None, "batch id must be set if more than 1 batch"
            chunk_ids = chunker(
                [i for i in range(len(coords_fns))], n_batches=n_batches
            )
            coords_fns = coords_fns[np.array(chunk_ids[batch])]
            forces_fns = forces_fns[np.array(chunk_ids[batch])]

        aa_coord_list = []
        aa_force_list = []
        # load the files, checking against the mol dictionary
        for cfn, ffn in tqdm(zip(coords_fns, forces_fns), total=len(coords_fns)):
            force = np.load(ffn)  # in AA
            coord = np.load(cfn)  # in kcal/mol/AA

            assert coord.shape == force.shape
            aa_coord_list.append(coord[::stride])
            aa_force_list.append(force[::stride])
        aa_coords = np.concatenate(aa_coord_list)
        aa_forces = np.concatenate(aa_force_list)
        return aa_coords, aa_forces


class A3D_loader(DatasetLoader):
    """
    Loader object for CHARMM22* A3D simulation dataset
    """

    def get_traj_top(self, name: str, pdb_fn: str):
        """
        For a given name, returns a loaded MDTraj object at the input resolution
        (generally atomistic) as well as the dataframe associated with its topology.

        Parameters
        ----------
        name:
            Name of input sample
        pdb_fn:
            Path to pdb structure file
        """
        pdb = md.load(pdb_fn.format(name))
        aa_traj = pdb.atom_slice(
            [a.index for a in pdb.topology.atoms if a.residue.is_protein]
        )
        top_dataframe = aa_traj.topology.to_dataframe()[0]
        return aa_traj, top_dataframe

    def load_coords_forces(
        self,
        base_dir: str,
        name: str,
        stride: int = 1,
        batch: Optional[int] = None,
        n_batches: Optional[int] = 1,
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        For a given name, returns np.ndarray's of its coordinates and forces at
        the input resolution (generally atomistic)

        Parameters
        ----------
        base_dir:
            Path to coordinate and force files
        name:
            Name of input sample
        stride : int
            Interval by which to stride loaded data
        batch: int or None
            if trajectories are loaded by batch, indicates the batch index to load
            must be set if n_batches > 1
        n_batches: int
            if greater than 1, divide the total trajectories to load into n_batches chunks
        """
        coords_fns = natsorted(
            glob(os.path.join(base_dir, f"coords_*_nowater/a3D_coor_folding-a3d-*.npy"))
        )
        forces_fns = []
        for coord_fn in coords_fns:
            if "coords_1_nowater" in coord_fn:
                forces_fns.append(
                    coord_fn.replace(
                        "coords_1_nowater/a3D_coor_folding",
                        "forces_1_nowater/a3D_force_folding",
                    )
                )
            elif "coords_2_nowater" in coord_fn:
                forces_fns.append(
                    coord_fn.replace(
                        "coords_2_nowater/a3D_coor_folding",
                        "forces_2_nowater/a3D_force_folding",
                    )
                )
            else:
                raise RuntimeError("wrong path in A3D loader")

        coords_fns = np.array(coords_fns)
        forces_fns = np.array(forces_fns)

        if n_batches > 1:
            assert batch is not None, "batch id must be set if more than 1 batch"
            chunk_ids = chunker(
                [i for i in range(len(coords_fns))], n_batches=n_batches
            )
            coords_fns = coords_fns[np.array(chunk_ids[batch])]
            forces_fns = forces_fns[np.array(chunk_ids[batch])]

        aa_coord_list = []
        aa_force_list = []
        # load the files, checking against the mol dictionary
        for cfn, ffn in tqdm(zip(coords_fns, forces_fns), total=len(coords_fns)):
            force = np.load(ffn)  # in AA
            coord = np.load(cfn)  # in kcal/mol/AA

            assert coord.shape == force.shape
            aa_coord_list.append(coord[::stride])
            aa_force_list.append(force[::stride])
        aa_coords = np.concatenate(aa_coord_list)
        aa_forces = np.concatenate(aa_force_list)
        return aa_coords, aa_forces


class OPEP_loader(DatasetLoader):
    """
    Loader for octapeptides dataset
    """

    def get_traj_top(self, name: str, pdb_fn: str):
        """
        For a given name, returns a loaded MDTraj object at the input resolution
        (generally atomistic) as well as the dataframe associated with its topology.

        Parameters
        ----------
        name:
            Name of input sample
        pdb_fn:
            Path to pdb structure file
        """
        pdb = md.load(pdb_fn.format(name))
        aa_traj = pdb.atom_slice(
            [a.index for a in pdb.topology.atoms if a.residue.is_protein]
        )
        top_dataframe = aa_traj.topology.to_dataframe()[0]
        return aa_traj, top_dataframe

    def load_coords_forces(
        self,
        base_dir: str,
        name: str,
        stride: int = 1,
        batch: Optional[int] = None,
        n_batches: Optional[int] = 1,
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        For a given name, returns np.ndarray's of its coordinates and forces at
        the input resolution (generally atomistic)

        Parameters
        ----------
        base_dir:
            Path to coordinate and force files
        name:
            Name of input sample
        stride : int
            Interval by which to stride loaded data
        batch: int or None
            if trajectories are loaded by batch, indicates the batch index to load
            must be set if n_batches > 1
        n_batches: int
            if greater than 1, divide the total trajectories to load into n_batches chunks
        """
        if n_batches > 1:
            raise NotImplementedError(
                "mol_num_batches can only be used for single-protein datasets for now"
            )

        coord_files = sorted(
            glob(os.path.join(base_dir, f"coords_nowater/opep_{name}/*.npy"))
        )
        if len(coord_files) == 0:
            coord_files = sorted(
                glob(os.path.join(base_dir, f"coords_nowater/coor_opep_{name}_*.npy"))
            )
        force_files = sorted(
            glob(os.path.join(base_dir, f"forces_nowater/opep_{name}/*.npy"))
        )
        if len(force_files) == 0:
            force_files = sorted(
                glob(os.path.join(base_dir, f"forces_nowater/force_opep_{name}_*.npy"))
            )

        aa_coord_list = []
        aa_force_list = []
        for c, f in zip(coord_files, force_files):
            coords = np.load(c)
            forces = np.load(f)
            aa_coord_list.append(coords[::stride])
            aa_force_list.append(forces[::stride])
        aa_coords = np.concatenate(aa_coord_list)
        aa_forces = np.concatenate(aa_force_list)
        return aa_coords, aa_forces


class HDF5_loader(DatasetLoader):
    r"""
    Base class for loading data stored in an HDF5 data format.

    The file can have any structure as long as all the
    independent trajectories are represented by groups
    that have keys "coords" and "Fs" with arrays of
    shape (n_frames,n_atoms,3)
    """

    @staticmethod
    def _get_all_traj_groups(h5file: h5py.File) -> List[str]:
        r"""
        Base unction to get all the possible trajectory groups.
        """
        traj_group_arr = []

        def visit_func(name, node):
            r"""
            Helper function to collect names of internal groups.
            """
            if isinstance(node, h5py.Group):
                node_keys = node.keys()
                if "coords" in node_keys and "Fs" in node_keys:
                    traj_group_arr.append(name)

        h5file.visititems(visit_func)

        return traj_group_arr

    @staticmethod
    def get_traj_groups(h5file: h5py.File) -> List[str]:
        r"""
        Function to get the  trajectory groups of this dataset.

        This is a wrapper over `_get_all_traj_groups` so that
        sublcasses can filter certain trajectories that are undesirable
        """
        return HDF5_loader._get_all_traj_groups(h5file)

    def get_traj_top(self, name: str, pdb_fn: str):
        """
        For a given name, returns a loaded MDTraj object at the input resolution
        (generally atomistic) as well as the dataframe associated with its topology.

        Parameters
        ----------
        name:
            Name of input sample
        pdb_fn:
            Path to pdb structure file
        """
        pdb = md.load(pdb_fn.format(name)).remove_solvent()
        aa_traj = pdb.atom_slice(
            [a.index for a in pdb.topology.atoms if a.residue.is_protein]
        )
        top_dataframe = aa_traj.topology.to_dataframe()[0]
        return aa_traj, top_dataframe

    def load_coords_forces(
        self,
        base_dir: str,
        name: str,
        stride: int = 1,
        batch: Optional[int] = None,
        n_batches: Optional[int] = 1,
    ) -> Tuple[np.ndarray, np.ndarray]:
        aa_coord_list = []
        aa_force_list = []

        with h5py.File(os.path.join(base_dir), "r") as dataset:
            traj_group_arr = self.get_traj_groups(dataset)
            traj_group_arr = np.array(traj_group_arr)
            if n_batches > 1:
                assert batch is not None, "batch id must be set if more than 1 batch"
                chunk_ids = chunker(
                    [i for i in range(len(traj_group_arr))], n_batches=n_batches
                )
                traj_group_arr = traj_group_arr[np.array(chunk_ids[batch])]

            if len(traj_group_arr) == 0:
                raise ValueError(f"No trajectories found in supplied h5")
            for traj_path in tqdm(traj_group_arr, total=len(traj_group_arr)):
                traj = dataset.get(traj_path)
                if traj is None:
                    raise ValueError(f"Failed to find trajectory in group {traj_path}")
                coord = traj["coords"][:]
                force = traj["Fs"][:]

                aa_coord_list.append(coord[::stride])
                aa_force_list.append(force[::stride])

        aa_coords = np.concatenate(aa_coord_list)
        aa_forces = np.concatenate(aa_force_list)

        return aa_coords, aa_forces


class SimInput_loader(DatasetLoader):
    """
    Loader for protein structures to be used in CG simulations
    """

    def get_traj_top(self, name: str, raw_data_dir: str):
        """
        For a given name, returns a loaded MDTraj object at the input resolution
        (generally atomistic) as well as the dataframe associated with its topology.

        Parameters
        ----------
        name:
            Name of input sample
        raw_data_dir:
            Path to pdb structure file
        """
        pdb = md.load(f"{raw_data_dir}{name}.pdb")
        input_traj = pdb.atom_slice(
            [a.index for a in pdb.topology.atoms if a.residue.is_protein]
        )
        top_dataframe = input_traj.topology.to_dataframe()[0]
        return input_traj, top_dataframe







class Lipids_loader(DatasetLoader):
    """
    Loader object for Lipids simulation dataset
    """

    def get_traj_top(self, name: str, pdb_fn: str, lipid_name: str = 'DPPC'):
        """
        For a given name, returns a loaded MDTraj object at the input resolution (generally atomistic) 
        as well as the dataframe associated with its topology.

        Parameters
        ----------
        name:
            Name of input sample
        pdb_fn:
            Path to pdb structure
        lipid_name:
            Name of the lipid residue to filter (e.g., 'DPPC', 'POPC')
        """
        pdb = md.load(pdb_fn)
        aa_traj = pdb.atom_slice(
            [a.index for a in pdb.topology.atoms if a.residue.name == lipid_name]
        )
        top_dataframe = aa_traj.topology.to_dataframe()[0]
        
        return aa_traj, top_dataframe
    
    def load_coords_forces_npy(
        self, base_dir: str, name: str
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        For a given name, returns np.ndarray's of its coordinates and forces at
        the input resolution (generally atomistic)

        Parameters
        ----------
        base_dir:
            Path to coordinate and force files
        name:
            Name of input sample
        """

        coords_fn = os.path.join(base_dir, f"*coords*.npy")

        forces_fn = os.path.join(base_dir, f"*forces*.npy")

        dims_fn = os.path.join(base_dir, f"*cg_dims*.npy")

        coord = np.load(coords_fn) # I don't have to convert from nm to angstrom because I am using MDAnalysis to save the coordinates and it does it already, otherwise I would need to convert it by multiplying by 10.
        force = np.load(forces_fn) / 4.184 # convert from kJ/mol/A (in MDAnalysis) to kcal/mol/ang
        dims = np.load(dims_fn)

        assert coord.shape == force.shape

        return coord, force, dims
    
    
    
    def generate_data(self, traj_dirs: list) -> Generator[Tuple[np.ndarray, np.ndarray], None, None]:
    """
    Generates coordinate and force data from trajectory directories.

    Parameters
    ----------
    traj_dirs : list
        A list of trajectory directory paths.

    Yields
    ------
    Tuple[np.ndarray, np.ndarray]
        Coordinates and forces extracted from each trajectory file.
    """
    for traj_dir in traj_dirs:
        file_paths = glob(os.path.join(traj_dir, "production_full_output/*.npz"))
        for file_path in file_paths:
            data = np.load(file_path, allow_pickle=True)
            coords = data["coords"].astype(np.float32)
            forces = data["Fs"].astype(np.float32)
            
            if coords.shape != forces.shape:
                raise ValueError("Mismatch in shapes of coordinates and forces")
            
            yield coords, forces
            
    def load_coords_forces_npz(self, base_dir: str, name: str) -> Generator[Tuple[np.ndarray, np.ndarray], None, None]:
        """
        Loads and yields batches of coordinates and forces from trajectory directories.

        Parameters
        ----------
        base_dir : str
            The base directory containing trajectory subdirectories.
        name : str
            The name of the dataset or simulation.

        Yields
        ------
        Tuple[np.ndarray, np.ndarray]
            A batch of concatenated coordinates and forces.
        """
        traj_dirs = [os.path.join(base_dir, f"{i}/") for i in range(1, 15)]  # Trajectory folders from 1 to 14
        batch_size = 100  # Reduce batch size to optimize memory usage

        data_generator = self.generate_data(traj_dirs)

        all_coords = []
        all_forces = []

        for coords, forces in data_generator:
            all_coords.append(coords)
            all_forces.append(forces)

            if len(all_coords) >= batch_size:
                full_coords = np.concatenate(all_coords, axis=0)
                full_forces = np.concatenate(all_forces, axis=0)
                print("Remaining data size:", full_coords.shape, full_forces.shape)
                yield full_coords, full_forces
                
                # Clear memory
                all_coords.clear()
                all_forces.clear()

        # Process any remaining data
        if all_coords:
            full_coords = np.concatenate(all_coords, axis=0)
            full_forces = np.concatenate(all_forces, axis=0)
            print("Remaining data size:", full_coords.shape, full_forces.shape)
            yield full_coords, full_forces