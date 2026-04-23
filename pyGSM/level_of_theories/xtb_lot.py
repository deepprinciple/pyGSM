# standard library imports
import os
import re
import shutil
import subprocess
import sys
from os import path

# third party
import numpy as np

try:
    from xtb.interface import Calculator
    from xtb.utils import get_method, get_solvent
    from xtb.interface import Environment
    from xtb.libxtb import VERBOSITY_FULL
except:
    print('xtb is not imported')

# local application imports
sys.path.append(path.dirname(path.dirname(path.abspath(__file__))))
from utilities import manage_xyz, units, elements
try:
    from .base_lot import Lot
except:
    from base_lot import Lot


def _parse_turbo_gradient(gradient_path):
    """Parse Turbomole-format gradient file written by xtb.

    Returns gradient as a (natoms, 3) numpy array in Hartree/Bohr.
    """
    with open(gradient_path) as handle:
        lines = handle.readlines()

    in_section = False
    coords = []
    grads = []
    reading_grads = False

    for line in lines:
        stripped = line.strip()
        if stripped.startswith("$grad"):
            in_section = True
            continue
        if stripped == "$end":
            break
        if not in_section:
            continue
        if not stripped:
            continue
        if "cycle =" in stripped or "cartesian gradients" in stripped:
            continue

        tokens = stripped.split()
        if len(tokens) == 4 and not reading_grads:
            coords.append(tokens)
        elif len(tokens) == 3:
            reading_grads = True
            grads.append([float(t.replace("D", "E")) for t in tokens])

    if len(coords) != len(grads):
        raise RuntimeError(
            f"Mismatch in gradient file: {len(coords)} atoms vs "
            f"{len(grads)} gradient entries"
        )

    return np.array(grads, dtype=float)


class xTB_lot(Lot):
    def __init__(self, options):
        super(xTB_lot, self).__init__(options)

        numbers = []
        E = elements.ElementData()
        for a in manage_xyz.get_atoms(self.geom):
            elem = E.from_symbol(a)
            numbers.append(elem.atomic_num)
        self.numbers = np.asarray(numbers)

        # Use CLI (subprocess) mode when gxtb, gbe, or cosmo are requested,
        # since these options are not available in the xtb Python API.
        self._use_cli = bool(
            self.gxtb or self.gbe is not None or self.cosmo is not None
        )

    def _build_xtb_cmd(self, multiplicity):
        """Construct the xtb command-line argument list."""
        cmd = ["xtb", "coords.xyz", "--grad"]

        # Charge and spin
        cmd.extend(["--chrg", str(self.charge)])
        uhf = multiplicity - 1
        cmd.extend(["--uhf", str(uhf)])

        # Accuracy and electronic temperature
        cmd.extend(["--acc", str(self.xTB_accuracy)])
        cmd.extend(["--etemp", str(self.xTB_electronic_temperature)])

        # Hamiltonian / parametrization
        ham = self.xTB_Hamiltonian.lower()
        if "gfnff" in ham or ham == "gfn-ff":
            cmd.append("--gfnff")
        elif "gfn0" in ham or "0" in ham:
            cmd.extend(["--gfn", "0"])
        elif "gfn1" in ham or "1" in ham:
            cmd.extend(["--gfn", "1"])
        else:
            # Default to GFN2
            cmd.extend(["--gfn", "2"])

        # Solvent models (mutually exclusive)
        if self.gbe is not None:
            cmd.extend(["--gbe", str(self.gbe)])
        elif self.cosmo is not None:
            cmd.extend(["--cosmo", str(self.cosmo)])
        elif self.solvent is not None:
            # Fallback: use ALPB for standard solvent strings.
            cmd.extend(["--alpb", str(self.solvent)])

        # Extended tight-binding flag
        if self.gxtb:
            cmd.append("--gxtb")

        return cmd

    def _run_cli(self, geom, multiplicity, state):
        """Run xtb via command-line interface (subprocess)."""
        owd = os.getcwd()
        scratch_dir = "scratch/{}".format(self.node_id)
        os.makedirs(scratch_dir, exist_ok=True)

        # Write XYZ file
        xyz_fn = os.path.join(scratch_dir, "coords.xyz")
        manage_xyz.write_xyz(xyz_fn, geom, scale=1.0)

        # Locate xtb executable
        xtb_cmd = shutil.which("xtb")
        if xtb_cmd is None:
            raise RuntimeError("xtb executable not found in PATH")

        # Build and run command
        cmd = [xtb_cmd] + self._build_xtb_cmd(multiplicity)
        out_fn = os.path.join(scratch_dir, "xtb.out")

        env = os.environ.copy()
        env["OMP_NUM_THREADS"] = str(self.nproc)
        env["MKL_NUM_THREADS"] = str(self.nproc)

        with open(out_fn, "w") as out_handle:
            result = subprocess.run(
                cmd,
                cwd=scratch_dir,
                stdout=out_handle,
                stderr=subprocess.STDOUT,
                env=env,
            )

        if result.returncode != 0:
            raise RuntimeError(
                f"xtb calculation failed with return code {result.returncode}. "
                f"See {out_fn} for details."
            )

        # Parse energy from xtb.out
        energy_re = re.compile(r"TOTAL ENERGY\s+([-+\d\.]+)\s+Eh")
        energy = None
        with open(out_fn) as handle:
            for line in handle:
                match = energy_re.search(line)
                if match:
                    energy = float(match.group(1))
        if energy is None:
            raise RuntimeError("Could not parse energy from xtb output")

        # Parse gradient
        grad_fn = os.path.join(scratch_dir, "gradient")
        if not os.path.exists(grad_fn):
            raise RuntimeError(
                f"gradient file not found in {scratch_dir}"
            )
        gradient = _parse_turbo_gradient(grad_fn)

        self._Energies[(multiplicity, state)] = self.Energy(energy, 'Hartree')
        self._Gradients[(multiplicity, state)] = self.Gradient(
            gradient, 'Hartree/Bohr'
        )
        self.write_E_to_file()

        os.chdir(owd)
        return

    def _run_api(self, geom, multiplicity, state):
        """Run xtb via Python API (original behavior)."""
        coords = manage_xyz.xyz_to_np(geom)

        # convert to bohr
        positions = coords * units.ANGSTROM_TO_AU
        calc = Calculator(
            get_method(self.xTB_Hamiltonian),
            self.numbers,
            positions,
            charge=self.charge,
        )

        calc.set_accuracy(self.xTB_accuracy)
        calc.set_electronic_temperature(self.xTB_electronic_temperature)

        if self.solvent is not None:
            calc.set_solvent(get_solvent(self.solvent))

        calc.set_output('lot_jobs_{}.txt'.format(self.node_id))
        res = calc.singlepoint()
        calc.release_output()

        # energy in hartree
        self._Energies[(multiplicity, state)] = self.Energy(
            res.get_energy(), 'Hartree'
        )

        # grad in Hatree/Bohr
        self._Gradients[(multiplicity, state)] = self.Gradient(
            res.get_gradient(), 'Hartree/Bohr'
        )

        # write E to scratch
        self.write_E_to_file()

        return res

    def run(self, geom, multiplicity, state, verbose=False):
        if self._use_cli:
            return self._run_cli(geom, multiplicity, state)
        else:
            return self._run_api(geom, multiplicity, state)


if __name__ == "__main__":

    geom = manage_xyz.read_xyz('../data/ethylene.xyz')
    xyz = manage_xyz.xyz_to_np(geom)

    lot = xTB_lot.from_options(
        states=[(1, 0)], gradient_states=[(1, 0)], geom=geom, node_id=0
    )

    E = lot.get_energy(xyz, 1, 0)
    print(E)

    g = lot.get_gradient(xyz, 1, 0)
    print(g)
