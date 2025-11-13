from aiida.plugins.factories import DataFactory


def make_simple_structure():
    StructureData = DataFactory("core.structure")
    unit_cell = [[10.5835431576, 0, 0], [0, 10.5835431576, 0], [0, 0, 10.5835431576]]

    structure = StructureData(cell=unit_cell)
    structure.append_atom(
        position=(6.2629149502, 6.3834874485, 6.08553686585), symbols="O"
    )
    structure.append_atom(
        position=(7.1075976684, 5.8856629291, 6.08553686585), symbols="H"
    )
    structure.append_atom(
        position=(5.5740299405, 5.6858323849, 6.08553686585), symbols="H"
    )

    return structure


def make_simple_kpoints():
    KpointsData = DataFactory("core.array.kpoints")
    kpoints_mesh = KpointsData()
    kpoints_mesh.set_kpoints_mesh([1, 1, 1])

    return kpoints_mesh


def make_organic_structure():
    from ase.io import read

    atoms = read("NEUTRAL_017.in")
    StructureData = DataFactory("core.structure")
    structure = StructureData(ase=atoms)
    structure.label = "240 small neutral organic molecule set, id: 17"

    return structure


def make_simple_parameters():
    from aiida.orm import Dict

    parameters = {
        "CONTROL": {
            "calculation": "scf",
            "restart_mode": "from_scratch",
            "tprnfor": True,
        },
        "SYSTEM": {
            "ecutrho": 300, 
            "ecutwfc": 30
        },
        "ELECTRONS": {
            "conv_thr": 5.0e-9,
            "diagonalization": "rmm-davidson",
            "mixing_beta": 0.4,
            "electron_maxstep": 200,
        },
    }

    return Dict(dict=parameters)


def make_simple_environ_parameters():
    from aiida.orm import Dict

    environ_parameters = {
        "ENVIRON": {
            "environ_thr": 0.1,
            "environ_type": "water",
        },
        "BOUNDARY": {},
        "ELECTROSTATIC": {
            "tol": 1e-10,
        },
    }
    environ_parameters = Dict(dict=environ_parameters)

    return environ_parameters
