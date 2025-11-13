# -*- coding: utf-8 -*-
from ase import Atoms, Atom
from aiida import load_profile
from aiida.engine import submit, run
from aiida.orm.utils import load_code
from aiida.plugins import WorkflowFactory, DataFactory
from aiida_quantumespresso.utils.resources import get_default_options
from make_inputs import (
    make_simple_environ_parameters,
    make_simple_parameters,
)

# Load your profile
load_profile()

# Load the code to execute
env_code = load_code('qe_environ@localhost')
phonopy_code = load_code('phonopy@localhost')
workchain = WorkflowFactory("environ.pka.env_phonon")

# Initiate your structure
StructureData = DataFactory('core.structure')
atoms = Atoms()
atoms.append(Atom('N', (-0.58, 0, 0)))
atoms.append(Atom('C', (0.58, 0, 0)))
atoms.append(Atom('H', (1.645, 0, 0)))
atoms.set_cell([10, 10, 10])
atoms.pbc = True
structure = StructureData(ase=atoms)

options = {
    'resources' : {
        'tot_num_mpiprocs': 4
    }
}

# Get your builder
builder = workchain.get_builder_from_protocol(
    pw_code = env_code, 
    phonopy_code = phonopy_code, 
    structure = structure,
    options = options
)

builder.metadata.label = "environ phonon example"
builder.metadata.description = "environ.pw with phonopy"

calc = submit(builder)
print(f'EnvRelaxPhonon<{calc.pk}> submitted to the queue.')
