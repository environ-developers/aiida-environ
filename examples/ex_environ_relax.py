# -*- coding: utf-8 -*-
from ase import Atoms, Atom
from aiida.engine import submit
from aiida.orm.nodes.data.upf import get_pseudos_from_structure
from aiida.orm.utils import load_code
from aiida.plugins.factories import WorkflowFactory
from aiida_quantumespresso.utils.resources import get_default_options
from make_inputs import (
    make_organic_structure,
    make_simple_environ_parameters,
    make_simple_kpoints,
    make_simple_parameters,
)

# try loading aiida-environ, everything stored as nodes already
code = load_code('qe_environ@localhost')
workchain = WorkflowFactory("environ.pw.relax")

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
    'resources': {
        'tot_num_mpiprocs': 4
    }
}
builder = workchain.get_builder_from_protocol(
    code, 
    structure, 
    options=options
)
builder.metadata.label = "environ example"
builder.metadata.description = "environ.pw relax"

parameters = make_simple_parameters()
parameters['CONTROL']['calculation'] = 'relax'

builder.base.pw.parameters = parameters

calc = submit(builder)
print(f'EnvPwRelax<{calc.pk}> submitted.')
