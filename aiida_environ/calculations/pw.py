# -*- coding: utf-8 -*-
from aiida import orm
from aiida.common.datastructures import CalcInfo
from aiida.common.folders import Folder
from aiida.engine import CalcJobProcessSpec
from aiida_quantumespresso.calculations import (
    BasePwCpInputGenerator,
    _lowercase_dict,
    _uppercase_dict,
)
from aiida_quantumespresso.calculations.pw import PwCalculation
from aiida_quantumespresso.utils.convert import convert_input_to_namelist_entry

from aiida_environ.data.charge import EnvironChargeData
from aiida_environ.data.dielectric import EnvironDielectricData

class EnvPwCalculation(PwCalculation):
    """`CalcJob` implementation for the pw.x code of Quantum ESPRESSO + Environ."""

    _DEFAULT_DEBUG_FILE = 'environ.debug'

    @classmethod
    def define(cls, spec: CalcJobProcessSpec) -> None:
        """Define the process specification."""
        # yapf: disable
        super().define(spec)
        spec.input(
            'metadata.options.parser_name', 
            valid_type = str, 
            default = 'environ.pw'
        )
        spec.input(
            'metadata.options.debug_filename', 
            valid_type = str, 
            default = cls._DEFAULT_DEBUG_FILE
        )
        spec.input(
            'environ_parameters', 
            valid_type = orm.Dict,
            help = ('The input parameters that are to be used to construct '
                    'the input file.')
        )
        spec.input(
            'external_charges', 
            valid_type = EnvironChargeData, 
            required = False,
            help = 'External charges'
        )
        spec.input(
            'environ_dielectric',
            valid_type=EnvironDielectricData,
            required=False,
            help='Dielectric regions'
        )

    def prepare_for_submission(self, folder: Folder) -> CalcInfo:
        calcinfo = BasePwCpInputGenerator.prepare_for_submission(self, folder)

        # add additional files to retrieve list
        # TODO: Add retrieval of .cube files
        calcinfo.retrieve_list.append(self.metadata.options.debug_filename)
        # TODO consider lists of length > 1
        codeinfo = calcinfo.codes_info[0]
        # prepend the command line parametes with --environ (so that it appears just after the executable call)
        # NOTE this might not always be the case
        codeinfo.cmdline_params.insert(0, '--environ')

        if 'settings' in self.inputs:
            settings = _uppercase_dict(self.inputs.settings.get_dict(), dict_name='settings')
        else:
            settings = {}
        input_filecontent = self._generate_environinputdata(self.inputs.environ_parameters, self.inputs.structure, settings)

        if 'external_charges' in self.inputs:
            input_filecontent += self.inputs.external_charges.environ_output()
            self.inputs.environ_parameters['ENVIRON']['env_external_charges'] = len(self.inputs['external_charges'])
        if 'environ_dielectric' in self.inputs:
            input_filecontent += self.inputs.environ_diectric.environ_output()
            self.inputs.environ_parameters['ENVIRON']['env_dielectric_regions'] = len(self.inputs['environ_dielectric'])
        # write the environ input file (name is fixed)
        with folder.open('environ.in', 'w') as handle:
            handle.write(input_filecontent)

        return calcinfo

    @classmethod
    def _generate_environinputdata(cls, parameters, structure, settings):  # pylint: disable=invalid-name 
        # NOTE currently, `settings` does nothing but the plan is to have a user-friendly `parameter` equivalence
        # which has a lower precedence than `parameters`. That is, the user can set parameters via `settings`, and
        # tweak them more individually in `parameters`
             
        # The following input_params declaration is taken from the aiida-qe (3.1.0)  
        # I put the first-level keys as uppercase (i.e., namelist and card names)
        # and the second-level keys as lowercase
        # (deeper levels are unchanged)
        input_params = _uppercase_dict(parameters.get_dict(), dict_name='environ_parameters')
        input_params = {k: _lowercase_dict(v, dict_name=k) for k, v in input_params.items()}

        # set namelists_toprint explicitly, environ has 3 standard namelists which are expected
        inputfile = ''
        namelists_toprint = ['ENVIRON', 'BOUNDARY', 'ELECTROSTATIC']

        # To create a mapping from the species to an incremental fortran 1-based index
        # we use the alphabetical order as in the inputdata generation
        kind_names = sorted([kind.name for kind in structure.kinds])
        mapping_species = {kind_name: (index + 1) for index, kind_name in enumerate(kind_names)}

        for namelist_name in namelists_toprint:
            inputfile += f'&{namelist_name}\n'
            # namelist content; set to {} if not present, so that we leave an empty namelist
            namelist = input_params.pop(namelist_name, {})
            for key, value in sorted(namelist.items()):
                inputfile += convert_input_to_namelist_entry(key, value, mapping=mapping_species)
            inputfile += '/\n'

        return inputfile