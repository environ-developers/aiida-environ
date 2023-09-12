# -*- coding: utf-8 -*-
from aiida import orm
from aiida.common.datastructures import CalcInfo
from aiida.common.folders import Folder
from aiida.engine import CalcJobProcessSpec

from aiida_quantumespresso.calculations.neb import NebCalculation
from aiida_environ.calculations.pw import EnvPwCalculation


class EnvNebCalculation(NebCalculation):
    _DEFAULT_DEBUG_FILE = 'environ.debug'
    
    @classmethod
    def define(cls, spec: CalcJobProcessSpec) -> None:
        """Define the process specification."""
        # yapf: disable
        super().define(spec)
        spec.input('metadata.options.parser_name', valid_type=str, default='environ.neb')
        spec.input('metadata.options.debug_filename', valid_type=str, default=cls._DEFAULT_DEBUG_FILE)
        spec.input('environ_parameters', valid_type=orm.Dict,
            help='The input parameters that are to be used to construct the input file.')
        
    def prepare_for_submission(self, folder: Folder) -> CalcInfo:
        calcinfo = NebCalculation.prepare_for_submission(self, folder)
        # add additional files to retrieve list
        calcinfo.retrieve_list.append(self.metadata.options.debug_filename)
        
        codeinfo = calcinfo.codes_info[0]
        codeinfo.cmdline_params.insert(0, '--environ')

        settings={}
        input_filecontent = EnvPwCalculation._generate_environinputdata(self.inputs.environ_parameters, self.inputs.first_structure, settings)

        # write the environ input file (name is fixed)
        with folder.open('environ.in', 'w') as handle:
            handle.write(input_filecontent)

        return calcinfo