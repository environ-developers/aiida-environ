# -*- coding: utf-8 -*-
"""
Workchain to perform relaxations and phonon calculations on conjugate acid base pairs
"""
from aiida import orm
from aiida.common import AttributeDict, exceptions
from aiida.common.lang import type_check
from aiida.engine import ToContext, WorkChain, append_, if_, while_
from aiida.plugins import WorkflowFactory, DataFactory, CalculationFactory
from aiida_quantumespresso.common.types import RelaxType
from aiida_quantumespresso.workflows.protocols.utils import ProtocolMixin
from aiida.orm import load_group, load_code, StructureData, ArrayData, to_aiida_type
import numpy as np

AcidBaseWorkChain = WorkflowFactory("environ.pka.acid_base")
EnvPwRelaxWorkChain = WorkflowFactory("environ.pw.relax")
EnvPwBaseWorkChain = WorkflowFactory("environ.pw.base")

class AcidBaseParameterSweepWorkChain(WorkChain, ProtocolMixin):
    """
        WorkChain to perform Relaxations and Phonon calculations on conjugate acid base pairs followed by a parameter
        sweep
    """

    @classmethod
    def define(cls, spec):
        """Define the process specification."""
        # yapf: disable
        super().define(spec)

        spec.expose_inputs(
            AcidBaseWorkChain,
            exclude=("clean_workdir")
        )

        spec.input_namespace(
            "parameters",
            dynamic=True,
            valid_type=orm.Dict,
            help="Parameters to test for Acid Base Pairs",
        )

        spec.input(
            "clean_workdir",
            valid_type=orm.Bool,
            default=lambda: orm.Bool(False),
            help='If `True`, work directories of all called calculation '
                 'will be cleaned at the end of execution.',
        )

        spec.input(
            "clean_parameter_workdir",
            valid_type=orm.Bool,
            default=lambda: orm.Bool(True),
            help='If `True`, work directories of all called calculations for parameters'
                 'will be cleaned at the end of execution.',
        )

        spec.input(
            "parameter_fail_hard",
            valid_type=orm.Bool,
            default=lambda: orm.Bool(False),
            help='If `True`, stops the workchain if any parameter calculation fails during parameter sweep',
        )
        
        spec.input(
            "run_parameter_parallel",
            valid_type=orm.Bool,
            default=lambda: orm.Bool(True),
            help='If `True`, runs the acid and base calculations simultaneously for each test parameter'
        )
        
        spec.input(
            "parameter_relax",
            valid_type=orm.Bool,
            default=lambda: orm.Bool(False),
            help='If `True`, performs geometry optimization on each tested parameter'
        )
        
        spec.input(
            "parameter.parallelization",
            valid_type=orm.Dict,
            required=False,
            serializer=to_aiida_type,
            help='Parallelization options to pass to each parameter calculation'
        )
        spec.input(
            "parameter.options",
            valid_type=orm.Dict,
            required=False,
            serializer=to_aiida_type,
            help='options for the metadata to pass to each parameter calculation'
        )
        spec.input(
            "parameter.max_iterations",
            valid_type=orm.Int,
            required=False,
            serializer=to_aiida_type,
            help='The maximum number of iterations each Parameter workchain will restart the process to finish successfully'
        )

        spec.outline(
            cls.setup,
            cls.acid_base_relax_phonon,
            cls.inspect_acid_base_relax_phonon,
            while_(cls.has_remaining_parameters)(
                if_(cls.should_run_acid_base_parameter_parallel)(
                    cls.run_acid_base_parameter_parallel,
                ).else_(
                    cls.run_parameter_acid,
                    cls.run_parameter_base,
                ),
                cls.inspect_parameter_results,
                
            ),
            cls.gather_results,
        )

        spec.expose_outputs(
            AcidBaseWorkChain,
        )

        spec.output_namespace(
            "acid.results",
            dynamic=True,
            valid_type=orm.Dict,
            help="Results for each set of parameters for acid",
        )

        spec.output_namespace(
            "base.results",
            dynamic=True,
            valid_type=orm.Dict,
            help="Results for each set of parameters for base",
        )

        spec.exit_code(
            400, 'ERROR_RELAX_PHONON_FAILED',
            message='Acid/Base relaxation or phonon calculation failed',
        )
        spec.exit_code(
            401, 'ERROR_PARAMETER_SUB_PROCESS_FAILED',
            message='Parameter sweep calculation failed with parameter_fail_hard set to True',
        )

    @classmethod
    def get_builder_from_protocol(
            cls,
            code: orm.Code,
            phonopy_code: orm.Code,
            acid_structure: StructureData,
            base_structure: StructureData,
            protocol: orm.Dict = None,
            overrides: orm.Dict = None,
            options=None,
            relax_type=RelaxType.POSITIONS,
            pseudo_family='SSSP/1.3/PBE/precision',
            clean_workdir: bool = False,
            clean_phonon_workdir: bool = True,
            clean_parameter_workdir: bool = True,
            **kwargs,
    ):
        builder = AcidBaseWorkChain.get_builder_from_protocol(
            code=code,
            phonopy_code=phonopy_code,
            acid_structure=acid_structure,
            base_structure=base_structure,
            protocol=protocol,
            overrides=overrides,
            options=options,
            relax_type=relax_type,
            pseudo_family=pseudo_family,
            clean_workdir=clean_workdir,
            clean_phonon_workdir=clean_phonon_workdir,
            **kwargs
        )
        builder.clean_parameter_workdir = clean_parameter_workdir
        return builder

    def setup(self):
        """Set up context variables for the workchain"""
        self.ctx.acid_parameters = list(self.inputs["parameters"].values())
        self.ctx.base_parameters = list(self.inputs["parameters"].values())
        self.ctx.acid_parameter_results = {}
        self.ctx.base_parameter_results = {}
        self.ctx.run_count_parameters = 0 
        return

    def acid_base_relax_phonon(self):
        inputs = AttributeDict(self.exposed_inputs(AcidBaseWorkChain))
        future = self.submit(AcidBaseWorkChain, **inputs)
        self.report(f'submitting acid `AcidBaseWorkChain` <PK={future.pk}> <UUID={future.uuid}>.')
        self.to_context(**{f'acid_base_relax_phonon': future})
        return

    def inspect_acid_base_relax_phonon(self):
        workchain = self.ctx.acid_base_relax_phonon

        if not workchain.is_finished_ok:
            self.report(
                f'AcidBaseRelaxPhononWorkChain` with <PK={workchain.pk}> <UUID={workchain.uuid} failed '
                f'with exit status {workchain.exit_status}'
            )
            return self.exit_codes.ERROR_RELAX_PHONON_FAILED
        else:
            self.report(f'Acid Base Relaxation and Phonon workchain '
                        f'<PK={workchain.pk}> <UUID={workchain.uuid} finished.')
            self.out_many(
                self.exposed_outputs(self.ctx.acid_base_relax_phonon, AcidBaseWorkChain, agglomerate=False)
            )
        return
    
    def should_run_acid_base_parameter_parallel(self):
        return self.inputs.run_parameter_parallel

    def run_acid_base_parameter_parallel(self):
        """Run the calculations for the parameter on Acid and Base in parallel"""
        self.report("Running acid and base parameters in parallel")
        self.run_parameter_acid()
        self.run_parameter_base()

    def has_remaining_parameters(self):
        """Check if there are any remaining untested parameters."""
        return len(self.ctx.acid_parameters) > 0

    def run_parameter_acid(self):
        from aiida_quantumespresso.workflows.protocols.utils import recursive_merge
        acid_inputs = AttributeDict(self.exposed_inputs(AcidBaseWorkChain)).acid
        base_inputs = acid_inputs.base
        environ_parameters = acid_inputs.solution.environ_parameters.get_dict()
        environ_parameters['ENVIRON']['environ_restart'] = True
        base_inputs.pw.environ_parameters = environ_parameters
        parameters = base_inputs.pw.parameters.get_dict()
        if self.inputs.parameter_relax:
            parameters['CONTROL']['calculation'] = 'relax'
        else:
            parameters['CONTROL']['calculation'] = 'scf'
            parameters['CONTROL']['tprnfor'] = False
            parameters.pop('IONS', None)
            parameters.pop('CELL', None)
        parameters['CONTROL']['restart_mode'] = 'from_scratch'
        parameters['ELECTRONS']['startingpot'] = 'file'
        parameters['ELECTRONS']['startingwfc'] = 'file'
        base_inputs.pw.parameters = parameters
        base_inputs.pw.parent_folder = self.ctx.acid_base_relax_phonon.outputs.acid.solution.scf.remote_folder

        test_parameters = self.ctx.acid_parameters.pop(0).get_dict()
        base_inputs.pw = recursive_merge(base_inputs.pw, test_parameters)
        base_inputs['pw']['structure'] = self.ctx.acid_base_relax_phonon.outputs.acid.solution.scf.output_structure
        base_inputs.clean_workdir = self.inputs.clean_workdir or self.inputs.clean_parameter_workdir 
        if 'parallelization' in self.inputs.parameter:
            base_inputs.pw.parallelization = self.inputs.parameter.parallelization
        if 'options' in self.inputs.parameter:
            base_inputs.pw.metadata.options = self.inputs.parameter.parallelization
        base_inputs.metadata.label = "acid_parameter"
        if 'max_iterations' in self.inputs.parameter:
            base_inputs.max_iterations = self.inputs.parameter.max_iterations
        running = self.submit(EnvPwBaseWorkChain, **base_inputs)
        self.report(f'submitting acid `EnvPwBaseWorkChain` <PK={running.pk}> <UUID={running.uuid}>')
        self.to_context(acid_subprocesses=append_(running))

    def run_parameter_base(self):
        from aiida_quantumespresso.workflows.protocols.utils import recursive_merge
        acid_inputs = AttributeDict(self.exposed_inputs(AcidBaseWorkChain)).base
        base_inputs = acid_inputs.base
        environ_parameters = acid_inputs.solution.environ_parameters.get_dict()
        environ_parameters['ENVIRON']['environ_restart'] = True
        base_inputs.pw.environ_parameters = environ_parameters
        parameters = base_inputs.pw.parameters.get_dict()
        if self.inputs.parameter_relax:
            parameters['CONTROL']['calculation'] = 'relax'
        else:
            parameters['CONTROL']['calculation'] = 'scf'
            parameters['CONTROL']['tprnfor'] = False
            parameters.pop('IONS',None)
            parameters.pop('CELL',None)

        parameters['CONTROL']['restart_mode'] = 'from_scratch'
        parameters['ELECTRONS']['startingpot'] = 'file'
        parameters['ELECTRONS']['startingwfc'] = 'file'
        base_inputs.pw.parameters = parameters
        base_inputs.pw.parent_folder = self.ctx.acid_base_relax_phonon.outputs.base.solution.scf.remote_folder

        test_parameters = self.ctx.base_parameters.pop(0).get_dict()
        base_inputs.pw = recursive_merge(base_inputs.pw, test_parameters)
        base_inputs['pw']['structure'] = self.ctx.acid_base_relax_phonon.outputs.base.solution.scf.output_structure
        base_inputs.clean_workdir = self.inputs.clean_workdir or self.inputs.clean_parameter_workdir 
        if 'parallelization' in self.inputs.parameter:
            base_inputs.pw.parallelization = self.inputs.parameter.parallelization
        if 'options' in self.inputs.parameter:
            base_inputs.pw.metadata.options = self.inputs.parameter.parallelization
        if 'max_iterations' in self.inputs.parameter:
            base_inputs.max_iterations = self.inputs.parameter.max_iterations
        base_inputs.metadata.label = "base_parameter"

        running = self.submit(EnvPwBaseWorkChain, **base_inputs)
        self.report(f'submitting base `EnvPwBaseWorkChain` <PK={running.pk}> <UUID={running.uuid}>')
        self.to_context(base_subprocesses=append_(running))

    def inspect_parameter_results(self):
        """Verify that the tested parameter finished successfully for both the acid and base calculations."""
        last_parameter_acid = self.ctx.acid_subprocesses[-1]
        acid_pk = last_parameter_acid.pk
        acid_uuid = last_parameter_acid.uuid
        acid_inputs = last_parameter_acid.inputs.pw.environ_parameters.get_dict()
        if not last_parameter_acid.is_finished_ok:
            status = last_parameter_acid.exit_status
            self.report(f"Warning: Acid subprocess <PK={acid_pk}> <UUID={acid_uuid}> failed with exit status {status}")
            self.ctx.acid_parameter_results[f"{self.ctx.run_count_parameters}"] = orm.Dict({
                "alpha": acid_inputs['BOUNDARY']['alpha'],
                "field_factor": acid_inputs['BOUNDARY']['field_factor'],
                "field_asymmetry": acid_inputs['BOUNDARY']['field_asymmetry'],
                "energy": np.finfo(np.float64).max,
            })
        else:
            self.report(f"Acid subprocess <PK={acid_pk}> <UUID={acid_uuid}> finished")
            self.ctx.acid_parameter_results[f"{self.ctx.run_count_parameters}"] = orm.Dict({
                "alpha": acid_inputs['BOUNDARY']['alpha'],
                "field_factor": acid_inputs['BOUNDARY']['field_factor'],
                "field_asymmetry": acid_inputs['BOUNDARY']['field_asymmetry'],
                "energy": last_parameter_acid.outputs.output_parameters.get_dict()['energy'],
            })

        last_parameter_base = self.ctx.base_subprocesses[-1]
        base_pk = last_parameter_base.pk
        base_uuid = last_parameter_base.uuid
        base_inputs = last_parameter_base.inputs.pw.environ_parameters.get_dict()
        if not last_parameter_base.is_finished_ok:
            status = last_parameter_base.exit_status
            self.report(f"Warning: Base subprocess <PK={base_pk}> <UUID={base_uuid}> failed with exit status {status}")
            self.ctx.base_parameter_results[f"{self.ctx.run_count_parameters}"] = orm.Dict({
                "alpha": base_inputs['BOUNDARY']['alpha'],
                "field_factor": base_inputs['BOUNDARY']['field_factor'],
                "field_asymmetry": base_inputs['BOUNDARY']['field_asymmetry'],
                "energy": np.finfo(np.float64).max,
            })
        else:
            self.report(f"Base subprocess <PK={base_pk}> <UUID={base_uuid}> finished")
            self.ctx.base_parameter_results[f"{self.ctx.run_count_parameters}"] = orm.Dict({
                "alpha": base_inputs['BOUNDARY']['alpha'],
                "field_factor": base_inputs['BOUNDARY']['field_factor'],
                "field_asymmetry": base_inputs['BOUNDARY']['field_asymmetry'],
                "energy": last_parameter_base.outputs.output_parameters.get_dict()['energy'],
            })

        if not last_parameter_acid.is_finished_ok or not last_parameter_base.is_finished_ok:
            if self.inputs.parameter_fail_hard.value:
                return self.exit_codes.ERROR_PARAMETER_SUB_PROCESS_FAILED
        self.ctx.run_count_parameters += 1
        return

    def gather_results(self):
        self.out('acid.results', self.ctx.acid_parameter_results)
        self.out('base.results', self.ctx.base_parameter_results)
        self.report(f"AcidBaseParameterSweepWorkChain finished")
        return