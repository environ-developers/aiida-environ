# -*- coding: utf-8 -*-
import traceback

from aiida import orm
from aiida.common import exceptions
from aiida_quantumespresso.parsers.parse_raw.pw import reduce_symmetries
from aiida_quantumespresso.parsers.pw import PwParser
from aiida_quantumespresso.utils.mapping import get_logging_container


class EnvPwParser(PwParser):
    def parse(self, **kwargs):
        """Parse the retrieved files of a completed `EnvPwCalculation` into output nodes.

        Two nodes that are expected are the default 'retrieved' `FolderData` node which will store the retrieved files
        permanently in the repository. The second required node is a filepath under the key `retrieved_temporary_files`
        which should contain the temporary retrieved files.

        This code is taken from the PwParser class, which it inherits, but contains a few minor changes. The raw parsing of
        the environ modified output file has some additional steps and the subsequent dict output has additional entries that are
        intended to be read by this plugin.
        """
        dir_with_bands = None
        self.exit_code_xml = None
        self.exit_code_stdout = None
        self.exit_code_parser = None

        try:
            settings = self.node.inputs.settings.get_dict()
        except exceptions.NotExistent:
            settings = {}

        # Look for optional settings input node and potential 'parser_options' dictionary within it
        parser_options = settings.get(self.get_parser_settings_key(), None)

        # Verify that the retrieved_temporary_folder is within the arguments if temporary files were specified
        if self.node.base.attributes.get("retrieve_temporary_list", None):
            try:
                dir_with_bands = kwargs["retrieved_temporary_folder"]
            except KeyError:
                return self.exit(self.exit_codes.ERROR_NO_RETRIEVED_TEMPORARY_FOLDER)

        parameters = self.node.inputs.parameters.get_dict()
        environ_parameters = self.node.inputs.environ_parameters.get_dict()
        parsed_xml, logs_xml = self.parse_xml(dir_with_bands, parser_options)
        parsed_stdout, logs_stdout = self.parse_stdout(
            parameters, parser_options, parsed_xml
        )
        if (
            "verbose" in environ_parameters["ENVIRON"]
            and environ_parameters["ENVIRON"]["verbose"] > 0
        ):
            parsed_debug, logs_debug = self.parse_debug(parser_options)
            parsed_stdout.update(parsed_debug)

        parsed_bands = parsed_stdout.pop("bands", {})
        parsed_structure = parsed_stdout.pop("structure", {})
        parsed_trajectory = parsed_stdout.pop("trajectory", {})
        parsed_parameters = self.build_output_parameters(parsed_stdout, parsed_xml)

        # Append the last frame of some of the smaller trajectory arrays to the parameters for easy querying
        self.final_trajectory_frame_to_parameters(parsed_parameters, parsed_trajectory)

        # If the parser option 'all_symmetries' is False, we reduce the raw parsed symmetries to save space
        all_symmetries = (
            False
            if parser_options is None
            else parser_options.get("all_symmetries", False)
        )
        if not all_symmetries and "cell" in parsed_structure:
            reduce_symmetries(parsed_parameters, parsed_structure, self.logger)

        structure = self.build_output_structure(parsed_structure)
        kpoints = self.build_output_kpoints(parsed_parameters, structure)
        bands = self.build_output_bands(parsed_bands, kpoints)
        trajectory = self.build_output_trajectory(parsed_trajectory, structure)

        # Determine whether the input kpoints were defined as a mesh or as an explicit list
        try:
            self.node.inputs.kpoints.get_kpoints()
        except AttributeError:
            input_kpoints_explicit = False
        else:
            input_kpoints_explicit = True

        # Only attach the `KpointsData` as output if there will be no `BandsData` output and inputs were defined as mesh
        if kpoints and not bands and not input_kpoints_explicit:
            self.out("output_kpoints", kpoints)

        if bands:
            self.out("output_band", bands)

        if trajectory:
            self.out("output_trajectory", trajectory)

        if not structure.is_stored:
            self.out("output_structure", structure)

        # Separate the atomic_occupations dictionary in its own node if it is present
        atomic_occupations = parsed_parameters.pop("atomic_occupations", None)
        if atomic_occupations:
            self.out("output_atomic_occupations", orm.Dict(dict=atomic_occupations))

        self.out("output_parameters", orm.Dict(dict=parsed_parameters))

        # Emit the logs returned by the XML and stdout parsing through the logger
        # If the calculation was an initialization run, reset the XML logs because they will contain a lot of verbose
        # warnings from the schema parser about incomplete data, but that is to be expected in an initialization run.
        if settings.get("ONLY_INITIALIZATION", False):
            logs_xml.pop("error")

        ignore = [
            "Error while parsing ethr.",
            "DEPRECATED: symmetry with ibrav=0, use correct ibrav instead",
        ]
        self.emit_logs([logs_stdout, logs_xml], ignore=ignore)

        # First check for specific known problems that can cause a pre-mature termination of the calculation
        exit_code = self.validate_premature_exit(logs_stdout)
        if exit_code:
            return self.exit(exit_code)

        # If the both stdout and xml exit codes are set, there was a basic problem with both output files and there
        # is no need to investigate any further.
        if self.exit_code_stdout and self.exit_code_xml:
            return self.exit(self.exit_codes.ERROR_OUTPUT_FILES)

        if self.exit_code_stdout:
            return self.exit(self.exit_code_stdout)

        if self.exit_code_xml:
            return self.exit(self.exit_code_xml)

        # First determine issues that can occurr for all calculation types. Note that the generic errors, that are
        # common to all types are done first. If a problem is found there, we return the exit code and don't continue
        for validator in [
            self.validate_electronic,
            self.validate_dynamics,
            self.validate_ionic,
        ]:
            exit_code = validator(trajectory, parsed_parameters, logs_stdout)
            if exit_code:
                return self.exit(exit_code)

    def parse_stdout(self, parameters, parser_options=None, parsed_xml=None):
        """Parse the stdout output file.

        :param parameters: the input parameters dictionary
        :param parser_options: optional dictionary with parser options
        :param parsed_xml: the raw parsed data from the XML output
        :return: tuple of two dictionaries, first with raw parsed data and second with log messages
        """
        from aiida_environ.parsers.parse_raw.pw import parse_stdout

        logs = get_logging_container()
        parsed_data = {}

        filename_stdout = self.node.base.attributes.get("output_filename")

        if filename_stdout not in self.retrieved.list_object_names():
            self.exit_code_stdout = self.exit_codes.ERROR_OUTPUT_STDOUT_MISSING
            return parsed_data, logs

        try:
            stdout = self.retrieved.get_object_content(filename_stdout)
        except IOError:
            self.exit_code_stdout = self.exit_codes.ERROR_OUTPUT_STDOUT_READ
            return parsed_data, logs

        try:
            parsed_data, logs = parse_stdout(
                stdout, parameters, parser_options, parsed_xml
            )
        except Exception:
            logs.critical.append(traceback.format_exc())
            self.exit_code_stdout = self.exit_codes.ERROR_UNEXPECTED_PARSER_EXCEPTION

        # If the stdout was incomplete, most likely the job was interrupted before it could cleanly finish, so the
        # output files are most likely corrupt and cannot be restarted from
        if "ERROR_OUTPUT_STDOUT_INCOMPLETE" in logs["error"]:
            self.exit_code_stdout = self.exit_codes.ERROR_OUTPUT_STDOUT_INCOMPLETE

        # Under certain conditions, such as the XML missing or being incorrect, the structure data might be incomplete.
        # Since following code depends on it, we replace missing information taken from the input structure.
        structure = self.node.inputs.structure
        parsed_data.setdefault("structure", {}).setdefault("cell", {})

        if "lattice_vectors" not in parsed_data["structure"]["cell"]:
            parsed_data["structure"]["cell"]["lattice_vectors"] = structure.cell

        if "atoms" not in parsed_data["structure"]["cell"]:
            symbols = {
                s.kind_name: structure.get_kind(s.kind_name).symbol
                for s in structure.sites
            }
            parsed_data["structure"]["cell"]["atoms"] = [
                (symbols[s.kind_name], s.position) for s in structure.sites
            ]

        return parsed_data, logs

    def parse_debug(self, parser_options=None):
        """Parse the stdout output file.

        :param parameters: the input parameters dictionary
        :param parser_options: optional dictionary with parser options
        :param parsed_xml: the raw parsed data from the XML output
        :return: tuple of two dictionaries, first with raw parsed data and second with log messages
        """
        from aiida_environ.parsers.parse_raw.pw import parse_debug

        logs = get_logging_container()
        parsed_data = {}

        debug_filename = self.node.base.attributes.get("debug_filename")

        try:
            stdout = self.retrieved.get_object_content(debug_filename)
        except IOError:
            self.exit_code_stdout = self.exit_codes.ERROR_OUTPUT_STDOUT_READ
            return parsed_data, logs

        try:
            parsed_data, logs = parse_debug(stdout, parser_options)
        except Exception:
            logs.critical.append(traceback.format_exc())
            self.exit_code_stdout = self.exit_codes.ERROR_UNEXPECTED_PARSER_EXCEPTION

        # If the stdout was incomplete, most likely the job was interrupted before it could cleanly finish, so the
        # output files are most likely corrupt and cannot be restarted from
        if "ERROR_OUTPUT_STDOUT_INCOMPLETE" in logs["error"]:
            self.exit_code_stdout = self.exit_codes.ERROR_OUTPUT_STDOUT_INCOMPLETE

        return parsed_data, logs

    @staticmethod
    def final_trajectory_frame_to_parameters(parameters, parsed_trajectory):
        """Copy the last frame of certain properties from the `TrajectoryData` to the outputs parameters.

        Environ has extra energy terms, so extract these too

        This makes these properties queryable.
        """
        include_keys = [
            "energy",
            "energy_accuracy",
            "energy_ewald",
            "energy_hartree",
            "energy_hubbard",
            "energy_one_electron",
            "energy_threshold",
            "energy_vdw",
            "energy_xc",
            "energy_smearing",
            "energy_one_center_paw",
            "energy_est_exchange",
            "energy_fock",
            "energy_embedding",
            "energy_cavitation",
            "energy_pv",
            "energy_confine",
            "energy_electrolyte",
            "energy_one_electron_environ",
            "scf_iterations",
            "fermi_energy",
            "fermi_energy_correction",
            "total_force",
            "total_magnetization",
            "absolute_magnetization",
        ]

        for property_key, property_values in parsed_trajectory.items():

            if property_key not in include_keys:
                continue

            parameters[property_key] = property_values[-1]
