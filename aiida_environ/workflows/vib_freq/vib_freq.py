
from aiida.engine import ToContext, WorkChain
from aiida.plugins import CalculationFactory, DataFactory
from aiida_environ.calculations.pw import EnvPwCalculation
from aiida.common.extendeddicts import AttributeDict
import aiida
from aiida import orm
import numpy as np

EnvCalculation = CalculationFactory('environ.pw')


class EnvVibfreqWorkchain(WorkChain):
    
    
    _RUN_PREFIX = 'scf_freq'
    @classmethod
    def define(cls, spec):
        super().define(spec)
        spec.input('structure', valid_type=orm.StructureData,
                   help='The inputs structure.')
        spec.input('clean_workdir', valid_type=orm.Bool, default=lambda: orm.Bool(False),
                   help='If `True`, work directories of all called calculation will be cleaned at the end of execution.')
        spec.expose_inputs(EnvPwCalculation, namespace='environ', exclude=('clean_workdir', 'parent_folder'))
        spec.input('step', valid_type=orm.Float,
                   help='atomic movement step.')
        spec.input('move_atoms', valid_type=orm.List,
                   help='atoms that can move.')
        spec.output_namespace(
            'supercells_forces', valid_type=(orm.ArrayData, orm.TrajectoryData), required=True,
            help='The forces acting on the atoms of each supercell.'
        )
        spec.output('eigen', valid_type=orm.List)
        spec.output('force_constant', valid_type=orm.List)
        spec.output('freq', valid_type=orm.List,help='Vibrational frequencies (cm-1).')
        
        
        spec.outline(
            cls.setup,
            cls.run_basic_scf,
            cls.inspect_basic_scf,
            cls.run_vib,
            cls.inspect_vib_calculations,
            cls.cal_freq,
        )
        spec.exit_code(401, 'ERROR_ENVIRON',
                       message='The environ  process failed')

    
    def setup(self):
    
        self.ctx.structure = self.inputs.structure
        
    def run_basic_scf(self):
        key = f'{self._RUN_PREFIX}_0'
        
        inputs = AttributeDict(self.exposed_inputs(EnvPwCalculation, namespace='environ'))
        inputs.structure = self.ctx.structure
        inputs.metadata.label = key
        inputs.metadata.call_link_label = key
        # inputs.parameters['CONTROL']['tprnfor']=True
        
        node = self.submit(EnvPwCalculation, **inputs)
        self.to_context(**{key: node})
        self.report(f'launching basic scf PwBaseWorkChain<{node.pk}>')
        
    def inspect_basic_scf(self):
        base_key = f'{self._RUN_PREFIX}_0'
        workchain = self.ctx[base_key]
        if not workchain.is_finished_ok:
            self.report(f'base basic scf failed')
            return self.exit_codes.ERROR_FAILED_BASE_SCF
    
    def run_vib(self):
        structures = self.generate_from_structure(self.inputs.structure, self.inputs.step, self.inputs.move_atoms.get_list())
        count = 0
        calc=[]
        base_key = f'{self._RUN_PREFIX}_0'
        base_out = self.ctx[base_key].outputs
        
        for i_structure in structures:
            
            inputs = AttributeDict(self.exposed_inputs(EnvPwCalculation, namespace='environ'))
            inputs.structure = i_structure
            
            key = f'{self._RUN_PREFIX}_{int((count)/12)}_{int((count)%12/4)}_{(count)%4}'
            inputs.metadata.label = key
            inputs.metadata.call_link_label = key
            inputs.parameters = self.inputs.environ.parameters.clone()
            # inputs.parameters['CONTROL']['tprnfor']=True
            inputs.parameters['ELECTRONS']['startingpot']='file'
            inputs.parameters['ELECTRONS']['startingwfc']='file'
            inputs.parent_folder = base_out.remote_folder
            inputs.environ_parameters=self.inputs.environ.environ_parameters.clone()
            inputs.environ_parameters['ENVIRON']['environ_restart']= True
            
            calc.append(self.submit(EnvPwCalculation, **inputs))
            count = count+1
            self.to_context(**{key: calc[count-1]})
            self.report(f'submitting `PhonopyCalculation{key}` <PK={calc[count-1].pk}>')
        
            
    def inspect_vib_calculations(self):
        failed_runs = []
        for label, workchain in self.ctx.items():
            
            if label.startswith(self._RUN_PREFIX):
                
                if workchain.is_finished_ok:
                    self.report(f'Finished <PK={workchain.pk}>')
                    forces = workchain.outputs.output_trajectory
                    self.out(f"supercells_forces.{label}", forces)
                else:
                    self.report(
                        f'PwBaseWorkChain with <PK={workchain.pk}> failed'
                        'with exit status {workchain.exit_status}'
                    )
                    failed_runs.append(workchain.pk)
        if failed_runs:
            self.report('one or more workchains did not finish succesfully')
            return self.exit_codes.ERROR_SUB_PROCESS_FAILED
        
        
    def cal_freq(self):
        forces_array = self.outputs['supercells_forces']
        force_0=forces_array['scf_freq_0'].get_array('forces')[0]
        bohr2ang = 0.529177249
        ryd2ev = 13.6057039763
        force_constant=[]
        count_i_atom = -1 
        for i_atom in self.inputs.move_atoms.get_list():
            count_i_atom = count_i_atom+1
            for i_dir in range(3):
                force_temp = []
                for i_move in range(5):
                    if (i_move==2):
                        force_temp.append(force_0)
                    else:
                        if (i_move<2):
                            temp_i_move=i_move
                        else:
                            temp_i_move=i_move-1
                        key = f'{self._RUN_PREFIX}_{count_i_atom}_{i_dir}_{temp_i_move}'
                        force_temp.append(forces_array[key].get_array('forces')[0])
                for key, info in aiida.common.constants.elements.items():
                    if info['symbol'] == self.inputs.structure.sites[i_atom].kind_name:
                        mass_temp = info['mass'] 
                        break
                force_constant_temp = self.cal_first_deriv(force_temp, mass_temp)
                force_constant.append(force_constant_temp)
        
        c1=[]
        for i in range(3*len(self.inputs.move_atoms)):
            temp_c1=[]
            for j in range(3*len(self.inputs.move_atoms)):
                temp_c1.append((force_constant[i][j]+force_constant[j][i])/2)
            c1.append(temp_c1)
        c=np.array(c1)
        
        temp_c = orm.List(c1).store()
        self.out('force_constant',temp_c)
        eigen= np.linalg.eig(c)[0]
        freq=np.sort(eigen)[3:]
        freq= pow(np.abs(freq),0.5)*521.47090038197
        temp_eig =  orm.List(eigen.tolist()).store()
        self.out('eigen',temp_eig)
        temp_freq = orm.List(freq.tolist()).store()
        self.out('freq',temp_freq)
        
        
    def generate_from_structure(self, structure, step, atoms):
        structure_list =[]
        for i_atom in range(len(atoms)):
            for i_dir in range(3):
                for i_move in range(5):
                    if (i_move==2):
                        continue
                    cell = []
                    for i in range(3):
                        cell.append([])
                        for j in range(3):
                            cell[i].append(structure.cell[i][j])
                    s = DataFactory('structure')(cell=cell)
                    for i in range(len(structure.sites)):
                        if i == atoms[i_atom]:
                            move_dir=[0,0,0]
                            move_dir[i_dir]=1
                            p=(structure.sites[i].position[0]+move_dir[0]*step.value*(i_move-2),
                               structure.sites[i].position[1]+move_dir[1]*step.value*(i_move-2), 
                               structure.sites[i].position[2]+move_dir[2]*step.value*(i_move-2))
                        else:
                            p=(structure.sites[i].position[0], structure.sites[i].position[1], structure.sites[i].position[2])

                        s.append_atom(position=p, symbols=structure.sites[i].kind_name)
                    structure_list.append(s)
        print(len(structure_list))
        return structure_list
    
    
    def cal_first_deriv(self, force, mass):
        force_constant=[]
        for i_atom in self.inputs.move_atoms.get_list(): 
            for key, info in aiida.common.constants.elements.items():
                if info['symbol'] == self.inputs.structure.sites[i_atom].kind_name:
                    mass_temp = info['mass'] 
                    break
            for i_dir in range(3):
                delta_f=[]
                delta_f.append((force[0][i_atom][i_dir]-force[2][i_atom][i_dir])/np.sqrt(mass_temp)/np.sqrt(mass))
                delta_f.append((force[1][i_atom][i_dir]-force[2][i_atom][i_dir])/np.sqrt(mass_temp)/np.sqrt(mass))
                delta_f.append((force[3][i_atom][i_dir]-force[2][i_atom][i_dir])/np.sqrt(mass_temp)/np.sqrt(mass))
                delta_f.append((force[4][i_atom][i_dir]-force[2][i_atom][i_dir])/np.sqrt(mass_temp)/np.sqrt(mass))
                force_constant.append(-(delta_f[0]-8*delta_f[1]+8*delta_f[2]-delta_f[3])/12.0/self.inputs.step.value)
        return force_constant


    
