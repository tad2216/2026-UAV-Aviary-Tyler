
import aviary.api as av
import numpy as np
import openmdao.api as om
from openmdao.utils.assert_utils import assert_check_partials, assert_near_equal

from aviary.subsystems.aerodynamics.UAV_Aero.custom_aero_builder import CustomAeroBuilder
from aviary.subsystems.mass.UAV_mass.mass_builder import MassBuilder as DBFMassBuilder
from aviary.subsystems.mass.UAV_mass.variable_info.mass_variables import Aircraft as DBFAircraft
from aviary.models.aircraft.small_uav.phases.UAV_energy_phase import get_cruise_phase_info
from aviary.subsystems.propulsion.rc_electric.UAV_Builder import RCBuilder
from aviary.subsystems.propulsion.rc_electric.model.UAV_mission import RCPropMission
from aviary.subsystems.propulsion.rc_electric.model.UAV_premission import RCPropPreMission
from aviary.utils.aviary_values import AviaryValues
from aviary.variable_info.dbf_variables import Aircraft, Dynamic
from aviary.variable_info.variables import Mission

#This is where you set the power balance mode for the RCPropMission. Options are 'feedforward' or 'solver'.
#Example for solver, RCBBuilder(power_balance_mode='solver')
rc_prop = RCBuilder(power_balance_mode='solver')


prob = av.AviaryProblem(verbosity=0)
prob.options['group_by_pre_opt_post'] = True


def _build_cruise_phase_info():
    phase_kwargs = {
        'external_subsystems': [CustomAeroBuilder()],
    }

    # Only solver mode needs a bounded throttle phase override here.
    if rc_prop.power_balance_mode == 'solver':
        phase_kwargs['throttle_enforcement'] = 'bounded'
        phase_kwargs['throttle_bounds'] = ((0.2, 0.9), 'unitless')

    return get_cruise_phase_info(**phase_kwargs)

prob.load_inputs(
    'validation_cases/validation_data/test_models/small_scale_uav.csv',
    _build_cruise_phase_info(),
)
prob.load_external_subsystems(external_subsystems=[rc_prop, CustomAeroBuilder(), DBFMassBuilder()])
prob.aviary_inputs.set_val(Aircraft.Engine.Propeller.PITCH, 12.0, units='inch')

prob.check_and_preprocess_inputs()

for _k in ('mission:design:gross_mass', 'aircraft:design:gross_mass', 'mission:gross_mass'):
    try:
        print('Loaded gross mass key:', _k)
        print('Loaded gross mass kg:', prob.aviary_inputs.get_val(_k, units='kg'))
        print('Loaded gross mass lbm:', prob.aviary_inputs.get_val(_k, units='lbm'))
        break
    except KeyError:
        continue

prob.add_pre_mission_systems()

prob.add_phases()
prob.add_post_mission_systems()
prob.link_phases()

prob.add_driver('IPOPT', use_coloring=False, max_iter=3)
prob.driver.opt_settings['print_level'] = 5
prob.driver.opt_settings['mu_strategy'] = 'adaptive'
prob.driver.opt_settings['tol'] = 1e-6
prob.driver.opt_settings['acceptable_tol'] = 5e-7
prob.driver.opt_settings['acceptable_iter'] = 0
prob.driver.opt_settings['constr_viol_tol'] = 1e-7
prob.driver.opt_settings['acceptable_constr_viol_tol'] = 5e-7
prob.driver.options['debug_print'] = ['desvars', 'objs', 'nl_cons', 'ln_cons']

prob.add_design_variables()

# Aviary already added these with transport scale (ref=175e3 lbm, upper=None), and
# add_design_var won't overwrite. Drop the existing entries, then re-add at UAV scale.
del prob.model._static_design_vars[Aircraft.Design.GROSS_MASS]
del prob.model._static_design_vars['mission:gross_mass']
# prob.model.add_design_var(Aircraft.Design.GROSS_MASS, units='kg', lower=2.0, upper=20.0, ref=7.0)
prob.model.add_design_var('mission:gross_mass', units='kg', lower=2.0, upper=20.0, ref=7.0)

# Geometry design variables at UAV scale.
prob.model.add_design_var(Aircraft.Wing.ROOT_CHORD, units='m', lower=0.08, upper=1.2, ref=0.4)
prob.model.add_design_var(Aircraft.Wing.WETTED_AREA, units='m**2', lower=0.1, upper=2.0, ref=0.8)

prob.model.add_design_var(Aircraft.HorizontalTail.ROOT_CHORD, units='m', lower=0.08, upper=0.8, ref=0.3)
prob.model.add_design_var(Aircraft.HorizontalTail.SPAN, units='m', lower=0.2, upper=2.0, ref=0.8)
prob.model.add_design_var(Aircraft.HorizontalTail.WETTED_AREA, units='m**2', lower=0.05, upper=1.2, ref=0.3)

prob.model.add_design_var(Aircraft.VerticalTail.ROOT_CHORD, units='m', lower=0.08, upper=0.8, ref=0.3)
prob.model.add_design_var(Aircraft.VerticalTail.SPAN, units='m', lower=0.2, upper=2.0, ref=0.8)
prob.model.add_design_var(Aircraft.VerticalTail.WETTED_AREA, units='m**2', lower=0.03, upper=0.8, ref=0.12)

prob.model.add_design_var(Aircraft.Fuselage.MAX_HEIGHT, units='m', lower=0.04, upper=0.5, ref=0.15)
prob.model.add_design_var(Aircraft.Fuselage.MAX_WIDTH, units='m', lower=0.04, upper=0.5, ref=0.15)
prob.model.add_design_var(Aircraft.Fuselage.WETTED_AREA, units='m**2', lower=0.1, upper=2.0, ref=0.6)

# Keep direct geometric bounds.

class MeanPowerComp(om.ExplicitComponent):
    # Simple smoother objective input: average cruise electric power over the full phase.
    def setup(self):
        self.add_input('p_cruise_kw', shape_by_conn=True, units='kW')
        self.add_output('p_avg_kw', val=1.0, units='kW')
        self.declare_partials('p_avg_kw', 'p_cruise_kw', method='fd')

    def compute(self, inputs, outputs):
        outputs['p_avg_kw'] = np.mean(inputs['p_cruise_kw'])


prob.model.add_subsystem('mean_power_comp', MeanPowerComp())
prob.model.add_subsystem(
    'endurance_comp',
    om.ExecComp(
        'endurance = energy / (1000.0 * p_avg_kw + 1.0e-3)',
        endurance={'val': 1.0, 'units': 'h'},
        energy={'val': 1.0, 'units': 'W*h'},
        p_avg_kw={'val': 1.0, 'units': 'kW'},
    ),
)
prob.model.connect(Aircraft.Battery.ENERGY_CAPACITY, 'endurance_comp.energy')
prob.model.connect('traj.cruise.timeseries.electric_power_in_total', 'mean_power_comp.p_cruise_kw')
prob.model.connect('mean_power_comp.p_avg_kw', 'endurance_comp.p_avg_kw')
prob.model.add_objective('endurance_comp.endurance', ref=-1.0)


prob.model.set_input_defaults(Aircraft.Battery.VOLTAGE, val=22.2, units='V')
prob.model.set_input_defaults(Aircraft.Engine.Motor.IDLE_CURRENT, val=2.2, units='A')
prob.model.set_input_defaults(Aircraft.Engine.Motor.MAX_CONT_CURRENT, val=100.0, units='A')

prob.setup()

prob.set_solver_print(level=0)
prob.set_initial_guesses()
prob.set_val(Aircraft.Design.GROSS_MASS, 7.0, units='kg')
prob.set_val(Mission.GROSS_MASS, 7.0, units='kg')
prob.set_val(Aircraft.Engine.Motor.MASS, 0.55, units='kg')
prob.set_val(Aircraft.Engine.Motor.IDLE_CURRENT, 2.0, units='A')
prob.set_val(Aircraft.Battery.VOLTAGE, 25.2, units='V')

# Seed geometry terms near a small-UAV baseline.

prob.set_val(Aircraft.Fuselage.MAX_HEIGHT, 0.15, units='m')
prob.set_val(Aircraft.Fuselage.MAX_WIDTH, 0.15, units='m')
prob.set_val(Aircraft.Wing.WETTED_AREA, 0.85, units='m**2')
prob.set_val(Aircraft.HorizontalTail.SPAN, 1.0, units='m')
prob.set_val(Aircraft.HorizontalTail.WETTED_AREA, 0.35, units='m**2')
prob.set_val(Aircraft.VerticalTail.SPAN, 1.0, units='m')
prob.set_val(Aircraft.VerticalTail.WETTED_AREA, 0.14, units='m**2')
prob.set_val(Aircraft.Fuselage.WETTED_AREA, 0.58, units='m**2')

if rc_prop.power_balance_mode == 'feedforward':
    prob.set_val('traj.cruise.controls:throttle', 0.7, units='unitless')
    prob.set_val('traj.cruise.controls:current_flow', 40.0, units='A')
    prob.set_val('traj.cruise.controls:current_flow_max', 60.0, units='A')
    prob.set_val('traj.cruise.controls:rpm_lookup', 90.0, units='rev/s')
    prob.set_val('traj.cruise.controls:rpm_lookup_max', 122.0, units='rev/s')
    prob.set_val('traj.cruise.states:mass', 4.4, units='kg')
    

prob.run_model()

prob.run_aviary_problem(run_driver=True, suppress_solver_print=False, make_plots=False)


prob.model.pre_mission.list_vars(units=True, print_arrays=True)

fuselage_wetted_area = prob.get_val('aircraft:fuselage:wetted_area', units='m**2')
fuselage_wetted_area = prob.get_val('aircraft:fuselage:wetted_area', units='inch**2')
print(fuselage_wetted_area)
