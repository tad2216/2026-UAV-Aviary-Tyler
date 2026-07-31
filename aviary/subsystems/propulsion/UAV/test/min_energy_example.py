
from copy import deepcopy


import aviary.api as av
import numpy as np
import openmdao.api as om

from aviary.subsystems.aerodynamics.UAV_Aero.aero_builder import AeroBuilder
from aviary.subsystems.mass.UAV_mass.mass_builder import MassBuilder as DBFMassBuilder
from aviary.models.aircraft.small_uav.phases.UAV_energy_phase import phase_info
from aviary.subsystems.propulsion.UAV.UAV_Builder import UAVBuilder
from aviary.variable_info.UAV_variables import Aircraft
from aviary.variable_info.variables import  Settings


from aviary.variable_info.UAV_variable_meta_data import ExtendedMetaData


UAV_Prop = UAVBuilder()


def min_energy_example():
    prob = av.AviaryProblem(name='min_energy_example',verbosity=2, meta_data=ExtendedMetaData)
    prob.options['group_by_pre_opt_post'] = True
    # just selecting cruise
    cruise_phase_info = {
        'pre_mission': deepcopy(phase_info['pre_mission']),
        'cruise': {
        'subsystem_options': {'aerodynamics': {'method': 'external'}},
        'user_options': {
            'num_segments': 5,
            'order': 3,
            'mach_optimize': True,

            'mach_initial': (0.08, 'unitless'),

            'mach_bounds': ((0.05, 0.3), 'unitless'),
            # 'mach_ref': (0.05, 'unitless'),
            'mass_ref': (4.0, 'kg'),

            # 'alt_ref': (100, 'ft'),
            # 'mach_final': (0.05, 'unitless'),


            'altitude_optimize': True,
            'altitude_initial': (200.0, 'ft'),
            'altitude_bounds': ((50,400), 'ft'),
            'altitude_final': (200.0, 'ft'),
            'distance_initial': (0.0, 'm'),

            'distance_ref': (1000.0, 'm'),
            'target_distance': (1000.0, 'm'),
            'throttle_enforcement': 'control',

            # 'throttle_polynomial_order': 1,

            #Time
            'time_initial': (0.0, 's'),
            'time_duration_bounds': ((0,180), 's'),
        },
        'initial_guesses': {
            'distance': ([0, 2000], 'm'),
            'time': ([0, 60], 's'),
        },
    },
        'post_mission': deepcopy(phase_info['post_mission']),
    }

    prob.load_inputs(
        'validation_cases/validation_data/test_models/small_scale_uav.csv', cruise_phase_info
    )

    number = prob.aviary_inputs.get_val(Aircraft.Wing.WETTED_AREA, units='m**2')
    print('Wetted Area:', number)

    prob.load_external_subsystems(
        external_subsystems=[UAV_Prop, AeroBuilder(), DBFMassBuilder()]
    )

    prob.check_and_preprocess_inputs()

    prob.build_model()

    """Objective: Minimize energy consumption during cruise flight. This is done by adding an objective to the cruise phase that minimizes the energy constraint at the final time step. The energy constraint is defined as the integral of the power required to maintain level flight over the duration of the cruise phase. By minimizing this objective, we can find the optimal flight profile that minimizes energy consumption while still meeting all other constraints and requirements."""
    cruise_phase = prob.model.traj.phases.cruise
    cruise_phase.add_objective('energy_used', loc='final', ref = 10, units='W*h')
    cruise_phase.add_path_constraint('lift_coefficient', lower=0.0, upper=1.2)
    cruise_phase.add_path_constraint('thrust_net_total', lower=0.0, units='lbf')

    prob.add_driver('IPOPT', use_coloring=False, max_iter=2000)

    prob.driver.opt_settings['print_level'] = 5
    prob.driver.opt_settings['mu_strategy'] = 'adaptive'
    prob.driver.opt_settings['tol'] = 1e-5
    prob.driver.opt_settings['mu_init'] = 0.01
    prob.driver.opt_settings['limited_memory_max_history'] = 50
    prob.driver.opt_settings['acceptable_tol'] = 5e-5
    prob.driver.opt_settings['constr_viol_tol'] = 1e-5
    prob.driver.opt_settings['acceptable_constr_viol_tol'] = 5e-5
    # prob.driver.options['debug_print'] = ['desvars', 'objs', 'nl_cons', 'ln_cons']

    prob.add_design_variables()



    prob.setup()

    prob.set_solver_print(level=0)
    prob.set_initial_guesses()

    # prob.set_val('traj.cruise.states:mass', 4.1, units='kg')

    prob.set_val('traj.cruise.controls:rpm_slack', 1800.0, units='rpm')
    prob.set_val('traj.cruise.controls:throttle', 0.5)
    prob.set_val('traj.cruise.controls:alpha', 1, units='deg')

    number = prob.aviary_inputs.get_val(Aircraft.Wing.WETTED_AREA, units='m**2')
    print('Wetted Area:', number)

    prob.run_aviary_problem(run_driver=True)


    print('fuselage kg:', prob.get_val('aircraft:fuselage:mass', units='kg'))   # want ~1.8, NOT 6.6
    print('vtail kg:  ', prob.get_val('aircraft:vertical_tail:mass', units='kg'))  # want ~0.2
    print('gross lbm: ', prob.get_val('mission:gross_mass', units='lbm'))
    print(prob.get_val('traj.cruise.rhs_all.thrust_required', units='lbf'))
    print(prob.get_val('traj.cruise.rhs_all.thrust_residual', units='lbf'))
    print(prob.get_val('traj.cruise.rhs_all.drag', units='lbf'))
    print(prob.get_val('traj.cruise.rhs_all.thrust_net_total', units='lbf'))
    gross_mass = prob.get_val('mission:gross_mass', units='lbm')
    zero_fuel_mass = prob.get_val('mission:zero_fuel_mass', units='lbm')
    taxi_out_fuel = prob.get_val('mission:taxi:fuel_mass_taxi_out', units='lbm')
    takeoff_fuel = prob.get_val('mission:takeoff:fuel_mass', units='lbm')
    cons = prob.driver.get_constraint_values(driver_scaling=True)
    worst = sorted(((np.nanmax(np.abs(np.atleast_1d(v))), k) for k, v in cons.items()), reverse=True)[:5]
    [print(f'{m:10.3f}  {k}') for m, k in worst]

    print('gross_mass:', gross_mass)
    print('zero_fuel_mass:', zero_fuel_mass)
    print('gross_mass - zero_fuel_mass:', gross_mass - zero_fuel_mass)
    print('taxi_out_fuel:', taxi_out_fuel)
    print('takeoff_fuel:', takeoff_fuel)
    print('gross_mass - taxi_out_fuel - takeoff_fuel:', gross_mass - taxi_out_fuel - takeoff_fuel)

    print('settings:problem_type:', prob.aviary_inputs.get_val(Settings.PROBLEM_TYPE))
    print('settings:equations_of_motion:', prob.aviary_inputs.get_val(Settings.EQUATIONS_OF_MOTION))
    print('settings:mass_method:', prob.aviary_inputs.get_val(Settings.MASS_METHOD))
    return prob





if __name__ == '__main__':
    min_energy_example()
