import unittest

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


def _build_cruise_phase_info():
    phase_kwargs = {
        'external_subsystems': [CustomAeroBuilder()],
    }

    # Only solver mode needs a bounded throttle phase override here.
    if rc_prop.power_balance_mode == 'solver':
        phase_kwargs['throttle_enforcement'] = 'bounded'
        phase_kwargs['throttle_bounds'] = ((0.2, 0.9), 'unitless')

    return get_cruise_phase_info(**phase_kwargs)


class CruiseExample:
    def build_problem(self):
        prob = av.AviaryProblem(verbosity=0)
        prob.options['group_by_pre_opt_post'] = True

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

        prob.add_driver('IPOPT', use_coloring=False, max_iter= 3)
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
            

        return prob

    def _print_design_vars_with_units(self, prob):
        print('Design Vars With Units')
        dv_meta = prob.model.get_design_vars(recurse=True, use_prom_ivc=True)

        for name in sorted(dv_meta):
            units = dv_meta[name].get('units')
            if units is None:
                units = 'unitless'
                value = prob.get_val(name)
            else:
                value = prob.get_val(name, units=units)

            value_str = np.array2string(np.asarray(value), precision=8, separator=', ')
            print(f"{name} [{units}] = {value_str}")

    def _print_objectives_with_units(self, prob):
        print('Objectives With Units')
        obj_meta = prob.model.get_objectives(recurse=True, use_prom_ivc=True)

        for name in sorted(obj_meta):
            units = obj_meta[name].get('units')
            if units is None:
                units = 'unitless'
                value = prob.get_val(name)
            else:
                value = prob.get_val(name, units=units)

            value_str = np.array2string(np.asarray(value), precision=8, separator=', ')
            print(f"{name} [{units}] = {value_str}")

    def _print_constraints_with_units(self, prob):
        print('Nonlinear Constraints With Units')
        con_meta = prob.model.get_constraints(recurse=True, use_prom_ivc=True)

        for name in sorted(con_meta):
            units = con_meta[name].get('units')
            if units is None:
                units = 'unitless'
                value = prob.get_val(name)
            else:
                value = prob.get_val(name, units=units)

            value_str = np.array2string(np.asarray(value), precision=8, separator=', ')
            print(f"{name} [{units}] = {value_str}")

    def run(self):
        prob = self.build_problem()
        prob.run_aviary_problem(run_driver=False, suppress_solver_print=False, make_plots=False)
        prob.run_aviary_problem(run_driver=True, suppress_solver_print=False, make_plots=False)
        self._print_design_vars_with_units(prob)
        self._print_objectives_with_units(prob)
        self._print_constraints_with_units(prob)
        return prob


class MeanPowerComp(om.ExplicitComponent):
    # Simple smoother objective input: average cruise electric power over the full phase.
    def setup(self):
        self.add_input('p_cruise_kw', shape_by_conn=True, units='kW')
        self.add_output('p_avg_kw', val=1.0, units='kW')
        self.declare_partials('p_avg_kw', 'p_cruise_kw', method='fd')

    def compute(self, inputs, outputs):
        outputs['p_avg_kw'] = np.mean(inputs['p_cruise_kw'])


# class TestRCPropMission(unittest.TestCase):
#     def test_residual(self):
#         nn = 3

#         prob = om.Problem(reports=False)
#         options = AviaryValues()
#         options.set_val(Aircraft.Engine.NUM_ENGINES, 1)
#         prob.model.add_subsystem('rc_prop_group', RCPropMission(num_nodes=nn, aviary_options=options), promotes=['*'])

#         # Solve the implicit current balance with Newton for this residual test.
#         prob.model.nonlinear_solver = om.NewtonSolver(solve_subsystems=True)
#         prob.model.nonlinear_solver.options['maxiter'] = 30
#         prob.model.nonlinear_solver.options['err_on_non_converge'] = True
#         prob.model.nonlinear_solver.linesearch = om.BoundsEnforceLS()
#         prob.model.nonlinear_solver.linesearch.options['bound_enforcement'] = 'scalar'
#         prob.model.linear_solver = om.DirectSolver(assemble_jac=True)

#         prob.setup(force_alloc_complex=True)

#         prob.set_val(Aircraft.Battery.VOLTAGE, 22.2, units='V')
#         prob.set_val(Aircraft.Battery.RESISTANCE, 0.05, units='ohm')
#         prob.set_val(Dynamic.Vehicle.Propulsion.THROTTLE, np.full(nn, 0.8))
#         prob.set_val(Aircraft.Engine.Motor.IDLE_CURRENT, 0.91, units='A')
#         prob.set_val(Aircraft.Engine.Motor.MAX_CONT_CURRENT, 120, units='A')
#         prob.set_val(Aircraft.Engine.Motor.RESISTANCE, 0.032, units='ohm')
#         prob.set_val(Aircraft.Engine.Motor.KV, 420, units='rpm/V')
#         prob.set_val(Dynamic.Atmosphere.DENSITY, 1.225, units='kg/m**3')
#         prob.set_val(Aircraft.Engine.Propeller.DIAMETER, 20, units='inch')
#         prob.set_val(Aircraft.Engine.Propeller.PITCH, 10, units='inch')
#         prob.set_val(Dynamic.Mission.VELOCITY, 20, units='ft/s')

#         prob.run_model()

#         battery_power = prob.get_val('battery.power', units='W')
#         esc_power = prob.get_val('esc.power', units='W')
#         motor_power = prob.get_val('motor.power', units='W')

#     def test_premission_calcs(self):
#         prob = om.Problem(reports=False)
#         options = AviaryValues()
#         options.set_val(Aircraft.Engine.Motor.KV_EQ_SLOPE, 2105.53674)
#         options.set_val(Aircraft.Engine.Motor.KV_EQ_INT, -80.83469)

#         prob.model.add_subsystem('rc_calcs', RCPropPreMission(aviary_options=options), promotes=['*'])

#         prob.setup(force_alloc_complex=True)

#         prob.set_val(Aircraft.Battery.MASS, 0.707, units='kg')
#         prob.set_val(Aircraft.Battery.VOLTAGE, 22.2, units='V')
#         prob.set_val(Aircraft.Engine.Motor.IDLE_CURRENT, 0.91, units='A')
#         prob.set_val(Aircraft.Engine.Motor.MAX_CONT_CURRENT, 120, units='A')
#         prob.set_val(Aircraft.Engine.Motor.MASS, 0.288, units='kg')

#         prob.run_model()

#         kv = prob.get_val(Aircraft.Engine.Motor.KV, 'rpm/V')
#         resistance = prob.get_val(Aircraft.Engine.Motor.RESISTANCE, 'ohm')
#         energy = prob.get_val(Aircraft.Battery.ENERGY_CAPACITY, 'W*h')

#         kv_expected = 796.472285
#         resistance_expected = 0.05582266503
#         energy_expected = 109.11522
#         assert_near_equal(kv, kv_expected, tolerance=1e-9)
#         assert_near_equal(resistance, resistance_expected, tolerance=1e-9)
#         assert_near_equal(energy, energy_expected, tolerance=1e-9)
#         partial_data = prob.check_partials(out_stream=None, method='cs')
#         assert_check_partials(partial_data, atol=1e-12, rtol=1e-12)


# # NOTE: no @use_tempdirs here. DBFMassBuilder reads its airfoil CSV via a repo-root-
# # relative path (like the dbf_based_mass unit tests), so this must run from the repo root.
# class TestRCCruiseAttempt(unittest.TestCase):
#     def test_subsystems_in_cruise_attempt(self):
#         prob = CruiseExample().run()

#         endurance = prob.get_val('endurance_comp.endurance', units='h')[0]
#         gross_mass = prob.get_val('mission:gross_mass', units='kg')[0]
#         motor_mass = prob.get_val('aircraft:engine:motor:mass', units='kg')[0]
#         current_flow = None
#         if rc_prop.power_balance_mode == 'solver':
#             for name in (
#                 Dynamic.Vehicle.Propulsion.CURRENT,
#                 'traj.cruise.controls:current_flow',
#                 'traj.cruise.rhs_all.rc_electric.esc.current_out',
#             ):
#                 try:
#                     current_flow = prob.get_val(name, units='A')
#                     break
#                 except KeyError:
#                     continue
#         electric_power = prob.get_val('traj.cruise.timeseries.electric_power_in_total', units='W')
#         distance_resid = prob.get_val('cruise_distance_constraint.distance_resid', units='nmi')[0]

#         print('cruise.endurance =', float(endurance), 'h')
#         print('cruise.mission_gross_mass =', float(gross_mass), 'kg')
#         print('cruise.motor_mass =', float(motor_mass), 'kg')
#         print('cruise.distance_resid =', float(distance_resid), 'nmi')
#         if current_flow is not None:
#             print('cruise.current_flow =', np.array2string(current_flow, precision=6, separator=', '), 'A')
#         print('cruise.electric_power_in_total =', np.array2string(electric_power, precision=6, separator=', '), 'W')

        # TODO: turn these back into assert_near_equal checks once the example values are confirmed.
        # self.assertTrue(np.isfinite(endurance) and endurance > 0.0)
        # self.assertTrue(np.isfinite(gross_mass) and 2.0 <= gross_mass <= 20.0)
        # self.assertTrue(0.25 < motor_mass < 0.65, f'Motor mass {motor_mass} kg is outside expected bounds.')
        # self.assertFalse(np.isnan(current_flow).any(), 'Current control contains NaN values.')
        # distance_resid = abs(distance_resid)
        # self.assertTrue(np.isfinite(distance_resid))
        # self.assertLess(distance_resid, 0.25, 'Cruise distance residual is unexpectedly large.')


        fuselage_wetted_area = prob.get_val('aircraft:fuselage:wetted_area', units='m**2')
        print(fuselage_wetted_area)
if __name__ == '__main__':
    unittest.main()
