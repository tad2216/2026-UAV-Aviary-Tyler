import numpy as np
import openmdao.api as om

from aviary.subsystems.propulsion.UAV.model.UAV_performance import \
    Throttle, Battery, ElectronicSpeedController, Motor, PropCoefficients, Propeller, Vectorization
from aviary.utils.aviary_values import AviaryValues
from aviary.variable_info.UAV_variables import Aircraft, Dynamic
from aviary.variable_info.UAV_variables import Aircraft, Dynamic


class UAVPropMission(om.Group):
    """Calculates the mission performance (ODE) of a single electric RCMotor."""

    def initialize(self):
        self.options.declare('num_nodes', types=int)
        self.options.declare(
            'aviary_options',
            types=AviaryValues,
            desc='collection of Aircraft/Mission specific options',
            default=None,
        )


    def setup(self):
        nn = self.options['num_nodes']



        # constraint ties the motor to the prop load; in solver mode the solver does.
        motor_load_factor = 1.0



        rpm_in = [(Dynamic.Vehicle.Propulsion.RPM, 'rev_per_sec_slack')]
        self.set_input_defaults('rev_per_sec_slack', val=np.ones(nn) * 60.0, units='rev/s')

        # Dymos chooses throttle; motor_prop_balance solves current and shaft RPM so the
        # motor's electrical prediction matches the direct-drive propeller load.
        # self.add_subsystem(
        #     'throttle',
        #     Throttle(num_nodes=nn),
        #     promotes_inputs=[
        #         Dynamic.Vehicle.Propulsion.THROTTLE,
        #         'current_slack',
        #     ],
        #     promotes_outputs=[Dynamic.Vehicle.Propulsion.CURRENT],
        # )


        self.add_subsystem(
            'esc_current',
            om.ExecComp(
                'battery_current = throttle * motor_current',
                battery_current={'val': np.zeros(nn), 'units': 'A'},
                throttle={'val': np.zeros(nn), 'units': 'unitless'},
                motor_current={'val': np.zeros(nn), 'units': 'A'},
                has_diag_partials=True,
            ),
            promotes_inputs=[
                ('throttle', Dynamic.Vehicle.Propulsion.THROTTLE),
                ('motor_current', Dynamic.Vehicle.Propulsion.CURRENT),
            ],
            promotes_outputs=['battery_current'],
        )


        self.add_subsystem(
            'battery',
            Battery(num_nodes=nn),
            promotes_inputs=[
                Aircraft.Battery.VOLTAGE,
                Aircraft.Battery.RESISTANCE,
                # Dynamic.Vehicle.Propulsion.CURRENT,
                (Dynamic.Vehicle.Propulsion.CURRENT, 'battery_current'),
            ]
        )

        self.add_subsystem(
            'esc',
            ElectronicSpeedController(num_nodes=nn),
            promotes_inputs=[
                Dynamic.Vehicle.Propulsion.THROTTLE,
                Dynamic.Vehicle.Propulsion.CURRENT
            ],
        )

        self.add_subsystem(
            'motor',
            Motor(num_nodes=nn, load_factor=motor_load_factor),
            promotes_inputs=[

                Aircraft.Engine.Motor.IDLE_CURRENT,
                Aircraft.Engine.Motor.RESISTANCE,
                Aircraft.Engine.Motor.KV,
                # Dynamic.Vehicle.Propulsion.CURRENT,
                ],
            promotes_outputs=[
                Dynamic.Vehicle.Propulsion.RPM,
                ]
        )


        self.add_subsystem('vectorize_geo', Vectorization(num_nodes=nn),
            promotes_inputs=[Aircraft.Engine.Propeller.DIAMETER, Aircraft.Engine.Propeller.PITCH],
            promotes_outputs=['temp_diameter', 'temp_pitch']
            )



        self.add_subsystem(
            'propco',
            PropCoefficients(method='lagrange2', extrapolate=True, training_data_gradients=True, vec_size=nn),
            promotes_inputs=[
                Dynamic.Mission.VELOCITY,
                'temp_diameter',
                'temp_pitch',
            ] + rpm_in,
            promotes_outputs=['ct', 'cp']
        )


        self.add_subsystem(
            'prop',
            Propeller(num_nodes=nn),
            promotes_inputs=[
                Aircraft.Engine.Propeller.DIAMETER,
                'ct',
                'cp',

                Dynamic.Atmosphere.DENSITY
                ] + rpm_in,
            promotes_outputs=[
                Dynamic.Vehicle.Propulsion.PROP_POWER,
                Dynamic.Vehicle.Propulsion.THRUST,

                ]
        )


        self.add_subsystem(
            'rev_per_sec_balance',
            om.ExecComp(
                'rev_per_sec_defect = rev_per_sec_slack - rev_per_sec_motor',
                rev_per_sec_defect={'val': np.zeros(nn), 'units': 'rev/s'},
                rev_per_sec_slack={'val': np.zeros(nn), 'units': 'rev/s'},
                rev_per_sec_motor={'val': np.zeros(nn), 'units': 'rev/s'},
                has_diag_partials=True,
            ),
            promotes_inputs=['rev_per_sec_slack'],
        )
        self.connect(Dynamic.Vehicle.Propulsion.RPM, 'rev_per_sec_balance.rev_per_sec_motor')

        self.add_subsystem(
            'power_balance',
            om.ExecComp(
                'power_defect = shaft_power - prop_power',
                power_defect={'val': np.zeros(nn), 'units': 'W'},
                prop_power={'val': np.zeros(nn), 'units': 'W'},
                shaft_power={'val': np.zeros(nn), 'units': 'W'},
                has_diag_partials=True,
            ),
        )
        self.connect('motor.shaft_power', 'power_balance.shaft_power')
        self.connect(Dynamic.Vehicle.Propulsion.PROP_POWER, 'power_balance.prop_power')

        motor_prop_balance = om.BalanceComp()
        motor_prop_balance.add_balance(
            Dynamic.Vehicle.Propulsion.CURRENT,
            val=np.full(nn, 15.0),
            units='A',
            lower=0.0,
            upper=100.0,
            lhs_name='power_defect',
            rhs_val=np.zeros(nn),
            eq_units='W',
            normalize=False,
            res_ref=200.0,
        )
        motor_prop_balance.add_balance(
            'rev_per_sec_slack',
            val=np.full(nn, 88.0),
            units='rev/s',
            lower=3.3,
            upper=180.0,
            lhs_name='rev_per_sec_defect',
            rhs_val=np.zeros(nn),
            eq_units='rev/s',
            normalize=False,
            res_ref=1.0,
        )
        self.add_subsystem(
            'motor_prop_balance',
            motor_prop_balance,
            promotes_outputs=[Dynamic.Vehicle.Propulsion.CURRENT, 'rev_per_sec_slack'],
        )
        self.connect('power_balance.power_defect', 'motor_prop_balance.power_defect')
        self.connect(
            'rev_per_sec_balance.rev_per_sec_defect',
            'motor_prop_balance.rev_per_sec_defect',
        )







        """This is dt_soc"""
        self.add_subsystem(
            'electric_power',
            om.ExecComp(
                [
                    'p_elec = v_batt * current',

                ],
                p_elec={'val': np.zeros(nn), 'units': 'W'},
                v_batt={'val': np.zeros(nn), 'units': 'V'},
                current={'val': np.zeros(nn), 'units': 'A'},
                has_diag_partials=True,
            ),
            promotes_inputs=[
                # ('current', Dynamic.Vehicle.Propulsion.CURRENT),
                ('current', 'battery_current'),
            ],
            promotes_outputs=[
                ('p_elec', Dynamic.Vehicle.Propulsion.ELECTRIC_POWER_IN),

            ],
        )


        self.add_subsystem(
            'energy_con',
            om.ExecComp(
                'energy_constraint = energy_capacity-energy_used',
                energy_constraint={'val':np.zeros(nn), 'units': 'W*h'},
                energy_capacity={'val':1.0 , 'units': 'W*h'},
                energy_used={'val':np.zeros(nn), 'units': 'W*h'},
                has_diag_partials=True,
            ),

            promotes_inputs=[
                ('energy_capacity', Aircraft.Battery.ENERGY_CAPACITY),
                'energy_used'
                ],
            promotes_outputs=[
                'energy_constraint'
                ],
        )



        self.connect('battery.voltage_out', 'electric_power.v_batt')
        self.connect('battery.voltage_out', 'esc.voltage_in')
        self.connect('esc.voltage_out', 'motor.voltage_in')
        self.connect('esc.current_out', 'motor.current')

        """Constraints"""
          # These defects are now driven to zero by motor_prop_balance.
          # self.add_constraint(
          #     'rev_per_sec_balance.rev_per_sec_defect',
          #     upper=1.0,
          #     lower=-1.0,
          #     ref=1.0,
          #     units='rev/s',
          # )


        """for min_energy_example this should be commented out, but for cruise example it should be active"""
        # self.add_constraint('energy_constraint', lower=0.0, indices=[-1], ref=100, units='W*h')
        # self.add_constraint(
        #     'power_balance.power_defect',
        #     lower=-5.0,
        #     upper=5.0,
        #     ref=200.0,
        #     units='W',
        # )



        self.options['auto_order'] = True
