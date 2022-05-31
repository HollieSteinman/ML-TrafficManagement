import sys, os
import sumolib
import traci

# check for SUMO_HOME env
if 'SUMO_HOME' in os.environ:
    tools = os.path.join(os.environ['SUMO_HOME'], 'tools')
    sys.path.append(tools)
else:
    sys.exit("Please ensure 'SUMO_HOME' environment variable is set")

class TrafficLight:
    """
    Represents a SUMO Traffic Light

    :param id: (int) The Traffic Light's ID
    :param action_dur: (int) The duration between actions
    :param yellow_dur: (int) The duration of a yellow phase
    :param green_dur: (int) The minimum duration of a green phase
    :param sumo: (traci) The TraCi SUMO instance
    """
    def __init__(
            self,
            id: int,
            action_dur: int,
            yellow_dur: int,
            green_dur: int,
            sumo: traci
        ):
        self.id = id
        self.sumo = sumo # type: traci

        self.action_dur = action_dur
        self.yellow_dur = yellow_dur
        self.green_dur = green_dur

        self.current_phase = 0
        self.set_phases()
        self.current_phase_dur = 0

    def set_phases(self):
        """
        Sets the Traffic Light's phases and updates SUMO's phase durations
        """
        # get all phases
        phases = self.sumo.trafficlight.getAllProgramLogics(self.id)[0].phases

        self.green_phases = []
        self.transitions = {}
        self.all_phases = []
        for i, phase in enumerate(phases):
            # retrieve state - 'GGrrrr'
            state = phase.state
            # if no yellow, green light & all lights are not red, add to green phase
            if 'y' not in state and 'G' in state and state.count('r') != len(state):
                # add green phase
                green_phase = self.sumo.trafficlight.Phase(self.green_dur, state)
                self.green_phases.append(green_phase)
                self.all_phases.append(green_phase)

                # add yellow phase
                yellow_phase = self.sumo.trafficlight.Phase(self.yellow_dur, phases[i + 1].state)
                self.all_phases.append(yellow_phase)

                # yellow is next phase, green is phase after
                # append transition from current green to next green
                self.transitions[(len(self.green_phases) - 1)] = i + 1
        
        # update phases in SUMO
        sumo_logic = self.sumo.trafficlight.getAllProgramLogics(self.id)[0]
        sumo_logic.type = 0
        sumo_logic.phases = self.all_phases

        self.sumo.trafficlight.setProgramLogic(self.id, sumo_logic)
        self.sumo.trafficlight.setRedYellowGreenState(self.id, self.all_phases[self.current_phase].state)

    def set_next_phase(self, next_phase):
        """
        Sets the next green phase.
        If the current green phase has not completed, no new phase is set.
        """
        assert next_phase < len(self.green_phases)

        # if phase is same as current phase
        # or current phase duration is less than yellow duration + green duration
        if self.current_phase == next_phase or self.current_phase_dur < self.yellow_dur + self.green_dur:
            self.sumo.trafficlight.setRedYellowGreenState(self.id, self.green_phases[self.current_phase].state)
        else:
            state = self.all_phases[self.transitions[(self.current_phase)]].state
            self.sumo.trafficlight.setRedYellowGreenState(self.id, state)
            self.current_phase = next_phase
            self.current_phase_dur = 0
        # next action time?

    def simulate_step(self):
        """
        Simulates a step for the Traffic Light.
        If the current phase is yellow and has completed, the current phase is set to the current green phase.
        """
        self.current_phase_dur += 1
        if self.is_yellow() and self.current_phase_dur >= self.yellow_dur:
            self.sumo.trafficlight.setRedYellowGreenState(self.id, self.green_phases[self.current_phase])

    def can_set(self):
        """
        Determines if the next action can be taken.
        """
        return self.current_phase_dur >= self.action_dur

    def is_yellow(self):
        """
        Determines if the current phase is yellow.
        """
        if 'y' in self.sumo.trafficlight.getRedYellowGreenState(self.id):
            return True
        else:
            return False
    


