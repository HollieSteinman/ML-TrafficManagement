from typing import Tuple
from gym import Env, spaces
import os, sys
import numpy as np
import sumolib
import traci
import traci.constants as tc

from env.traffic_light import TrafficLight

TEMP_ROUTE = "tmp.rou.xml"
MAX_QUEUE = 50

# check for SUMO_HOME env
if 'SUMO_HOME' in os.environ:
    tools = os.path.join(os.environ['SUMO_HOME'], 'tools')
    sys.path.append(tools)
else:
    sys.exit("Please ensure 'SUMO_HOME' environment variable is set")

class TrafficIntersection(Env):
    """
    Represents a single SUMO Traffic Intersection environment.

    :param network: (str) The path to the network file
    :param route: (str) The path to the route file
    :param add: (str) The path to the additional file
    :param view: (Tuple[int,int]) The dimensions of the GUI view
    :param action_dur: (int) The duration between actions
    :param yellow_dur: (int) The duration of a yellow phase
    :param green_dur: (int) The minimum duration of a green phase
    :param delay: (int) The delay of the GUI
    :param gui: (bool) If true, use SUMO GUI
    """
    def __init__(
            self,
            network: str,
            route: str,
            add: str,
            view: Tuple[int,int]=[1000,1000],
            action_dur: int=5,
            yellow_dur: int=3,
            green_dur: int=5,
            delay: int=0,
            gui: bool=True
        ):
        # check for gui
        self.gui = gui
        if self.gui:
            self._sumo_bin = sumolib.checkBinary('sumo-gui')
        else:
            self._sumo_bin = sumolib.checkBinary('sumo')

        # set SUMO files
        self._network = network
        self._route = route
        self._add = add
        self._view = view
        self.delay = delay

        # set phase & action lengths
        assert action_dur > yellow_dur
        self.action_dur = action_dur
        self.yellow_dur = yellow_dur
        self.green_dur = green_dur

        self.sumo = None # type: traci
        self.sumo_step = 0

    
    def start_sumo(self):
        """
        Starts SUMO
        """
        # set default cmd args
        sumo_cmd = [self._sumo_bin, # sumo bin
            '-n', self._network, # network file
            '-r', self._route, # route file
            '-a', self._add, # additional file
            '--random',] # random seed

        # set gui specific args
        if self.gui:
            sumo_cmd.extend(['--start', '--quit-on-end'])
            if self._view is not None:
                sumo_cmd.extend([
                    '--window-size', f'{self._view[0]},{self._view[1]}', # window size
                    '-d', self.delay, # delay
                    ])

                
            
        # start sumo
        traci.start(sumo_cmd)
        self.sumo = traci
        self.sumo_step = 0

        self.sumo.gui.setZoom("View #0", 500)

        self.light = TrafficLight('0', self.action_dur, self.yellow_dur, self.green_dur, self.sumo)
        
        # set lanes
        all_lanes = self.sumo.lane.getIDList()
        self.lanes = []
        for lane in all_lanes:
            if ':' not in lane:
                self.lanes.append(lane)

        # get detector
        self.detector = self.sumo.multientryexit.getIDList()[0]

        # action space is all green phases
        self.action_space = spaces.Discrete(len(self.light.green_phases))

        # set observation space
        # 0-1 for [num green phases], [can change], [num lanes]
        self.observation_space = spaces.Box(
            low=np.zeros(len(self.light.green_phases) + 1 + len(self.lanes), dtype=np.float32),
            hight=np.ones(len(self.light.green_phases) + 1 + len(self.lanes), dtype=np.float32)
        )

        # adjust gui
        if self.gui:
            self.sumo.gui.setSchema(traci.gui.DEFAULT_VIEW, "standard")

        # step to start SUMO
        self.sumo.simulation.step()
    
    def stop_sumo(self):
        """
        Stops SUMO
        """
        if self.sumo is None:
            return
        
        traci.close()
        self.sumo = None

    def calculate_observation(self):
        """
        Calculates observation
        """
        # one-hot encode green phases
        phase = [1 if self.light.current_phase == i else 0 for i in range(len(self.light.green_phases))]
        # if minimum green phase time has elapsed, lights can be changed
        can_change = [0 if self.light.current_phase_dur > self.light.yellow_dur + self.light.green_dur else 1]
        # each lane's queue (max of MAX_QUEUE)
        queued = [min(1, self.sumo.lane.getLastStepHaltingNumber(l) / MAX_QUEUE) for l in self.lanes]

        return np.array(phase + can_change + queued)
        
    def calculate_reward(self):
        """
        Calculates reward for a step
        """
        # reward = mean speed * total vehicles
        mean_speed = self.sumo.multientryexit.getLastStepMeanSpeed(self.detector)
        total_vehicles = self.sumo.multientryexit.getLastStepVehicleNumber(self.detector)

        return max(mean_speed * total_vehicles, 0)

    def calculate_info(self, reward):
        """
        Calculates info for a step
        """
        return {
            'sumo_step': self.sumo_step,
            'reward': reward,
            'total_queued': sum(self.sumo.lane.getLastStepHaltingNumber(l) for l in self.lanes)
        }

    def reset(self):
        """
        Resets SUMO
        """
        self.stop_sumo()
        self.start_sumo()

    def render(self):
        return;

    def step_sumo(self):
        """
        Steps SUMO
        """
        self.sumo.simulation.step()
        self.sumo_step += 1
    
    def step(self, action):
        """
        Step

        :param action: (int) Step to take
        """
        # if action is undefined, step in SUMO
        if action is None:
            for _ in range(self.action_dur):
                self.step_sumo()
        else:
            assert action in self.action_space
            # if next light phase can be set
            if self.light.can_set():
                self.light.set_next_phase(action)
            
            # continue stepping until next phase
            self.step_sumo()
            self.light.simulate_step()
            while not self.light.can_set():
                self.step_sumo()
                self.light.simulate_step()
        
        # retrieve returns
        obsv = self.calculate_observation()
        reward = self.calculate_reward()
        done = self.sumo_step >= RUNTIME
        info = self.calculate_info(reward)

        # shutdown sumo if complete
        if done:
            self.stop_sumo()
        
        return obsv, reward, done, info



