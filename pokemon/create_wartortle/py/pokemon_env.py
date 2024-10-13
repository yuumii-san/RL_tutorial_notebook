import sys
import numpy as np
import math
import matplotlib.pyplot as plt
from pyboy import PyBoy

from gymnasium import Env, spaces
from pyboy.utils import WindowEvent
from memory_addresses import *
from skimage.transform import resize
from IPython.display import clear_output
import cv2
import copy

# Memory address
MAP_ID, X_POS_ADDRESS, Y_POS_ADDRESS = 0xD35E, 0xD362, 0xD361
LEVELS_ADDRESSES = [0xD18C, 0xD1B8, 0xD1E4, 0xD210, 0xD23C, 0xD268]
IS_IN_BATTLE_ADDRESS = 0xD057
HP_ADDRESSES = [0xD16D, 0xD199, 0xD1C5, 0xD1F1, 0xD21D, 0xD249] # only check higher byte value. When HP becomes 3 degit number, need to add lower byte value. 
WILD_POKEMON_HP = 0xCFE7  # ref: https://archives.glitchcity.info/forums/board-76/thread-4200/page-0.html
wTileInFrontOfPlayer = 0xCFC6

class RedGymEnv(Env):
    def __init__(self, config):
        super(RedGymEnv, self).__init__() # should I do this?
        # Define action psace
        self.valid_actions = [
            WindowEvent.PRESS_ARROW_DOWN,
            WindowEvent.PRESS_ARROW_LEFT,
            WindowEvent.PRESS_ARROW_RIGHT,
            WindowEvent.PRESS_ARROW_UP,
            WindowEvent.PRESS_BUTTON_A,
            WindowEvent.PRESS_BUTTON_B,
        ]
        self.action_space = spaces.Discrete(len(self.valid_actions))
        
        # self.valid_actions.extend([
        #     WindowEvent.PRESS_BUTTON_START,
        #     WindowEvent.PASS
        # ])

        self.release_arrow = [
            WindowEvent.RELEASE_ARROW_DOWN,
            WindowEvent.RELEASE_ARROW_LEFT,
            WindowEvent.RELEASE_ARROW_RIGHT,
            WindowEvent.RELEASE_ARROW_UP
        ]

        self.release_button = [
            WindowEvent.RELEASE_BUTTON_A,
            WindowEvent.RELEASE_BUTTON_B
        ]
        
        # Define observation space
        self.output_shape = (144, 160, 1)
        self.output_full_shape = (144, 160, 3) # 3: RGB
        self.observation_space = spaces.Box(low=0, high=255, shape=self.output_full_shape, dtype=np.uint8)

        # Define action frequency
        self.act_freq = config['env']['action_freq']

        # Create pyboy object
        head = 'SDL2'
        self.pyboy = PyBoy(
                config['path']['gb_path'],
                # debugging=False,
                # disable_input=False,
                # window_type=head, # old version of pyboy
                window=head # new version
                # hide_window='--quiet' in sys.argv,
            )

        # Initialize the state
        self.init_state = config['path']['init_state']
        with open(self.init_state, "rb") as f:
            self.pyboy.load_state(f)

        # Initialize a generator of a game image
        # self.screen = self.pyboy.botsupport_manager().screen() # for old pyboy version
        self.screen = self.pyboy.screen

        # Initailize variables to monitor agent's state and reward        
        self.agent_stats = []
        self.total_reward = 0
        self.prev_in_battle = 0  # Keeps track of the previous battle type
        self.encounter_wild_pokemon = 0 # 1 if agent encounters a wild pokemon by taking the current action
        self.win_battle = 0
        self.lose_battle = 0
        self.in_grass = 0
        self.visited_states = []
        self.novel_state = 0
        self.low_hp = 0
        self.last_state = ''

        # Define rewards use
        # self.reward_type = config['reward']['reward_type']
        self.use_level_reward = config['reward_type']['use_level_reward']
        self.use_encounter_reward = config['reward_type']['use_encounter_reward']
        self.use_win_battle_reward = config['reward_type']['use_win_battle_reward']
        self.use_in_grass_reward = config['reward_type']['use_in_grass_reward']
        self.use_novel_state_reward = config['reward_type']['use_novel_state_reward']
        self.use_heal_reward = config['reward_type']['use_heal_reward']
        self.use_move_to_different_loc = config['reward_type']['use_move_to_different_loc']
        
        # Define reward value base
        self.level_reward_base = config['reward_base']['level_reward_base']
        self.encounter_reward_base = config['reward_base']['encounter_reward_base']
        self.win_battle_reward_base = config['reward_base']['win_battle_reward_base']
        self.in_grass_reward_base = config['reward_base']['in_grass_reward_base']
        self.novel_state_reward_base = config['reward_base']['novel_state_reward_base']
        self.heal_reward_base = config['reward_base']['heal_reward_base']
        self.move_to_different_loc_base = config['reward_base']['move_to_different_loc_base']
        
        # Define state type
        self.state_type = config['env']['state_type'] # {'map', 'image'}
        
        # Define termination condition
        self.termination_cond = config['env']['termination_cond']
        
        # Define coordinate of Red mother's location
        self.target_coord_door = [0,5,5]
        self.target_coord_mother = [37,5,6]
        
    def render(self):
        # game_pixels_render = self.screen.screen_ndarray() # (144, 160, 3) # for old pyboy version
        game_pixels_render = self.screen.ndarray
        return game_pixels_render


    def reset(self):
        # restart game, skipping credits
        print(self.init_state)
        with open(self.init_state, "rb") as f:
            self.pyboy.load_state(f)  
        
        # reset reward value
        self.total_reward = 0
        
        if self.state_type=='image':
            self.last_state = self.render()
            return self.last_state, {}
        elif self.state_type=='map':
            map_id, x_pos, y_pos = self.get_map_state()
            current_tile = self.pyboy.memory[wTileInFrontOfPlayer]
            low_hp = 0
            self.last_state = f"{map_id}_{x_pos}_{y_pos}_{low_hp}"
            # return f"{map_id}_{x_pos}_{y_pos}", {}
            # return f"{map_id}_{x_pos}_{y_pos}_{current_tile}_{low_hp}", {}
            return f"{map_id}_{x_pos}_{y_pos}_{low_hp}", {}

    
    def step(self, action):
        # take an aciton        
        # press button
        original_action = action
        if self.pyboy.memory[IS_IN_BATTLE_ADDRESS]==1: # in battle
        # if self.prev_in_battle==1: # in battle # with this, Red got stuck
            action = 4 # during a battle, only use attack
            
        map_id, x_pos, y_pos = self.get_map_state()
        if map_id==37 and x_pos==5 and y_pos==6 and self.low_hp==1:
            action = 4 # at mon's location, only use A
            
        self.pyboy.send_input(self.valid_actions[action])
        for i in range(self.act_freq):
            # release action not to keep taking the action
            if i == 8:
                if action < 4:
                    # release arrow
                    self.pyboy.send_input(self.release_arrow[action])
                if action > 3 and action < 6:
                    # release button 
                    self.pyboy.send_input(self.release_button[action - 4])
                if self.valid_actions[action] == WindowEvent.PRESS_BUTTON_START:
                    self.pyboy.send_input(WindowEvent.RELEASE_BUTTON_START)
                    
            # render pyBoy image at the last frame of each block
            if i == self.act_freq-1:
                # self.pyboy.__rendering(True)
                self.pyboy.tick(1, True)
            else:
                self.pyboy.tick(1, False)

        # store the new agent state obtained from the corresponding memory address
        # memory addresses from https://datacrystal.romhacking.net/wiki/Pok%C3%A9mon_Red/Blue:RAM_map
        map_id, x_pos, y_pos = self.get_map_state()
        # map_id = self.pyboy.get_memory_value(MAP_ID)
        # x_pos = self.pyboy.get_memory_value(X_POS_ADDRESS)
        # y_pos = self.pyboy.get_memory_value(Y_POS_ADDRESS)
        levels = [self.pyboy.memory[a] for a in LEVELS_ADDRESSES]
        in_battle = self.pyboy.memory[IS_IN_BATTLE_ADDRESS]
        pokemon_hps = [self.pyboy.memory[a] for a in HP_ADDRESSES]
        wild_pokemon_hp = self.pyboy.memory[WILD_POKEMON_HP]
        current_tile = self.pyboy.memory[wTileInFrontOfPlayer]
        
        # Update the HP states
        total_hp = np.sum(pokemon_hps)
        all_fainted = all(hp == 0 for hp in pokemon_hps) # Used to check loss
            
        # Check if total hp is in low state
        if total_hp<10:
            self.low_hp = 1
        else:
            self.low_hp = 0
            
        # Check if the new state is a novel state or not
        if self.state_type=='image':
            # store the new screen image (i.e. new observation) and reward    
            obs_memory = self.render()
            new_state = obs_memory
        elif self.state_type=='map':
            new_state = f"{map_id}_{x_pos}_{y_pos}_{self.low_hp}"
        if new_state not in self.visited_states:
            self.novel_state = 1
            self.visited_states.append(new_state)
        else:
            self.novel_state = 0

        # Check if battle state has transitioned to a non-buttle state without fainting all pokemons (i.e. lose)
        if in_battle!=1 and self.prev_in_battle==1: # use "!=1" instead of "==0" because the value will be 255 when losing the battle
            if wild_pokemon_hp==0:
                self.win_battle = 1
                self.lose_battle = 0
            elif total_hp==0:
                self.win_battle = 0
                self.lose_battle = 1
        else:
            self.win_battle = 0
            self.lose_battle = 0
        
        # Check if battle state has transitioned to a wild Pokémon encounter
        if in_battle==1 and self.prev_in_battle==0:
            # Transitioning from no battle to wild Pokémon battle
            self.encounter_wild_pokemon = 5
        else:
            self.encounter_wild_pokemon = 0
            
        # Update the previous battle type for the next step
        self.prev_in_battle = in_battle
        
        # if current_tile==82 and in_battle!=1:
        if current_tile==82:
            self.in_grass = 1
        elif self.low_hp==1:
            self.in_grass = 0 # encourage to leave from grass when low-hp
        else:
            self.in_grass = 0
            
        # Check if Red moved to a different location from the last state
        if in_battle==1: # no penalty of staying at the same location during a battle
            self.diff_loc = 0
        elif self.in_grass!=1: # diff loc reward only in grass
            self.diff_loc = 0
        elif self.low_hp==1:
            self.diff_loc = 0 # prioritize the distance base reward when low-hp
        else:
            if self.last_state==new_state:
                self.diff_loc = -1
            else:
                self.diff_loc = 1
        self.last_state = new_state # update the last state          
                       
        self.agent_stats.append({
            'map_id': map_id, 'x': x_pos, 'y': y_pos, 'levels': np.sum(levels), 
            'wild_pokemon': self.encounter_wild_pokemon, 'loss': all_fainted,
            'total_hp': total_hp
        })

        # Compute reward value for each reward category
        level_reward, encounter_reward, battle_reward, grass_reward, novel_state_reward, heal_reward, diff_loc_reward = 0, 0, 0, 0, 0, 0, 0
        if self.use_level_reward:
            level_reward = np.sum(levels) * self.level_reward_base
        if self.use_encounter_reward:
            encounter_reward = self.encounter_reward_base * self.encounter_wild_pokemon
        if self.use_win_battle_reward:
            battle_reward = self.win_battle_reward_base * self.win_battle
        if self.use_in_grass_reward:
            grass_reward = self.in_grass_reward_base * self.in_grass
        if self.use_novel_state_reward:
            novel_state_reward = self.novel_state_reward_base * self.novel_state
        if self.use_heal_reward:
            heal_reward = self.heal_reward_base * self.get_distance_reward(map_id, x_pos, y_pos)
        if self.use_move_to_different_loc:
            diff_loc_reward = self.move_to_different_loc_base * self.diff_loc

        reward = 0 + level_reward + encounter_reward + battle_reward + grass_reward + novel_state_reward + heal_reward + diff_loc_reward
        
        if self.termination_cond==None:
            # for simplicity, don't handle terminate or truncated conditions here
            terminated = False
            truncated = False
        elif self.termination_cond=='battle_lose':
            if self.lose_battle==1:
                terminated = True
                truncated = True
            else:
                terminated = False
                truncated = False

        return new_state, reward, terminated, truncated, {}


    def close(self):
        self.pyboy.stop(save=False) # terminate pyboy session
        super().close() # call close function of parent's class


    def get_map_state(self):
        map_id = self.pyboy.memory[MAP_ID]
        x_pos = self.pyboy.memory[X_POS_ADDRESS]
        y_pos = self.pyboy.memory[Y_POS_ADDRESS]
        return map_id, x_pos, y_pos
    

    def manhattan_distance(self, x1, y1, x2, y2):
        # Function to calculate the Manhattan distance between two points
        return abs(x1 - x2) + abs(y1 - y2)


    def get_distance_reward(self, map_id, x, y): 
        if map_id==0 or map_id==12: # 0: pallet town, 12: route 1
            target_coord = self.get_coordinate(self.target_coord_door[0], self.target_coord_door[1], self.target_coord_door[2])
            base_reward = 15
        elif map_id==37: # 37: red house first floor
            target_coord = self.get_coordinate(self.target_coord_mother[0], self.target_coord_mother[1], self.target_coord_mother[2])
            base_reward = 25
        else:
            # TOOD: decide what should be the target coordinate for map_id!=0,12,37
            target_coord = self.get_coordinate(self.target_coord_mother[0], self.target_coord_mother[1], self.target_coord_mother[2])
            base_reward = 0
        curr_coord = self.get_coordinate(map_id, x, y)

        # Set a default reward
        reward = 0
        
        if self.low_hp==1:
            # If HP is low, compute reward based on proximity to the Pokémon Center
            distance = self.manhattan_distance(curr_coord[0], curr_coord[1], target_coord[0], target_coord[1])
            
            # Reward increases as agent gets closer to the Pokémon Center
            reward = base_reward - distance  # Higher reward as the distance decreases
        else:
            reward = 0
        return reward
    
    
    def get_coordinate(self, map_id, x, y):        
        map_offsets = {
            # https://bulbapedia.bulbagarden.net/wiki/List_of_locations_by_index_number_(Generation_I)
            0: np.array([0,0]), # pallet town
            1: np.array([-10, -72]), # viridian
            2: np.array([-10, -180]), # pewter
            12: np.array([0, -36]), # route 1
            13: np.array([0, -144]), # route 2
            37: np.array([-9, -2]), # red house first
            38: np.array([-9, 7]), # red house second
            39: np.array([9+12, -2]), # blues house
            40: np.array([21, 6]), # oaks lab
            51: np.array([-35, -137]) # viridian forest
        }
        offset = map_offsets[map_id]
        
        # Calculate coordinates for sprite placement
        coord = offset + np.array([x, y])
        
        return coord