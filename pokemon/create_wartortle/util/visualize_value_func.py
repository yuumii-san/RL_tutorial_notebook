import numpy as np
import pandas as pd
from pathlib import Path
import matplotlib.pyplot as plt
from matplotlib import transforms
from PIL import Image
from einops import rearrange
import requests
import io
import json
from tqdm import tqdm
import mediapy as media

            
def load_map(map_path):
    main_map = np.array(Image.open(map_path))
    return main_map

def load_sprite(sprite_path):
    sprite = Image.open(sprite_path)
    return sprite

def get_sprite_by_coords(img, x, y):
    sy = 34+17*y
    sx = 9 +17*x
    alpha_val = np.array([255, 127,  39, 255], dtype=np.uint8)
    sprite = img[sy:sy+16, sx:sx+16]
    return np.where((sprite == alpha_val).all(axis=2).reshape(16,16,1), np.array([[[0,0,0,0]]]), sprite).astype(np.uint8)


def load_tex(sprites, name):
    resp = requests.get(sprites[name])
    return np.array(Image.open(io.BytesIO(resp.content)))


def add_sprite(overlay_map, sprite, x, y, map_idx, opacity=1.0, add=True):
    global_offset = np.array([865, 3670]) #np.array([865, 3670]): [0,0] of mapid=1 is mapped to pallet town [0,0]
    map_offsets = {
        # https://bulbapedia.bulbagarden.net/wiki/List_of_locations_by_index_number_(Generation_I)
        0: np.array([0,0]), # pallet town (Done)
        1: np.array([-10, -72]), # viridian (Done)
        2: np.array([-10, -180]), # pewter (Done)
        12: np.array([0, -36]), # route 1 (Done)
        13: np.array([0, -144]), # route 2 (Done)
        37: np.array([-9, -2]), # red house first (Done)
        38: np.array([-9, 7]), # red house second (Done)
        39: np.array([9+12, -2]), # blues house (Done)
        40: np.array([21, 6]), # oaks lab (Done)
        51: np.array([-35, -137]) # viridian forest (Done)
    }
    if map_idx in map_offsets.keys():
        offset = map_offsets[map_idx]
    else:
        offset = np.array([0,0])
        x, y = 0, 0
    
    # Calculate coordinates for sprite placement
    coord = global_offset + 16*(offset + np.array([x, y]))
    
    # Extract the base where the sprite will be placed
    base = overlay_map[coord[1]:coord[1]+16, coord[0]:coord[0]+16, :].astype(np.float32)
    new = opacity * sprite.astype(np.float32)  # Apply opacity to the sprite
    
    if add:
        intermediate = base + new  # Add sprite to the base
    else:
        intermediate = new  # Replace base with new sprite
    
    # Apply the sprite onto the overlay map
    overlay_map[coord[1]:coord[1]+16, coord[0]:coord[0]+16, :] = intermediate.clip(0, 255).astype(np.uint8)
    
    return overlay_map

def blend_overlay(background, over):
    # Blend two images by alpha transparency (if available in the overlay)
    al = over[...,3].reshape(over.shape[0], over.shape[1], 1)  # Alpha channel
    ba = (255 - al) / 255  # Background alpha
    oa = al / 255  # Overlay alpha
    return (background[..., :3] * ba + over[..., :3] * oa).astype(np.uint8)


def change_arrow_color(arrow, new_color):
    """
    Changes the color of the transparent arrow.
    `new_color` should be in (R, G, B) format.
    """
    # Separate alpha channel from RGB
    r, g, b, a = arrow[:,:,0], arrow[:,:,1], arrow[:,:,2], arrow[:,:,3]

    # Apply new color where alpha is not 0 (ignoring fully transparent pixels)
    mask = a != 0
    arrow[:,:,0][mask] = new_color[0]  # Red channel
    arrow[:,:,1][mask] = new_color[1]  # Green channel
    arrow[:,:,2][mask] = new_color[2]  # Blue channel

    return arrow

def generate_color_codes(values):
    """
    Generates color codes from blue (for smaller values) to red (for larger values).
    :param values: List of values to map to colors.
    :return: List of color codes in RGB format.
    """
    # Normalize values to range between 0 and 1
    min_val = float(np.min(values))
    max_val = float(np.max(values))
    if min_val==max_val:
        norm_values = values
    else:
        norm_values = (values - min_val) / (max_val - min_val)

    # Generate RGB color for each value, mapping 0 to blue and 1 to red
    color_codes = [plt.cm.coolwarm(v)[:3] for v in norm_values]  # Use a colormap like coolwarm
    
    # Convert to 0-255 scale for RGB
    color_codes = [(int(r * 255), int(g * 255), int(b * 255)) for r, g, b in color_codes]

    return color_codes


def generate_color_code(value, min_val, max_val):
    """
    Generates one color code from blue (for smaller values) to red (for larger values).
    :param values: List of values to map to colors.
    :return: A color code in RGB format.
    """
    # Normalize values to range between 0 and 1
    if float(min_val)==float(max_val):
        norm_value = value
    else:
        norm_value = (value - min_val) / (max_val - min_val)

    # Generate RGB color for each value, mapping 0 to bluse and 1 to red
    color_code = plt.cm.coolwarm(norm_value)[:3]
    r, g, b = color_code
    
    # Convert to 0-255 scale for RGB
    color_code = (int(r * 255), int(g * 255), int(b * 255))

    return color_code

def main(vis_config, value_func_path, fig_path):
    # Load map and sprite
    blend_map = load_map(vis_config['map_path'])
    arrow = load_sprite(vis_config['arrow_path'])

    # Create an empty overlay (all zeros with the same size as map)
    overlay_low_hp0 = np.zeros_like(blend_map, dtype=np.uint8)
    overlay_low_hp1 = np.zeros_like(blend_map, dtype=np.uint8)

    # Load learned value_func
    with open(value_func_path) as f:
        value_func = json.load(f)
    value_keys = list(value_func.keys())
    value_keys_low_hp0 = [k for k in value_func.keys() if k.endswith('_0')]
    value_keys_low_hp1 = [k for k in value_func.keys() if k.endswith('_1')]
    # max_val = np.max([max(value_func[k].values()) for k in value_keys])
    # min_val = np.min([min(value_func[k].values()) for k in value_keys])
    max_val_low_hp0 = np.max([max(value_func[k].values()) for k in value_keys_low_hp0])
    min_val_low_hp0 = np.min([min(value_func[k].values()) for k in value_keys_low_hp0])
    max_val_low_hp1 = np.max([max(value_func[k].values()) for k in value_keys_low_hp1])
    min_val_low_hp1 = np.min([min(value_func[k].values()) for k in value_keys_low_hp1])


    # Plot the best valued action at each location
    for key in value_keys:
        map_id, x, y, low_hp = key.split('_')
        # if low_hp=='1': # for now, just focus on plotting the value at high_hp state
        #     continue
        best_value = max(value_func[key].values())
        # best_action = max(value_func[key], key=value_func[key].get)
        best_actions = [k for k, v in value_func[key].items() if v == best_value]
        best_action = np.random.choice(best_actions) # Greedy action (break tie randomly)
        
        # Rotate arrow
        rotated_arrow = np.array(arrow.rotate(vis_config['arrow_rotation'][int(best_action)]))
        
        # Add sprite to blend
        # overlay_map = add_sprite(overlay, walks[0], x, y, map_id) # add arrow
        if low_hp=='0':
            # Change arrow color
            color_code = generate_color_code(best_value, min_val_low_hp0, max_val_low_hp0)
            rotated_colored_arrow = change_arrow_color(rotated_arrow, color_code)        
            overlay_low_hp0 = add_sprite(overlay_low_hp0, rotated_colored_arrow, int(x), int(y), int(map_id)) # add arrow
        elif low_hp=='1':
            # Change arrow color
            color_code = generate_color_code(best_value, min_val_low_hp1, max_val_low_hp1)
            rotated_colored_arrow = change_arrow_color(rotated_arrow, color_code)
            overlay_low_hp1 = add_sprite(overlay_low_hp1, rotated_colored_arrow, int(x), int(y), int(map_id)) # add arrow

    # Blend arrows onto map
    blend_map_low_hp0 = blend_overlay(blend_map, overlay_low_hp0)
    blend_map_low_hp1 = blend_overlay(blend_map, overlay_low_hp1)

    # Visualize the final map with Red's icon overlaid
    plt.figure(figsize=(20,20))
    plt.imshow(blend_map_low_hp0[3400:3960, 650:1200]) # only show the areas around pallet town
    plt.axis('off')
    plt.savefig(f"{fig_path}_low_hp0.png")
    # plt.show()
    
    plt.figure(figsize=(20,20))
    plt.imshow(blend_map_low_hp1[3400:3960, 650:1200]) # only show the areas around pallet town
    plt.axis('off')
    plt.savefig(f"{fig_path}_low_hp1.png")
    # plt.show()
    

if __name__=="__main__":
    main()
    