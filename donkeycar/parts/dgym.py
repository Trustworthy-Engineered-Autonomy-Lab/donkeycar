from datetime import datetime
import os
import time
import gym
import gym_donkeycar
import random
import cv2
import numpy as np
import logging
logger = logging.getLogger(__name__)
import csv
import matplotlib.pyplot as plt
import torch
from PIL import Image
from donkeycar.parts.models.noise_generator import ImageAugmentor

def is_exe(fpath):
    return os.path.isfile(fpath) and os.access(fpath, os.X_OK)


class DonkeyGymEnv(object):

    def __init__(self, sim_path, host="127.0.0.1", port=9091, headless=0, noise="default_noise",env_name="donkey-generated-track-v0", sync="asynchronous", conf={}, record_location=False, record_gyroaccel=False, record_velocity=False, record_lidar=False, record_orientation=False,delay=0, num_drop=0, name="", folder_name=''):

        if sim_path != "remote":
            if not os.path.exists(sim_path):
                raise Exception(
                    "The path you provided for the sim does not exist.")

            if not is_exe(sim_path):
                raise Exception("The path you provided is not an executable.")

        conf["exe_path"] = sim_path
        conf["host"] = host
        conf["port"] = port
        conf["guid"] = 0
        conf["frame_skip"] = 1
        self.env = gym.make(env_name, conf=conf)
        print('debug', conf)
        print('debug 2', self.env)
        self.frame = self.env.reset()
        self.action = [0.0, 0.0, 0.0]
        self.running = True
        self.info = {'pos': (0., 0., 0.),
                     'speed': 0,
                     'cte': 0,
                     'gyro': (0., 0., 0.),
                     'accel': (0., 0., 0.),
                     'vel': (0., 0., 0.),
                     'lidar': [], 
                     'roll': 0.0,
                     'pitch': 0.0,
                     'yaw': 0.0}
        self.delay = float(delay) / 1000
        self.record_location = record_location
        self.record_gyroaccel = record_gyroaccel
        self.record_velocity = record_velocity
        self.record_lidar = record_lidar
        self.record_orientation = record_orientation
        self.cte_values = []
        self.abs_cte_values = []
        self.all_data = []
        self.buffer = []
        self.noise = noise
        self.augmentor = ImageAugmentor()
        self.num_drop = num_drop
        self.drop_counter = 0
        self.frozen_frame = None

        folder_path = folder_name + f"data_{env_name}_{noise}_{name}"
        self.data_folder = folder_path  # Store the folder name as an attribute
        os.makedirs(folder_path, exist_ok=True)
        

    def delay_buffer(self, frame, info):
        now = time.time()
        buffer_tuple = (now, frame, info)
        self.buffer.append(buffer_tuple)

        # go through the buffer
        num_to_remove = 0
        for buf in self.buffer:
            if now - buf[0] >= self.delay:
                num_to_remove += 1
                self.frame = buf[1]
            else:
                break

        # clear the buffer
        del self.buffer[:num_to_remove]

    def update(self):
        time_step = 0
        start_time = time.time()

        # File path in the dynamically created folder
        file_path = os.path.join(self.data_folder, 'cte_values.csv')

        # Open the file once outside the loop
        with open(file_path, mode='w', newline='') as file:
            self.cte_writer = csv.writer(file)
            self.cte_writer.writerow(['Time Step', "CTE"])  # Write the header

            try:
                while self.running and (time.time() - start_time) < 60:
                    controller_data = {
                    'steering_cmd': self.action[0],
                    'throttle_cmd': self.action[1],
                    'brake_cmd': self.action[2]
                }
                    if self.delay > 0.0:
                        current_frame, _, _, current_info = self.env.step(self.action)
                        self.delay_buffer(current_frame, current_info)
                    else:
                        self.frame, _, _, self.info = self.env.step(self.action)
                        cte_value = self.info["cte"]
                        self.cte_values.append(cte_value)
                        self.cte_writer.writerow([time_step, cte_value])
                        time_step += 1
            finally:
                print(f"Data saved in {file_path}")
                # kill the env
                self.env.close()




                
    def run_threaded(self, steering, throttle, brake=None):
        if steering is None or throttle is None:
            steering = 0.0 
            throttle = 0.0
        if brake is None:
            brake = 0.0

        self.action = [steering, throttle, brake]

        if self.num_drop > 0:
            if self.drop_counter == 0:
                self.frozen_frame = self.frame.copy()
            self.drop_counter = (self.drop_counter + 1) % (self.num_drop)
            frame_out = self.frozen_frame
        else:
            frame_out = self.frame
        

        if self.noise == "blur":
            frame_out = self.augmentor.add_defocus(frame_out)
            # Image.fromarray(self.frame).save(os.path.join(self.data_folder, "blur_sample.jpg"))

        # Output Sim-car position information if configured
        outputs = [frame_out]
        if self.record_location:
            outputs += self.info['pos'][0],  self.info['pos'][1],  self.info['pos'][2],  self.info['speed'], self.info['cte']
        if self.record_gyroaccel:
            outputs += self.info['gyro'][0], self.info['gyro'][1], self.info['gyro'][2], self.info['accel'][0], self.info['accel'][1], self.info['accel'][2]
        if self.record_velocity:
            outputs += self.info['vel'][0],  self.info['vel'][1],  self.info['vel'][2]
        if self.record_lidar:
            outputs += [self.info['lidar']]
        if self.record_orientation:
            outputs += self.info['roll'], self.info['pitch'], self.info['yaw']
        
        if len(outputs) == 1:
            return frame_out
        else:
            return outputs

    def plot_cte_values(self, arr ,file_name):
        plt.figure()
        plt.plot(arr)
        plt.xlabel('Time Steps')
        plt.ylabel('CTE (Cross-Track Error)')
        plt.title('CTE Over Time')
        plt.savefig(file_name)

    def shutdown(self):
        self.running = False
        time.sleep(0.2)

        # Save the plots in the dynamically created folder
        cte_plot_path = os.path.join(self.data_folder, "cte_plot.png")
        # abs_cte_plot_path = os.path.join(self.data_folder, "abs_cte_plot.png")

        self.plot_cte_values(self.cte_values, cte_plot_path)
        # self.plot_cte_values(self.abs_cte_values, abs_cte_plot_path)

        print(f"Plots saved in directory: {self.data_folder}")
        self.env.close()

