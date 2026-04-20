import csv
import os
import time
import math

class RunLogger:
    def __init__(self, log_dir, summary_path=None, run_id=0,
                 anomaly_type='normal', intensity_param=0.0, anomaly_flag=None, start_pos=-1,
                 steering_gain=1.0, steering_bias=0.0, frame_drop=0, brightness_coeff=1.0,
                 cmd_latency=0, mass_scale=1.0, cam_pitch=0.0, occlusion_fraction=0.4,
                 friction_scale=1.0):
        log_dir = log_dir + f'/log_{run_id}.csv'
        os.makedirs(os.path.dirname(log_dir), exist_ok=True)
        self.f = open(log_dir, 'w', newline='')
        self.writer = csv.writer(self.f)
        self.writer.writerow([
            'frame_id', 'uncorrupted_image_path', 'corrupted_image_path', 'timestamp_ms',
            'steering_cmd', 'steering_act',
            'throttle_cmd', 'throttle_act',
            'pos_x', 'pos_z',
            'yaw_rate', 'speed', 'cte',
            'accel_x', 'accel_z', 'yaw', 'pitch', 'roll', 'anomaly_param', 'anomaly_intensity'
        ])
        self.frame_id = 0
        if anomaly_flag is None:
            anomaly_flag = []
        self.anomaly_flag_list = anomaly_flag if isinstance(anomaly_flag, list) else [str(anomaly_flag)]
        self.start_pos = start_pos

        self.summary_path = summary_path
        self.run_id = run_id
        self.anomaly_type = anomaly_type
        self.anomaly_param = intensity_param
        self.outcome = 'SAFE'
        
        # Store anomaly/parameter values
        self.steering_gain = steering_gain
        self.steering_bias = steering_bias
        self.frame_drop = frame_drop
        self.brightness_coeff = brightness_coeff
        self.cmd_latency = cmd_latency
        self.mass_scale = mass_scale
        self.cam_pitch = cam_pitch
        self.occlusion_fraction = occlusion_fraction
        self.friction_scale = friction_scale
        
        # Build anomaly intensity dict for per-timestep recording
        self.anomaly_intensities = {}
        for anomaly in self.anomaly_flag_list:
            if anomaly == 'steering_gain':
                self.anomaly_intensities['steering_gain'] = steering_gain
            elif anomaly == 'steering_bias':
                self.anomaly_intensities['steering_bias'] = steering_bias
            elif anomaly == 'frame_drop':
                self.anomaly_intensities['frame_drop'] = frame_drop
            elif anomaly == 'brightness_coeff':
                self.anomaly_intensities['brightness_coeff'] = brightness_coeff
            elif anomaly == 'cmd_latency':
                self.anomaly_intensities['cmd_latency'] = cmd_latency
            elif anomaly == 'mass_scale':
                self.anomaly_intensities['mass_scale'] = mass_scale
            elif anomaly == 'cam_pitch':
                self.anomaly_intensities['cam_pitch'] = cam_pitch
            elif anomaly == 'occlusion_fraction':
                self.anomaly_intensities['occlusion_fraction'] = occlusion_fraction
            elif anomaly == 'friction_scale':
                self.anomaly_intensities['friction_scale'] = friction_scale

        self.cumulative_cte = 0.0
        self.euclidean_dist = 0.0
        self.prev_x, self.prev_z = None, None

        if summary_path:
            write_header = not os.path.exists(summary_path)
            self.summary_f = open(summary_path, 'a', newline='')
            self.summary_writer = csv.writer(self.summary_f)
            if write_header:
                self.summary_writer.writerow([
                    'run_id', 'start_pos', 'anomaly_param',
                    'outcome', 'total_distance', 'time_to_failure',
                    'cumulative_cte', 'avg_cte'
                ])

    def run(self, steering_cmd, steering_act, throttle_cmd, throttle_act,
            pos_x, pos_z, yaw_rate, speed, cte,
            accel_x, accel_z, yaw, pitch, roll, hit):
        ts = int(time.time() * 1000)
        normal_img_path = f"imgs/normal/image_{self.run_id}.jpg"
        corrupt_path = f"imgs/noise/noise_image_{self.run_id}.jpg"
        
        # Format anomaly parameter as list of types: {anomaly1, anomaly2}
        if self.anomaly_flag_list:
            anomaly_param_str = '{' + ', '.join(self.anomaly_flag_list) + '}'
        else:
            anomaly_param_str = '{}'
        
        # Format anomaly intensities as list of values: {1.0, 2.0}
        if self.anomaly_intensities:
            intensity_values = [str(v) for v in self.anomaly_intensities.values()]
            anomaly_intensity_str = '{' + ', '.join(intensity_values) + '}'
        else:
            anomaly_intensity_str = '{}'
        
        self.writer.writerow([
            self.frame_id, normal_img_path, corrupt_path, ts,
            steering_cmd, steering_act,
            throttle_cmd, throttle_act,
            pos_x, pos_z,
            yaw_rate, speed, cte,
            accel_x, accel_z, yaw, pitch, roll, anomaly_param_str, anomaly_intensity_str
        ])
        self.cumulative_cte += abs(cte) if cte else 0.0
        if self.prev_x is not None:
            self.euclidean_dist += math.sqrt((pos_x - self.prev_x)**2 + (pos_z - self.prev_z)**2)
        self.prev_x, self.prev_z = pos_x or 0.0, pos_z or 0.0
        # Change this to hit != 'none' to detect any barrier crash
        if hit and hit == 'barrier':
            self.outcome = 'CRASH'
        self.frame_id += 1

    def set_outcome(self, outcome):
        self.outcome = outcome

    def shutdown(self):
        self.f.close()
        if self.summary_path:
            avg_cte = self.cumulative_cte / self.frame_id if self.frame_id > 0 else 0.0
            
            # Format anomaly parameter as list of types: {anomaly1, anomaly2}
            if self.anomaly_flag_list:
                anomaly_param_str = '{' + ', '.join(self.anomaly_flag_list) + '}'
            else:
                anomaly_param_str = '{}'
            
            self.summary_writer.writerow([
                self.run_id, self.start_pos, anomaly_param_str,
                self.outcome, round(self.euclidean_dist, 3), self.frame_id,
                round(self.cumulative_cte, 3), round(avg_cte, 5)
            ])
            self.summary_f.close()
