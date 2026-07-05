import csv
import os
import time
import math

class RunLogger:
    def __init__(self, log_dir, summary_path=None, run_id=0,
                 anomaly_type='normal', intensity_param=0.0, anomaly_flag=None, start_pos=-1,
                 steering_gain=1.0, steering_bias=0.0, frame_drop=0, brightness_coeff=1.0,
                 cmd_latency=0, mass_scale=1.0, cam_pitch=0.0, occlusion_fraction=0.4,
                 friction_scale=1.0, drag_force=0.0, blur_kernel=7,
                 one_wheel_friction_scale=1.0, one_wheel_friction_index=-1,
                 anomaly_intensities=None):
        log_dir = log_dir + f'/log_{run_id}.csv'
        self.log_path = log_dir
        os.makedirs(os.path.dirname(log_dir), exist_ok=True)
        self.f = open(log_dir, 'w', newline='')
        self.writer = csv.writer(self.f)
        self.writer.writerow([
            'frame_id', 'uncorrupted_image_path', 'corrupted_image_path', 'timestamp_ms',
            'sim_time', 'run_sim_time',
            'steering_cmd', 'steering_act',
            'throttle_cmd', 'throttle_act',
            'pos_x', 'pos_z',
            'yaw_rate', 'speed', 'cte',
            'accel_x', 'accel_z', 'yaw', 'pitch', 'roll',
            'anomaly_param', 'anomaly_intensity', 'crashed'
        ])
        self.frame_id = 0
        self.first_sim_time = None
        if anomaly_flag is None:
            anomaly_flag = []
        self.anomaly_flag_list = anomaly_flag if isinstance(anomaly_flag, list) else [str(anomaly_flag)]
        self.start_pos = start_pos

        self.summary_path = summary_path
        self.run_id = run_id
        self.anomaly_type = anomaly_type
        self.anomaly_param = intensity_param
        self.outcome = 'SAFE'
        
        default_intensities = {
            'steering_gain': steering_gain,
            'steering_bias': steering_bias,
            'frame_drop': frame_drop,
            'brightness_coeff': brightness_coeff,
            'cmd_latency': cmd_latency,
            'mass_scale': mass_scale,
            'cam_pitch': cam_pitch,
            'occlusion_fraction': occlusion_fraction,
            'friction_scale': friction_scale,
            'drag_force': drag_force,
            'blur_kernel': blur_kernel,
            'one_wheel_friction_scale': one_wheel_friction_scale,
            'one_wheel_friction_index': one_wheel_friction_index,
        }
        intensity_source = anomaly_intensities or default_intensities

        # Build anomaly intensity dict for per-timestep recording.
        self.anomaly_intensities = {
            anomaly: intensity_source[anomaly]
            for anomaly in self.anomaly_flag_list
            if anomaly in intensity_source
        }
        if anomaly_intensities and not self.anomaly_intensities:
            self.anomaly_flag_list = list(anomaly_intensities.keys())
            self.anomaly_intensities = dict(anomaly_intensities)

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
            sim_time, pos_x, pos_z, yaw_rate, speed, cte,
            accel_x, accel_z, yaw, pitch, roll, hit):
        ts = int(time.time() * 1000)
        if sim_time is not None and self.first_sim_time is None:
            self.first_sim_time = sim_time
        run_sim_time = 0.0 if sim_time is None or self.first_sim_time is None else sim_time - self.first_sim_time
        sim_time_value = '' if sim_time is None else sim_time
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
            sim_time_value, run_sim_time,
            steering_cmd, steering_act,
            throttle_cmd, throttle_act,
            pos_x, pos_z,
            yaw_rate, speed, cte,
            accel_x, accel_z, yaw, pitch, roll,
            anomaly_param_str, anomaly_intensity_str, 0
        ])
        self.cumulative_cte += abs(cte) if cte else 0.0
        if self.prev_x is not None:
            self.euclidean_dist += math.sqrt((pos_x - self.prev_x)**2 + (pos_z - self.prev_z)**2)
        self.prev_x, self.prev_z = pos_x or 0.0, pos_z or 0.0
        # Change this to hit != 'none' to detect any barrier crash
        #if hit and hit == 'barrier': ###REMOVED IN ORDER TO IMPLEMENT 3 SECOND TIMEOUT RATHER THAN INSTANT TIMEOUT
        #    self.outcome = 'CRASH'
        self.frame_id += 1

    def set_outcome(self, outcome):
        self.outcome = outcome

    def _mark_last_row_crashed(self):
        if self.frame_id == 0:
            return

        with open(self.log_path, 'r', newline='') as f:
            rows = list(csv.reader(f))

        if len(rows) <= 1:
            return

        header = rows[0]
        if 'crashed' not in header:
            header.append('crashed')

        crashed_idx = header.index('crashed')
        for row in rows[1:]:
            while len(row) <= crashed_idx:
                row.append('0')

        rows[-1][crashed_idx] = '1'

        with open(self.log_path, 'w', newline='') as f:
            csv.writer(f).writerows(rows)

    def shutdown(self):
        self.f.close()
        if self.outcome == 'CRASH':
            self._mark_last_row_crashed()
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
