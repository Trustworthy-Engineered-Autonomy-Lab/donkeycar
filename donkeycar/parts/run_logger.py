import csv
import os
import time
import math

class RunLogger:
    def __init__(self, log_dir, summary_path=None, run_id=0,
                 anomaly_type='normal', intensity_param=0.0, anomaly_flag=0, start_pos = -1):
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
            'accel_x', 'accel_z', 'yaw', 'pitch', 'roll', 'anomaly_flag'
        ])
        self.frame_id = 0
        self.anomaly_flag = anomaly_flag
        self.start_pos = start_pos

        self.summary_path = summary_path
        self.run_id = run_id
        self.anomaly_type = anomaly_type
        self.anomaly_param = intensity_param
        self.outcome = 'SAFE'

        self.cumulative_cte = 0.0
        self.euclidean_dist = 0.0
        self.prev_x, self.prev_z = None, None

        if summary_path:
            write_header = not os.path.exists(summary_path)
            self.summary_f = open(summary_path, 'a', newline='')
            self.summary_writer = csv.writer(self.summary_f)
            if write_header:
                self.summary_writer.writerow([
                    'run_id', 'start_pos', 'anomaly_type', 'anomaly_param',
                    'outcome', 'total_distance', 'time_to_failure',
                    'cumulative_cte', 'avg_cte'
                ])

    def run(self, steering_cmd, steering_act, throttle_cmd, throttle_act,
            pos_x, pos_z, yaw_rate, speed, cte,
            accel_x, accel_z, yaw, pitch, roll, hit):
        ts = int(time.time() * 1000)
        normal_img_path = f"imgs/normal/image_{self.run_id}.jpg"
        corrupt_path = f"imgs/noise/noise_image_{self.run_id}.jpg"
        self.writer.writerow([
            self.frame_id, normal_img_path, corrupt_path, ts,
            steering_cmd, steering_act,
            throttle_cmd, throttle_act,
            pos_x, pos_z,
            yaw_rate, speed, cte,
            accel_x, accel_z, yaw, pitch, roll, self.anomaly_flag
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
            self.summary_writer.writerow([
                self.run_id, self.start_pos, self.anomaly_type, self.anomaly_param,
                self.outcome, round(self.euclidean_dist, 3), self.frame_id,
                round(self.cumulative_cte, 3), round(avg_cte, 5)
            ])
            self.summary_f.close()
