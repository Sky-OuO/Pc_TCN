import numpy as np
from sgp4.api import Satrec, jday
from datetime import datetime, timedelta
import math
import json
from tqdm import tqdm
import matplotlib.pyplot as plt
import glob
class Feature_Generator:
    def __init__(self, tle_line1, tle_line2, tle_line3, tle_line4, tca_time):
        self.sat1 = Satrec.twoline2rv(tle_line1, tle_line2)
        self.sat2 = Satrec.twoline2rv(tle_line3, tle_line4)
        self.tca_time = tca_time

        self.sat1_epoch = (datetime(1949, 12, 31) + timedelta(days=self.sat1.jdsatepoch + self.sat1.jdsatepochF))
        self.sat2_epoch = (datetime(1949, 12, 31) + timedelta(days=self.sat2.jdsatepoch + self.sat2.jdsatepochF))

    def calculate_distance(self, jd, fr):
        e1, r1, v1 = self.sat1.sgp4(jd, fr)
        if e1 != 0:
            return float('inf')
        e2, r2, v2 = self.sat2.sgp4(jd, fr)
        if e2 != 0:
            return float('inf')
        distance = math.sqrt((r1[0]-r2[0])**2 + (r1[1]-r2[1])**2 + (r1[2]-r2[2])**2)
        return distance
    
    def generate_tca_centered_data(self, delta_t_minutes=5, time_step_seconds=1):
        start_time = self.tca_time - 2*timedelta(minutes=delta_t_minutes)
        end_time = self.tca_time
        positions = []
        velocities = []
        times = [] 
        
        while start_time <= end_time:
            jd, fr = jday(start_time.year, start_time.month, start_time.day,
                         start_time.hour, start_time.minute, start_time.second)

            e1, r1, v1 = self.sat1.sgp4(jd, fr)
            e2, r2, v2 = self.sat2.sgp4(jd, fr)
            
            if e1 == 0 and e2 == 0:
                positions.append((r1, r2))
                velocities.append((v1, v2))
                times.append(start_time) 
            start_time += timedelta(seconds=time_step_seconds)

        return positions, velocities, times 
    
    def calculate_relative_features(self, positions, velocities, times):
        features = []

        age1 = (self.tca_time - self.sat1_epoch).total_seconds() / 86400.0
        age2 = (self.tca_time - self.sat2_epoch).total_seconds() / 86400.0

        for (r1, r2), (v1, v2) in zip(positions, velocities):
            r_rel = np.array(r2) - np.array(r1)
            v_rel = np.array(v2) - np.array(v1)
            
            distance = np.linalg.norm(r_rel)
            if distance > 1e-6:  
                closure_rate = np.dot(r_rel, v_rel) / distance
                radial_unit = r_rel / distance
            else:
                closure_rate = 0
                radial_unit = np.zeros(3)

            relative_speed = np.linalg.norm(v_rel)
            
            v_radial = np.dot(v_rel, radial_unit)
            v_radial_vec = v_radial * radial_unit
            v_tangential = v_rel - v_radial_vec
            v_tangential_mag = np.linalg.norm(v_tangential)

            r1_mag = np.linalg.norm(r1)
            v1_mag = np.linalg.norm(v1)
            r2_mag = np.linalg.norm(r2)
            v2_mag = np.linalg.norm(v2)

            features.append([
                distance,           # relative distance  
                closure_rate,          
                relative_speed,     
                v_tangential_mag,   # relative tangential velocity
                *r_rel,             # relative position vector (3D)
                *v_rel,             # relative velocity vector (3D)
                r1_mag,             # sat1 orbit radius
                v1_mag,             # sat1 orbit velocity
                r2_mag,             # sat2 orbit radius
                v2_mag,             # sat2 orbit velocity
                age1,               # TLE 1 age 
                age2,               # TLE 2 age
            ])
        
        return np.array(features) 

def feature_generator_iterater(train_data):
    features = []
    labels = []
    for unit in tqdm(train_data):
        tca = unit.get("TCA")
        pc_gt = float(unit.get("pc_gt", 0))
        tle_1 = unit.get("sat_1")
        tle_2 = unit.get("sat_2")
        tle_1_line1 = tle_1[0].strip()
        tle_1_line2 = tle_1[1]
        tle_2_line1 = tle_2[0].strip()  
        tle_2_line2 = tle_2[1]
        tca_time = datetime.strptime(tca, "%Y-%m-%d %H:%M:%S.%f")
        Feature_generator = Feature_Generator(tle_1_line1, tle_1_line2, tle_2_line1, tle_2_line2, tca_time)

        positions, velocities, times = Feature_generator.generate_tca_centered_data(delta_t_minutes=25, time_step_seconds=10)
        unit_features = Feature_generator.calculate_relative_features(positions, velocities, times)
        features.append(unit_features)
        labels.append(float(pc_gt))
    return np.array(features), np.array(labels)


def augment_sparse_bins(all_data, min_samples=300, tca_offsets_sec=(-5, -3, 3, 5), seed=42):
    rng = np.random.RandomState(seed)
    bin_edges = [-30, -8, -7, -6, -5, -4, -3, -2, -1, 0.01]

    # Bin the data
    binned = {i: [] for i in range(len(bin_edges) - 1)}
    for unit in all_data:
        pc = float(unit.get("pc_gt", 0))
        if pc <= 0:
            continue
        log_pc = math.log10(pc)
        for i in range(len(bin_edges) - 1):
            if bin_edges[i] <= log_pc < bin_edges[i + 1]:
                binned[i].append(unit)
                break

    aug_features, aug_labels = [], []
    for i, items in binned.items():
        count = len(items)
        if count == 0 or count >= min_samples:
            continue
        n_needed = min_samples - count
        print(f"  Bin [{bin_edges[i]:+.0f}, {bin_edges[i+1]:+.0f}): {count} samples, augmenting {n_needed} more")

        generated = 0
        while generated < n_needed:
            unit = items[rng.randint(len(items))]
            offset = rng.choice(tca_offsets_sec)

            try:
                tle_1 = unit["sat_1"]
                tle_2 = unit["sat_2"]
                tca_time = datetime.strptime(unit["TCA"], "%Y-%m-%d %H:%M:%S.%f")
                shifted_tca = tca_time + timedelta(seconds=float(offset))

                fg = Feature_Generator(
                    tle_1[0].strip(), tle_1[1],
                    tle_2[0].strip(), tle_2[1],
                    shifted_tca
                )
                positions, velocities, times = fg.generate_tca_centered_data(
                    delta_t_minutes=25, time_step_seconds=10
                )
                feat = fg.calculate_relative_features(positions, velocities, times)
                if len(feat) > 0:
                    aug_features.append(feat)
                    aug_labels.append(float(unit["pc_gt"]))
                    generated += 1
            except Exception:
                continue

    print(f"generated {len(aug_labels)} total new samples")
    return aug_features, aug_labels

def plot_label_distribution(labels, save_path_prefix="labels_dist"):
    labels = np.array(labels)
    positive = labels[labels > 0]
    if positive.size <= 0:
        return
    log_labels = np.log10(positive)
    plt.figure(figsize=(8, 4))
    plt.hist(log_labels, bins=50)
    plt.xlabel("log10(Pc)")
    plt.ylabel("Count")
    plt.title("Distribution of log10(Pc)")
    plt.tight_layout()
    plt.savefig(f"{save_path_prefix}_log10.png", dpi=200)
    plt.close()

if __name__ == "__main__":
    total_features = []
    total_labels = []
    all_raw_data = []
    for data_files in glob.glob("data/*.json"):
        train_data = json.load(open(data_files, "r"))
        all_raw_data.extend(train_data)
        features, labels = feature_generator_iterater(train_data)
        total_features.append(features)
        total_labels.append(labels)

    print("\nAugmenting sparse bins via TCA time perturbation...")
    aug_features, aug_labels = augment_sparse_bins(all_raw_data, min_samples=300)
    if aug_features:
        total_features.append(np.array(aug_features))
        total_labels.append(np.array(aug_labels))

    all_features = np.concatenate(total_features, axis=0)
    all_labels = np.concatenate(total_labels, axis=0)

    feature_path = "features.npy"
    np.save(feature_path, all_features)
    label_path = "labels.npy"
    np.save(label_path, all_labels)
    plot_label_distribution(all_labels, save_path_prefix="labels_dist")
    print(f"Features shape: {all_features.shape}, Labels shape: {all_labels.shape}")
    print(f"Features saved to {feature_path}, Labels saved to {label_path}")
