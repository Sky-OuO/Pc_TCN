import numpy as np
from sgp4.api import Satrec, jday
from datetime import datetime, timedelta
import math
import json
from tqdm import tqdm
import matplotlib.pyplot as plt
import glob
import os

if not os.path.exists("params"):
    os.makedirs("params", exist_ok=True)
if not os.path.exists("figures"):
    os.makedirs("figures", exist_ok=True)

def _parse_launch_year(intldesg):
    try:
        yy = int(intldesg.strip()[:2])
        return (2000 + yy) if yy < 57 else (1900 + yy)
    except (ValueError, IndexError):
        return 2000

def _compute_launch_age_years(intldesg, tca_time):
    launch_year = _parse_launch_year(intldesg)
    launch_date = datetime(launch_year, 7, 1)
    return (tca_time - launch_date).total_seconds() / (365.25 * 86400.0)

def _is_debris_by_name(name):
    if not name or not isinstance(name, str):
        return False
    upper = name.upper()
    return "DEB" in upper or "R/B" in upper

def _debris_phase(launch_age_years, a=2.0, b=7.0):
    if launch_age_years < a:
        return [1, 0, 0]
    elif launch_age_years < b:
        return [0, 1, 0]
    else:
        return [0, 0, 1]

def _build_uncertainty_features(sat, tle_age_days, tca_time, name=None):
    e     = sat.ecco
    bstar = sat.bstar
    tau   = tle_age_days
    phase = (_debris_phase(_compute_launch_age_years(sat.intldesg, tca_time))
             if _is_debris_by_name(name) else [0, 0, 0])
    return [e, bstar, tau, bstar * tau, e * tau, tau ** 2] + phase

class Feature_Generator:
    def __init__(self, tle_line1, tle_line2, tle_line3, tle_line4, tca_time,
                 sat1_name=None, sat2_name=None):
        self.sat1 = Satrec.twoline2rv(tle_line1, tle_line2)
        self.sat2 = Satrec.twoline2rv(tle_line3, tle_line4)
        self.tca_time = tca_time
        self.sat1_name = sat1_name
        self.sat2_name = sat2_name

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
        unc1 = _build_uncertainty_features(self.sat1, age1, self.tca_time, self.sat1_name)
        unc2 = _build_uncertainty_features(self.sat2, age2, self.tca_time, self.sat2_name)

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
                *unc1,              # obj1 uncertainty features (9-dim)
                *unc2,              # obj2 uncertainty features (9-dim)
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
        sat_1_name = unit.get("sat_1_name", "")
        sat_2_name = unit.get("sat_2_name", "")
        tle_1_line1 = tle_1[0].strip()
        tle_1_line2 = tle_1[1]
        tle_2_line1 = tle_2[0].strip()  
        tle_2_line2 = tle_2[1]
        tca_time = datetime.strptime(tca, "%Y-%m-%d %H:%M:%S.%f")
        Feature_generator = Feature_Generator(tle_1_line1, tle_1_line2, tle_2_line1, tle_2_line2, tca_time,
                                              sat_1_name, sat_2_name)

        positions, velocities, times = Feature_generator.generate_tca_centered_data(delta_t_minutes=25, time_step_seconds=10)
        unit_features = Feature_generator.calculate_relative_features(positions, velocities, times)
        features.append(unit_features)
        labels.append(float(pc_gt))
    return np.array(features), np.array(labels)

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

def _dedup_key(unit):
    tle_1 = unit.get("sat_1", ["", ""])
    tle_2 = unit.get("sat_2", ["", ""])
    return (
        unit.get("TCA", ""),
        tle_1[0].strip() if len(tle_1) > 0 else "",
        tle_1[1].strip() if len(tle_1) > 1 else "",
        tle_2[0].strip() if len(tle_2) > 0 else "",
        tle_2[1].strip() if len(tle_2) > 1 else "",
    )

if __name__ == "__main__":
    all_raw: list[dict] = []
    for data_files in glob.glob("data/*.json"):
        train_data = json.load(open(data_files, "r"))
        all_raw.extend(train_data)

    seen = set()
    deduped_raw = []
    for unit in all_raw:
        key = _dedup_key(unit)
        if key not in seen:
            seen.add(key)
            deduped_raw.append(unit)

    print(f"Total records: {len(all_raw)}, after deduplication: {len(deduped_raw)} "
          f"({len(all_raw) - len(deduped_raw)} duplicates removed)")

    all_features, all_labels = feature_generator_iterater(deduped_raw)

    feature_path = "params/features.npy"
    np.save(feature_path, all_features)
    label_path = "params/labels.npy"
    np.save(label_path, all_labels)
    plot_label_distribution(all_labels, save_path_prefix="figures/labels_dist")
    print(f"Features shape: {all_features.shape}, Labels shape: {all_labels.shape}")
    print(f"Features saved to {feature_path}, Labels saved to {label_path}")
