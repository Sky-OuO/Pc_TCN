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
cfg = json.load(open("config.json"))
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

import math

def _build_uncertainty_features(sat, tle_age_days, tca_time, name=None):
    EPS = 1e-10
    MU = 398600.4418          # km^3/s^2
    RE = 6378.137             # Earth radius (km)

    e = sat.ecco
    inc = sat.inclo                  # rad
    n = sat.no_kozai                 # rad/min
    bstar = sat.bstar
    age = tle_age_days

    n_rad_sec = n / 60.0
    a = (MU / (n_rad_sec ** 2)) ** (1.0 / 3.0)
    perigee = a * (1 - e) - RE
    apogee  = a * (1 + e) - RE
    period = 2 * math.pi / n        

    raw_features = [
        e,
        inc,
        math.log(max(n, EPS)),
        math.log(abs(bstar) + EPS),
        math.log1p(max(age, 0.0)),
        age * abs(bstar),
        age * n,
        perigee,
        apogee,
        period,
    ]

    debris_cfg = cfg.get("debris_phase", {})
    phase = (
        _debris_phase(
            _compute_launch_age_years(sat.intldesg, tca_time),
            a=debris_cfg.get("threshold_a_years", 2.0),
            b=debris_cfg.get("threshold_b_years", 7.0),
        )
        if _is_debris_by_name(name)
        else [0, 0, 0]
    )
    return raw_features + phase

def _encounter_plane_distance(r_rel, v_rel):
    speed = np.linalg.norm(v_rel)
    if speed <= 1e-10:
        return np.linalg.norm(r_rel)
    v_hat = v_rel / speed
    r_plane = r_rel - np.dot(r_rel, v_hat) * v_hat
    return np.linalg.norm(r_plane)

def _max_pc_proxy_log10(d_enc_km):
    eps = 1e-12
    d = max(float(d_enc_km), eps)
    radii_km = [0.005, 0.01, 0.025, 0.05]
    aspect_ratios = [1.0, 5.0, 15.0]
    sigma_x_values = np.logspace(-3, 3, 49)
    pmax = eps
    for rc in radii_km:
        for ar in aspect_ratios:
            sigma_y_values = sigma_x_values / ar
            pc_values = (rc ** 2 / (2.0 * sigma_x_values * sigma_y_values)) * \
                        np.exp(-0.5 * (d / sigma_x_values) ** 2)
            pmax = max(pmax, float(np.max(pc_values)))
    return math.log10(min(max(pmax, eps), 1.0))

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

    def generate_tca_centered_data(self, delta_t_minutes=10, time_step_seconds=10,
                                   near_tca_minutes=1, near_tca_time_step_seconds=2):
        start_time = self.tca_time - timedelta(minutes=delta_t_minutes)
        split_time = self.tca_time - timedelta(minutes=near_tca_minutes)
        end_time = self.tca_time
        positions = []
        velocities = []
        times = [] 

        def _append_sample(sample_time):
            jd, fr = jday(sample_time.year, sample_time.month, sample_time.day,
                         sample_time.hour, sample_time.minute, sample_time.second)

            e1, r1, v1 = self.sat1.sgp4(jd, fr)
            e2, r2, v2 = self.sat2.sgp4(jd, fr)

            if e1 == 0 and e2 == 0:
                positions.append((r1, r2))
                velocities.append((v1, v2))
                times.append(sample_time)
        
        current_time = start_time
        while current_time <= split_time:
            _append_sample(current_time)
            current_time += timedelta(seconds=time_step_seconds)

        current_time = split_time + timedelta(seconds=near_tca_time_step_seconds)
        while current_time <= end_time:
            _append_sample(current_time)
            current_time += timedelta(seconds=near_tca_time_step_seconds)

        if not times or times[-1] != end_time:
            _append_sample(end_time)

        return positions, velocities, times 
    
    def calculate_relative_features(self, positions, velocities, times):
        features = []

        age1 = (self.tca_time - self.sat1_epoch).total_seconds() / 86400.0
        age2 = (self.tca_time - self.sat2_epoch).total_seconds() / 86400.0
        unc1 = _build_uncertainty_features(self.sat1, age1, self.tca_time, self.sat1_name)
        unc2 = _build_uncertainty_features(self.sat2, age2, self.tca_time, self.sat2_name)

        distances = []
        for (r1, r2), (v1, v2) in zip(positions, velocities):
            d = np.linalg.norm(np.array(r2) - np.array(r1))
            distances.append(d)
        distances = np.array(distances)
        min_dist_idx = int(np.argmin(distances))
        tca_idx = len(times) - 1  # last timestep = labeled TCA
        tca_to_min_offset = float(min_dist_idx - tca_idx)  # <0: min dist before TCA
        min_distance = float(distances[min_dist_idx])

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
            d_enc = _encounter_plane_distance(r_rel, v_rel)
            log_pmax_proxy = _max_pc_proxy_log10(d_enc)

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
                d_enc,              # encounter-plane miss distance
                log_pmax_proxy,     # max Pc proxy from radius scan
                tca_to_min_offset,  # timesteps from TCA to SGP4 min-dist point
                min_distance,       # SGP4-computed minimum distance (km)
                *unc1,              # obj1 uncertainty features (7-dim: 4 raw + 3 phase)
                *unc2,              # obj2 uncertainty features (7-dim: 4 raw + 3 phase)
            ])
        
        return np.array(features) 

def feature_generator_iterater(train_data, delta_t_minutes=10, time_step_seconds=10):
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

        positions, velocities, times = Feature_generator.generate_tca_centered_data(
            delta_t_minutes=delta_t_minutes,
            time_step_seconds=time_step_seconds,
            near_tca_minutes=cfg.get("feature_generation", {}).get("near_tca_minutes", 2),
            near_tca_time_step_seconds=cfg.get("feature_generation", {}).get("near_tca_time_step_seconds", 2),
        )
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

    # Load config for params
    cfg = json.load(open("config.json"))

    gen_cfg = cfg.get("feature_generation", {})
    all_features, all_labels = feature_generator_iterater(
        deduped_raw,
        delta_t_minutes=gen_cfg.get("delta_t_minutes", 10),
        time_step_seconds=gen_cfg.get("time_step_seconds", 10),
    )
    print(f"Generated features shape: {all_features.shape}, Labels shape: {all_labels.shape}")
    # min_distance_filter
    filter_cfg = cfg.get("feature_generation", {}).get("min_distance_filter", {})
    if filter_cfg.get("enabled", True):
        threshold_km = filter_cfg.get("threshold_km", 5.0)
        pc_min = filter_cfg.get("pc_min", 1e-3)
        min_dists = all_features[:, 0, 17] 
        remove_mask = (min_dists > threshold_km) & (all_labels > pc_min)
        n_removed = int((remove_mask).sum())
        all_features = all_features[~remove_mask]
        all_labels = all_labels[~remove_mask]
        print(f"Min-distance filter (window-min>{threshold_km:.0f} km & Pc>{pc_min:.0e}): "
              f"removed {n_removed}, kept {len(all_labels)}")

    feature_path = "params/features.npy"
    np.save(feature_path, all_features)
    label_path = "params/labels.npy"
    np.save(label_path, all_labels)
    plot_label_distribution(all_labels, save_path_prefix="figures/labels_dist")
    print(f"Features shape: {all_features.shape}, Labels shape: {all_labels.shape}")
    print(f"Features saved to {feature_path}, Labels saved to {label_path}")
