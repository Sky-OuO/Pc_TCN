import pandas as pd
import json
import time
import io
import requests
from collections import defaultdict
import os

class DataCollectionPipeline:
    def __init__(self, config):
        self.prob_bins = config.get('prob_bins', [
            (1e-10, 1e-8),
            (1e-8, 1e-6),
            (1e-6, 1e-5),
            (1e-5, 1e-4),
            (1e-4, 1e-3),
            (1e-3, 1e-2),
            (1e-2, 1e-0) 
        ])
        self.min_samples_per_bin = config.get('min_samples_per_bin', 50)
        self.max_iterations = config.get('max_iterations', 10)
        self.max_results_per_query = config.get('max_results_per_query', 100)
        
        self._min_request_interval = 14.0
        self._last_request_time = 0.0
        self._ban_until = 0.0          # epoch time until which we are banned (403)
        
        # Data storage
        self.all_data = []  
        self.seed_pool = set()  
        self.queried_satellites = set()  
        self.tle_cache = {} 
        
    def _rate_limit_wait(self):
        elapsed = time.time() - self._last_request_time
        if elapsed < self._min_request_interval:
            wait = self._min_request_interval - elapsed
            print(f"  [Rate limit] Waiting {wait:.1f}s...")
            time.sleep(wait)
        self._last_request_time = time.time()

    def _wait_if_banned(self):
        remaining = self._ban_until - time.time()
        if remaining > 0:
            print(f"  [Ban active] Waiting {remaining / 3600:.2f}h for ban to expire...")
            time.sleep(remaining)

    def _handle_403(self):
        now = time.time()
        if now < self._ban_until:
            # Already slept for this ban window — skip immediately
            print("  [403 Forbidden] Still within ban window, skipping request.")
            return False
        # New ban: sleep 2 hours once
        self._ban_until = now + 7200
        print(f"  [403 Forbidden] CelesTrak ban detected. Sleeping 2h...")
        time.sleep(7200)
        return True

    def _get_with_retry(self, url):
        self._wait_if_banned()
        try:
            resp = requests.get(url, timeout=30)
            if resp.status_code == 403:
                retried = self._handle_403()
                if not retried:
                    return None
                # Retry once after sleeping
                resp = requests.get(url, timeout=30)
                if resp.status_code == 403:
                    print(f"  [403] Still forbidden after sleep, skipping: {url}")
                    return None
            resp.raise_for_status()
            return resp
        except requests.exceptions.RequestException as e:
            print(f"  Request error: {e}")
            return None

    def get_bin_for_probability(self, pc):
        pc = float(pc)
        for bin_range in self.prob_bins:
            if bin_range[0] <= pc < bin_range[1]:
                return bin_range
        return None
    
    def analyze_distribution(self):
        distribution = defaultdict(list)
        
        for item in self.all_data:
            pc = float(item.get('Pc_gt', 0))
            bin_range = self.get_bin_for_probability(pc)
            if bin_range:
                distribution[bin_range].append(item)
        
        return distribution
    
    def identify_insufficient_bins(self, distribution):
        insufficient = []
        
        for bin_range in self.prob_bins:
            current_count = len(distribution.get(bin_range, []))
            if current_count < self.min_samples_per_bin:
                gap = self.min_samples_per_bin - current_count
                insufficient.append({
                    'bin': bin_range,
                    'current': current_count,
                    'gap': gap,
                    'data': distribution.get(bin_range, [])
                })
        
        return insufficient
    
    def extract_seed_satellites(self, bin_data, strategy='all'):
        satellite_ids = set()
        
        for item in bin_data:
            sat1 = item.get('norad_id_1')
            sat2 = item.get('norad_id_2')
            
            if sat1 and sat1 not in self.queried_satellites:
                satellite_ids.add(sat1)
            if sat2 and sat2 not in self.queried_satellites:
                satellite_ids.add(sat2)
        
        if strategy == 'high_frequency':
            # Count satellite occurrence frequency
            freq = defaultdict(int)
            for sat_id in satellite_ids:
                for item in bin_data:
                    if sat_id in [item.get('norad_id_1'), item.get('norad_id_2')]:
                        freq[sat_id] += 1
            
            # Select top N most frequent satellites
            sorted_sats = sorted(freq.items(), key=lambda x: x[1], reverse=True)
            satellite_ids = set([sat for sat, _ in sorted_sats[:10]])
        
        return satellite_ids
    
    def query_satellite(self, norad_id, max_results=100):
        url = f"https://celestrak.org/SOCRATES/table-socrates.php?CATNR={norad_id},&ORDER=MAXPROB&MAX={max_results}"
        
        try:
            print(f"Querying satellite {norad_id}...", end='')
            resp = self._get_with_retry(url)
            if resp is None:
                print(" ✗ Failed")
                return []
            tables = pd.read_html(io.StringIO(resp.text))
            
            if tables and len(tables) > 3:
                df = tables[3].values
                result, _ = self.parse_celestrak_data(df)
                print(f" Retrieved {len(result)} events")
                return result
            else:
                print(" ✗ No data")
                return []
                
        except Exception as e:
            print(f" Failed: {e}")
            return []
    
    def parse_celestrak_data(self, df):
        if df is None or len(df) == 0:
            return [], []
        
        n = df.shape[0]
        pair_count = (n + 1) // 2
        result = [{} for _ in range(pair_count)]
        sat_ids = []
        
        def safe_get(r, i):
            try:
                return r[i]
            except:
                return None
        
        for index, row in enumerate(df):
            norad_id = safe_get(row, 1)
            sat_name = safe_get(row, 2)
            if isinstance(norad_id, int):
                norad_id = str(norad_id)
            if not norad_id or len(str(norad_id)) > 5:
                continue
            
            if norad_id not in sat_ids:
                sat_ids.append(norad_id)
            
            tca = safe_get(row, 4)
            num_index = index // 2
            
            if index % 2 != 0:
                Pc_gt = safe_get(row, 5)
                result[num_index].update({
                    "norad_id_2": norad_id,
                    "sat_name_2": sat_name,
                    "TCA": tca,
                    "Pc_gt": Pc_gt
                })
            else:
                result[num_index].update({
                    "norad_id_1": norad_id,
                    "sat_name_1": sat_name
                })
        
        return result, sat_ids
    
    def initialize_seed_pool(self):
        print("\n" + "="*60)
        print("Initializing Seed Pool")
        print("="*60)
        
        # Get Top10 high-risk satellites
        url = "https://celestrak.org/SOCRATES/table-socrates.php?NAME=,&ORDER=MAXPROB&MAX=10"
        try:
            resp = self._get_with_retry(url)
            if resp:
                tables = pd.read_html(io.StringIO(resp.text))
                if tables and len(tables) > 3:
                    df = tables[3].values
                    initial_data, sat_ids = self.parse_celestrak_data(df)
                    
                    self.all_data.extend(initial_data)
                    self.seed_pool.update(sat_ids)
                    
                    print(f"Initial seed pool: {len(self.seed_pool)} satellites")
                    print(f"Initial data: {len(initial_data)} events")
                
        except Exception as e:
            print(f"Initialization failed: {e}")
    
    def expand_from_seeds(self, seed_ids):
        new_data = []
        
        for sat_id in seed_ids:
            if sat_id in self.queried_satellites:
                continue
            
            data = self.query_satellite(sat_id, self.max_results_per_query)
            new_data.extend(data)
            self.queried_satellites.add(sat_id)
            
            # Extract new satellite IDs and add to seed pool
            for item in data:
                for key in ['norad_id_1', 'norad_id_2']:
                    new_sat = item.get(key)
                    if new_sat:
                        self.seed_pool.add(new_sat)
        
        return new_data
    
    def print_distribution(self, distribution):
        print("\nCurrent Data Distribution:")
        print("-" * 70)
        print(f"{'Probability Range':<35} {'Count':<10} {'Status':<15}")
        print("-" * 70)
        
        total = 0
        for bin_range in self.prob_bins:
            count = len(distribution.get(bin_range, []))
            total += count
            status = "Sufficient" if count >= self.min_samples_per_bin else f"Need {self.min_samples_per_bin - count}"
            print(f"[{bin_range[0]:.2e}, {bin_range[1]:.2e})"
                  f"{'':<15} {count:<10} {status}")
        
        print("-" * 70)
        print(f"Total: {total} records | Queried satellites: {len(self.queried_satellites)}")
        print("-" * 70)
    
    def deduplicate(self):
        seen = set()
        unique_data = []
        
        for item in self.all_data:
            key = (
                item.get('norad_id_1'),
                item.get('norad_id_2'),
                item.get('TCA')
            )
            
            if key not in seen:
                seen.add(key)
                unique_data.append(item)
        
        removed = len(self.all_data) - len(unique_data)
        if removed > 0:
            print(f"  Deduplication: Removed {removed} duplicate records")
        
        self.all_data = unique_data

    def fetch_tle(self, norad_id):
        enable_rate_limit_wait = False
        norad_id = str(norad_id).strip()
        if norad_id in self.tle_cache:
            return self.tle_cache[norad_id]

        url = f"https://celestrak.org/NORAD/elements/gp.php?CATNR={norad_id}&FORMAT=TLE"
        max_retries = 5
        for attempt in range(max_retries):
            try:  
                if enable_rate_limit_wait:
                    self._rate_limit_wait()
                self._wait_if_banned()
                resp = requests.get(url, timeout=15)

                if resp.status_code == 403:
                    retried = self._handle_403()
                    if not retried:
                        self.tle_cache[norad_id] = None
                        return None
                    continue

                resp.raise_for_status()
                lines = [l for l in resp.text.strip().splitlines() if l.strip()]
                if len(lines) >= 2:
                    tle_lines = [l for l in lines if l.startswith('1 ') or l.startswith('2 ')]
                    if len(tle_lines) >= 2:
                        result = [tle_lines[0], tle_lines[1]]
                        self.tle_cache[norad_id] = result
                        return result
                self.tle_cache[norad_id] = None
                return None

            except (requests.exceptions.ConnectionError,
                    requests.exceptions.Timeout) as e:
                enable_rate_limit_wait = True
                wait_time = 30 * (attempt + 1)
                if attempt < max_retries - 1:
                    print(f"  Timeout for {norad_id}, retrying in {wait_time}s...")
                    time.sleep(wait_time)
                else:
                    print(f"  Failed TLE for {norad_id} after {max_retries} attempts: {e}")
                    self.tle_cache[norad_id] = None
                    return None

            except Exception as e:
                enable_rate_limit_wait = True
                print(f"  Failed to fetch TLE for {norad_id}: {e}")
                self.tle_cache[norad_id] = None
                return None

        self.tle_cache[norad_id] = None
        return None

    def fetch_all_tles(self):
        sat_ids = set()
        for item in self.all_data:
            sat_ids.add(str(item.get('norad_id_1', '')).strip())
            sat_ids.add(str(item.get('norad_id_2', '')).strip())
        sat_ids.discard('')

        to_fetch = [sid for sid in sat_ids if sid not in self.tle_cache]
        print(f"\nFetching TLE data for {len(to_fetch)} satellites "
              f"({len(sat_ids) - len(to_fetch)} already cached)...")

        for i, sid in enumerate(to_fetch, 1):
            if i % 20 == 0 or i == len(to_fetch):
                success_so_far = sum(1 for v in self.tle_cache.values() if v is not None)
                print(f"  Progress: {i}/{len(to_fetch)} (success: {success_so_far})")
            self.fetch_tle(sid)

        success = sum(1 for v in self.tle_cache.values() if v is not None)
        print(f"  TLE cache: {success} satellites with data, "
              f"{len(self.tle_cache) - success} failed")

    def format_output(self):
        output = []
        skipped = 0

        for item in self.all_data:
            norad_1 = str(item.get('norad_id_1', '')).strip()
            norad_2 = str(item.get('norad_id_2', '')).strip()
            tle_1 = self.tle_cache.get(norad_1)
            tle_2 = self.tle_cache.get(norad_2)

            if tle_1 is None or tle_2 is None:
                skipped += 1
                continue
            pc = item.get('Pc_gt', 0)
            try:
                pc_str = f"{float(pc):.3E}"
            except (ValueError, TypeError):
                pc_str = str(pc)

            output.append({
                "TCA": item.get('TCA', ''),
                "pc_gt": pc_str,
                "sat_1_name": item.get('sat_name_1', ''),
                "sat_1": tle_1,
                "sat_2_name": item.get('sat_name_2', ''),
                "sat_2": tle_2
            })

        if skipped:
            print(f"  Skipped {skipped} events due to missing TLE data")
        print(f"  Final output: {len(output)} events with TLE data")
        return output
    
    def save_results(self, timestamp=None):
        self.fetch_all_tles()
        formatted = self.format_output()
        path = f"data/collected_tle_datas_{timestamp}.json"
        with open(path, "w") as f:
            json.dump(formatted, f, indent=4)
        print(f"  Checkpoint saved: {path} ({len(formatted)} events)")
        return formatted

    def collect(self):
        print("\n" + "="*60)
        print("Starting Smart Adaptive Data Collection")
        print("="*60)

        self.initialize_seed_pool()

        print(f"\nExpanding initial seed pool...")
        initial_expansion = self.expand_from_seeds(list(self.seed_pool))
        self.all_data.extend(initial_expansion)
        self.deduplicate()
        timestamp = time.strftime("%Y%m%d-%H%M%S")
        for iteration in range(1, self.max_iterations + 1):
            print(f"\n{'='*60}")
            print(f"Iteration {iteration}")
            print(f"{'='*60}")
            
            # Analyze distribution
            distribution = self.analyze_distribution()
            self.print_distribution(distribution)
            
            # Identify insufficient bins
            insufficient = self.identify_insufficient_bins(distribution)
            
            if not insufficient:
                print("\n All bins have sufficient data!")
                break
            
            print(f"\nFound {len(insufficient)} bins that need supplementation")
            
            # Collect new seeds for each insufficient bin
            new_seeds = set()
            for bin_info in insufficient:
                bin_range = bin_info['bin']
                bin_data = bin_info['data']
                gap = bin_info['gap']
                
                print(f"\nBin [{bin_range[0]:.2e}, {bin_range[1]:.2e}): "
                      f"Need {gap} records")
                
                # Extract new seeds from this bin
                seeds = self.extract_seed_satellites(bin_data, strategy='all')
                new_seeds.update(seeds)
                print(f"  Extracted {len(seeds)} new seed satellites")
            
            if not new_seeds:
                print("\n Cannot extract more seeds, trying unqueried seed pool...")
                # Use unqueried satellites from seed pool
                unqueried = self.seed_pool - self.queried_satellites
                if unqueried:
                    new_seeds = set(list(unqueried)[:20])  
                else:
                    print("✗ Seed pool exhausted")
                    break
            
            # Save checkpoint before this iteration's expansion
            print(f"\nSaving checkpoint before iteration {iteration} expansion...")
            self.save_results(timestamp=timestamp)

            # Expand new seeds
            print(f"\nQuerying {len(new_seeds)} new satellites...")
            new_data = self.expand_from_seeds(new_seeds)
            self.all_data.extend(new_data)
            self.deduplicate()
            
            print(f"Added {len(new_data)} raw records this round")
        
        # Final report
        print(f"\n{'='*60}")
        print("Collection Complete")
        print(f"{'='*60}")
        
        final_distribution = self.analyze_distribution()
        self.print_distribution(final_distribution)

        # Fetch TLE data and format output
        print(f"\n{'='*60}")
        print("Fetching TLE Data from CelesTrak")
        print(f"{'='*60}")
        self.fetch_all_tles()

        print(f"\n{'='*60}")
        print("Formatting Output")
        print(f"{'='*60}")
        formatted = self.format_output()
        
        return formatted

if __name__ == "__main__":
    config = {
        'prob_bins': [
            (1e-10, 1e-8),
            (1e-8, 1e-6),
            (1e-6, 1e-5),
            (1e-5, 1e-4),
            (1e-4, 1e-3),
            (1e-3, 1e-2),
            (1e-2, 0.0)
        ],
        'min_samples_per_bin': 30,  
        'max_iterations': 8,
        'max_results_per_query': 100
    }
    
    collector = DataCollectionPipeline(config)
    data = collector.collect()
    
    
    print(f"Total {len(data)} collision events with TLE data")