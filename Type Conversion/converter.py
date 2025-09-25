import argparse
from pathlib import Path
from tqdm import tqdm
import numpy as np
import pandas as pd
import uproot
import awkward as ak
import h5py as h5
from typing import List, Dict, Any, Tuple, Optional
import warnings

def convert_calo_hits(input_file, output_file, n_events=None, calo_type="All"):
    """
    Convert calorimeter hits + max contribution info from ROOT to HDF5.
    
    Args:
        input_file (str): Path to input ROOT file
        output_file (str): Path to output HDF5 file
        n_events (int, optional): Number of events to process. If None, process all.
        calo_type (str): "ECal", "HCal", or "All"
    """
    
    collections = {}
    if calo_type in ["ECal", "All"]:
        collections.update({
            "ECalBarrelCollection": 1,
            "ECalEndcapCollection": 2
        })
    if calo_type in ["HCal", "All"]:
        collections.update({
            "HCalBarrelCollection": 3,
            "HCalEndcapCollection": 4
        })

    with uproot.open(input_file) as file, h5.File(output_file, "w") as h5f:
        tree = file["events;33"]
        total_events = tree.num_entries
        if n_events is None or n_events > total_events:
            n_events = total_events

        for iev in range(n_events):
            event_group = h5f.create_group(f"events/event{iev}")

            all_cellID = []
            all_energy = []
            all_pos_x  = []
            all_pos_y  = []
            all_pos_z  = []
            all_region = []
            all_max_e  = []
            all_max_t  = []

            for coll, region_label in collections.items():
                contrib_name = coll + "Contributions"

                try:
                    cellID = tree[f"{coll}/{coll}.cellID"].array(entry_start=iev, entry_stop=iev+1)[0]
                except KeyError:
                    continue

                energy = tree[f"{coll}/{coll}.energy"].array(entry_start=iev, entry_stop=iev+1)[0]
                pos_x  = tree[f"{coll}/{coll}.position.x"].array(entry_start=iev, entry_stop=iev+1)[0]
                pos_y  = tree[f"{coll}/{coll}.position.y"].array(entry_start=iev, entry_stop=iev+1)[0]
                pos_z  = tree[f"{coll}/{coll}.position.z"].array(entry_start=iev, entry_stop=iev+1)[0]

                contrib_begin = tree[f"{coll}/{coll}.contributions_begin"].array(entry_start=iev, entry_stop=iev+1)[0]
                contrib_end   = tree[f"{coll}/{coll}.contributions_end"].array(entry_start=iev, entry_stop=iev+1)[0]

                contrib_energy = tree[f"{contrib_name}/{contrib_name}.energy"].array(entry_start=iev, entry_stop=iev+1)[0]
                contrib_time   = tree[f"{contrib_name}/{contrib_name}.time"].array(entry_start=iev, entry_stop=iev+1)[0]

                max_contrib_energy = []
                max_contrib_time   = []

                for begin, end in zip(contrib_begin, contrib_end):
                    contribs_e = contrib_energy[begin:end]
                    contribs_t = contrib_time[begin:end]
                    if len(contribs_e) > 0:
                        idx = np.argmax(contribs_e)
                        max_contrib_energy.append(contribs_e[idx])
                        max_contrib_time.append(contribs_t[idx])
                    else:
                        max_contrib_energy.append(0.0)
                        max_contrib_time.append(0.0)

                all_cellID.extend(cellID)
                all_energy.extend(energy)
                all_pos_x.extend(pos_x)
                all_pos_y.extend(pos_y)
                all_pos_z.extend(pos_z)
                all_region.extend([region_label]*len(cellID))
                all_max_e.extend(max_contrib_energy)
                all_max_t.extend(max_contrib_time)

            event_group.create_dataset("cellID", data=np.asarray(all_cellID), compression="gzip")
            event_group.create_dataset("energy", data=np.asarray(all_energy), compression="gzip")
            event_group.create_dataset("pos_x", data=np.asarray(all_pos_x), compression="gzip")
            event_group.create_dataset("pos_y", data=np.asarray(all_pos_y), compression="gzip")
            event_group.create_dataset("pos_z", data=np.asarray(all_pos_z), compression="gzip")
            event_group.create_dataset("region_label", data=np.asarray(all_region), compression="gzip")
            event_group.create_dataset("max_energy_contribution", data=np.asarray(all_max_e), compression="gzip")
            event_group.create_dataset("time_of_max_energy_contribution", data=np.asarray(all_max_t), compression="gzip")

            print(f"Stored event {iev} with {len(all_cellID)} hits from {calo_type}")

    print(f"\n Saved {n_events} events to {output_file}")


def convert_mcparticles(input_file, output_file, n_events=None):
    """
    Convert MCParticles from ROOT to HDF5 with per-event structure.
    Includes daughters as ragged arrays reconstructed from begin/end + index.
    """
    with uproot.open(input_file) as file, h5.File(output_file, "w") as h5f:
        tree = file["events;33"]

        # Get all MCParticles branches (skip .fBits)
        mc_branches = [
            b for b in tree.keys()
            if b.startswith("MCParticles/") and not b.endswith(".fBits")
        ]

        # Daughters arrays
        daughters_begin = tree["MCParticles/MCParticles.daughters_begin"].array()
        daughters_end   = tree["MCParticles/MCParticles.daughters_end"].array()
        daughters_index = tree["_MCParticles_daughters/_MCParticles_daughters.index"].array()

        total_events = tree.num_entries
        if n_events is None or n_events > total_events:
            n_events = total_events

        for iev in range(n_events):
            event_group = h5f.create_group(f"events/event{iev}")

            # Store all standard MCParticles branches
            for branch in mc_branches:
                data = tree[branch].array(entry_start=iev, entry_stop=iev+1)[0]
                name = branch.split("/")[-1]  # e.g. MCParticles.energy
                event_group.create_dataset(name, data=np.asarray(data), compression="gzip")

            # Now build daughters mapping
            begins = daughters_begin[iev]
            ends   = daughters_end[iev]
            indices = daughters_index[iev]

            daughters_list = []
            for b, e in zip(begins, ends):
                daughters_list.append(indices[b:e].tolist())

            # Store as variable-length dataset
            dt = h5.special_dtype(vlen=np.int64)
            event_group.create_dataset("daughters", (len(daughters_list),), dtype=dt)
            for i, dlist in enumerate(daughters_list):
                event_group["daughters"][i] = dlist

            print(f"Stored event {iev} with {len(begins)} particles")

    print(f"\nSaved {n_events} events to {output_file}")
    
def root_to_h5_tracker(input_root, output_h5, num_events, selected_detectors):
    """
    Converts selected ROOT tracker branches to an HDF5 file in the format:
    events/event0, events/event1, ...

    Each event is stored as a group with datasets:
        - cellID, eDep, time, pathLength, quality,
          position.x, position.y, position.z,
          momentum.x, momentum.y, momentum.z
        - region_label  (int, one per hit)

    Parameters
    ----------
    input_root : str
        Path to input ROOT file.
    output_h5 : str
        Path to output HDF5 file to create.
    num_events : int
        Number of events to process.
    selected_detectors : dict
        Dictionary mapping detector name → integer label, e.g.
        {
            "PixelBarrelReadout": 1,
            "PixelEndcapReadout": 2,
            "LongStripBarrelReadout": 3
        }
    """

    hit_features = [
        "cellID",
        "eDep",
        "time",
        "pathLength",
        "quality",
        "position.x",
        "position.y",
        "position.z",
        "momentum.x",
        "momentum.y",
        "momentum.z",
    ]

    with uproot.open(input_root) as file:
        tree = file["events;33"]

        
        total_entries = tree.num_entries
        n_events = min(num_events, total_entries)
        print(f"Processing {n_events} events (requested {num_events}, available {total_entries})")

        with h5.File(output_h5, "w") as h5f:
            for evt_idx in range(n_events):
                all_hits = {feat: [] for feat in hit_features}
                all_labels = []

                
                for det_name, label in selected_detectors.items():
                    valid = True
                    cols = {}

                    for feat in hit_features:
                        branch = f"{det_name}.{feat}"
                        if branch not in tree:
                            valid = False
                            break
                        arr = tree[branch].array(entry_start=evt_idx, entry_stop=evt_idx+1)

                        if len(arr) == 0 or len(arr[0]) == 0:
                            valid = False
                            break
                        cols[feat] = ak.to_numpy(arr[0])

                    if not valid:
                        continue

                    n_hits = len(next(iter(cols.values())))
                    for feat in hit_features:
                        all_hits[feat].extend(cols[feat])
                    all_labels.extend([label] * n_hits)

            
                event_group = h5f.create_group(f"events/event{evt_idx}")

                for feat in hit_features:
                    event_group.create_dataset(feat, data=np.asarray(all_hits[feat]), compression="gzip")

                event_group.create_dataset("region_label", data=np.asarray(all_labels, dtype=np.int32), compression="gzip")

                print(f"Stored event {evt_idx} with {len(all_labels)} hits from {len(selected_detectors)} detectors")

        print(f"\nSaved {n_events} events to {output_h5}")
        
def inspect_h5_file(h5_file, n_events=5, n_hits=5):
    """
    Inspect the structure and some values of the HDF5 calorimeter hits file.

    Args:
        h5_file (str): Path to the HDF5 file
        n_events (int): Number of events to inspect
        n_hits (int): Number of hits per event to print
    """
    with h5.File(h5_file, "r") as f:
        if "events" not in f:
            print("No 'events' group found in file!")
            return

        events_group = f["events"]
        event_keys = list(events_group.keys())
        n_events_file = len(event_keys)
        print(f"File contains {n_events_file} events.")
        n_events_to_print = min(n_events, n_events_file)

        for iev in range(n_events_to_print):
            event_name = f"event{iev}"
            if event_name not in events_group:
                continue
            event_group = events_group[event_name]
            print(f"\n--- events/{event_name} ---")
            for dset_name in event_group.keys():
                data = event_group[dset_name]
                print(f"Dataset '{dset_name}': shape={data.shape}, dtype={data.dtype}")
                preview = np.array(data[:n_hits])
                print(f"  First {n_hits} entries: {preview}")

