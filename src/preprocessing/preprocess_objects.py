import zipfile
import json
import pandas as pd
from pathlib import Path
from tqdm import tqdm
from collections import defaultdict
import gc

def process_objects(raw_dir: Path, out_dir: Path, frames_df: pd.DataFrame):
    obj_zips = list(raw_dir.glob("objects*.zip"))
    if not obj_zips:
        raise FileNotFoundError(f"Không tìm thấy objects*.zip trong {raw_dir}")
    obj_zip_path = obj_zips[0]
    obj_out_dir = out_dir / "objects"
    obj_out_dir.mkdir(parents=True, exist_ok=True)
    
    vid_to_n_to_global = defaultdict(dict)
    for _, row in frames_df.iterrows():
        vid_to_n_to_global[row['video_id']][row['keyframe_index']] = row['global_id']
        
    summary_data = []
    
    # We will process batch by batch (L21, K01...) to avoid memory overflow
    current_batch_prefix = None
    batch_data = []
    
    with zipfile.ZipFile(obj_zip_path, 'r') as z:
        json_files = sorted([f for f in z.namelist() if f.endswith('.json')])
        
        for f in tqdm(json_files, desc="Processing Object JSONs"):
            parts = f.split('/')
            if len(parts) < 2:
                continue
                
            # Usually path is something like `objects/L21_V001/024.json` or `objects/K01_V001/024.json`
            video_id = parts[-2]
            if '_' not in video_id:
                video_id = parts[0]
                
            stem = Path(f).stem
            try:
                keyframe_index = int(stem)
            except ValueError:
                continue
                
            global_id = vid_to_n_to_global.get(video_id, {}).get(keyframe_index)
            
            if global_id is None:
                continue
                
            batch_prefix = video_id.split('_')[0] # e.g., 'L21'
            
            if current_batch_prefix is not None and batch_prefix != current_batch_prefix:
                # Save current batch
                if batch_data:
                    df = pd.DataFrame(batch_data)
                    df.to_parquet(obj_out_dir / f"{current_batch_prefix}.parquet", index=False)
                    batch_data = []
                    gc.collect()
            
            current_batch_prefix = batch_prefix
            
            with z.open(f) as jf:
                data = json.load(jf)
                
            scores = data.get("detection_scores", [])
            names = data.get("detection_class_names", [])
            entities = data.get("detection_class_entities", [])
            boxes = data.get("detection_boxes", [])
            labels = data.get("detection_class_labels", [])
            
            person_count = 0
            entity_counts = defaultdict(int)
            high_conf_entities = set()
            
            for i in range(len(scores)):
                score = float(scores[i])
                entity = entities[i]
                
                # Keep objects with some confidence for the big parquet to save space?
                # BTC says don't drop raw info if possible, but let's just keep everything or > 0.05
                # For safety, keep all 100
                
                box = [float(x) for x in boxes[i]]
                
                batch_data.append({
                    "global_id": global_id,
                    "video_id": video_id,
                    "keyframe_index": keyframe_index,
                    "detection_rank": i,
                    "entity": entity,
                    "class_name": names[i],
                    "class_label": int(labels[i]),
                    "score": score,
                    "bbox_0": box[0],
                    "bbox_1": box[1],
                    "bbox_2": box[2],
                    "bbox_3": box[3]
                })
                
                if score > 0.3:
                    if entity.lower() in ['person', 'man', 'woman', 'human face', 'boy', 'girl']:
                        person_count += 1
                    entity_counts[entity] += 1
                    
                if score > 0.7:
                    high_conf_entities.add(entity)
                    
            # Top entities
            top_entities = [e for e, c in sorted(entity_counts.items(), key=lambda x: x[1], reverse=True)[:5]]
            
            summary_data.append({
                "global_id": global_id,
                "person_count": person_count,
                "top_entities": top_entities,
                "high_conf_entities": list(high_conf_entities)
            })
            
    # Save the last batch
    if batch_data:
        df = pd.DataFrame(batch_data)
        df.to_parquet(obj_out_dir / f"{current_batch_prefix}.parquet", index=False)
        
    summary_df = pd.DataFrame(summary_data)
    summary_df.to_parquet(out_dir / "object_summary.parquet", index=False)
    print(f"Saved object summary to {out_dir / 'object_summary.parquet'}")
