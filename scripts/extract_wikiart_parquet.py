import os, glob, io, time
import pyarrow.parquet as pq
from PIL import Image

print("=== EXTRACTING WIKIART PARQUET IMAGES TO EXTRACTED POOL ===")
out_dir = "/mnt/ai-storage/aigc_data/datasets/extracted_parquet_pool/wikiart_real"
os.makedirs(out_dir, exist_ok=True)

parquet_files = sorted(glob.glob("/mnt/ai-storage/aigc_data/datasets/wikiart_hard_negatives/data/*.parquet"))
print(f"Found {len(parquet_files)} WikiArt Parquet files.")

total_extracted = 0
start_t = time.time()

for p_idx, pf in enumerate(parquet_files):
    table = pq.read_table(pf)
    n_rows = len(table)
    img_col = table["image"]
    
    for r in range(n_rows):
        row_dict = img_col[r].as_py()
        if isinstance(row_dict, dict) and "bytes" in row_dict:
            img_bytes = row_dict["bytes"]
        elif isinstance(row_dict, list):
            # list of tuples
            d = dict(row_dict)
            img_bytes = d.get("bytes", None)
        else:
            img_bytes = None
            
        if img_bytes:
            out_name = f"wikiart_extracted_{p_idx:03d}_{r:05d}.jpg"
            out_p = os.path.join(out_dir, out_name)
            if not os.path.exists(out_p):
                with open(out_p, "wb") as f:
                    f.write(img_bytes)
            total_extracted += 1
            
    if (p_idx + 1) % 10 == 0 or (p_idx + 1) == len(parquet_files):
        print(f"  Processed {p_idx+1:2d}/{len(parquet_files)} files | Total Extracted: {total_extracted:,d} | Time: {time.time()-start_t:.1f}s")

print(f"WikiArt Extraction Complete! Total Extracted: {total_extracted:,d} images.")
