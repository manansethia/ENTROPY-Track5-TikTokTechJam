import os, glob, io, time
from concurrent.futures import ProcessPoolExecutor
import pyarrow.parquet as pq

out_dir = "/mnt/ai-storage/aigc_data/datasets/extracted_parquet_pool/wikiart_real"
os.makedirs(out_dir, exist_ok=True)

parquet_files = sorted(glob.glob("/mnt/ai-storage/aigc_data/datasets/wikiart_hard_negatives/data/*.parquet"))

def process_file(args):
    p_idx, pf = args
    table = pq.read_table(pf)
    n_rows = len(table)
    img_col = table["image"]
    count = 0
    for r in range(n_rows):
        row_dict = img_col[r].as_py()
        if isinstance(row_dict, dict) and "bytes" in row_dict:
            img_bytes = row_dict["bytes"]
        elif isinstance(row_dict, list):
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
            count += 1
    return count

if __name__ == "__main__":
    t0 = time.time()
    args_list = [(i, f) for i, f in enumerate(parquet_files)]
    with ProcessPoolExecutor(max_workers=8) as executor:
        results = list(executor.map(process_file, args_list))
    total = sum(results)
    print(f"Parallel Extraction Finished in {time.time()-t0:.2f}s! Total images: {total:,d}")
