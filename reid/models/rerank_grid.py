"""Fine k-reciprocal grid search offline on a saved raw_embeddings.npy."""
from __future__ import annotations
import argparse, csv, json, os, sys
import numpy as np
from reid import rerank
from reid.data.io import TRAIN_COLUMNS, read_csv
from reid.data.splits import holdout_val
from reid.eval.harness import run_official


def flat(rep):
    r = rep.get("ranking", {})
    return {"mAP@10": r.get("mAP@10"), "Rank-1": r.get("Rank-1"), "Rank-5": r.get("Rank-5")}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", required=True)
    ap.add_argument("--dataset", required=True)
    ap.add_argument("--json", default=None)
    args = ap.parse_args()
    out = os.path.abspath(args.run)
    emb = np.load(os.path.join(out, "raw_embeddings.npy"))
    df = read_csv(os.path.join(args.dataset, "train.csv"), required=TRAIN_COLUMNS)
    _, vq, vg = holdout_val(df, val_fraction=0.2, open_set_fraction=0.2, seed=42)
    vq = vq.reset_index(drop=True); vg = vg.reset_index(drop=True)
    nq = len(vq); q, g = emb[:nq], emb[nq:]
    q_ids = vq["image_id"].tolist(); g_ids = vg["image_id"].tolist()
    import pandas as pd
    gt = os.path.join(out, "gt.csv")
    a = vq[["image_id", "vehicle_id", "camera_id"]].copy(); a["split"] = "query"
    b = vg[["image_id", "vehicle_id", "camera_id"]].copy(); b["split"] = "gallery"
    pd.concat([a, b], ignore_index=True).to_csv(gt, index=False)
    qcsv = os.path.join(out, "query.csv"); gcsv = os.path.join(out, "gallery.csv")
    vq.to_csv(qcsv, index=False); vg.to_csv(gcsv, index=False)
    sub = os.path.join(out, "submission.csv"); cand = os.path.join(out, "candidates.csv")
    np.save(os.path.join(out, "embeddings.npy"), emb)
    results = []
    for pool in (100, 200):
      preps = [rerank.prepare_query(q[i], g, pool_size=pool) for i in range(nq)]
      for k1 in (6, 8, 10, 12, 14):
        for k2 in (3, 4):
            for lam in (0.5, 0.6, 0.7):
                o = np.empty((nq, len(g)), np.int64); sc = np.empty((nq, len(g)), np.float32)
                for i in range(nq):
                    o[i], sc[i] = rerank.rank_prepared(preps[i], k1=k1, k2=k2, lam=lam)
                with open(sub, "w", encoding="utf-8", newline="") as f:
                    w = csv.writer(f)
                    for i in range(nq):
                        w.writerow([q_ids[i]] + [g_ids[j] for j in o[i, :10]])
                with open(cand, "w", encoding="utf-8", newline="") as f:
                    w = csv.writer(f); w.writerow(["query_id", "gallery_id", "confidence"])
                    for i in range(nq):
                        j = int(o[i, 0]); w.writerow([q_ids[i], g_ids[j], f"{float(sc[i,0]):.6f}"])
                m = flat(run_official(gt_csv=gt, submission=sub, candidates=cand,
                                      embeddings=os.path.join(out, "embeddings.npy"),
                                      query=qcsv, gallery=gcsv,
                                      json_out=os.path.join(out, f"official_grid_p{pool}_{k1}_{k2}_{lam}.json")))
                results.append({"pool": pool, "k1": k1, "k2": k2, "lam": lam, **m})
                print(f"pool={pool} k1={k1} k2={k2} lam={lam}: mAP@10={m['mAP@10']:.4f} R1={m['Rank-1']:.4f}", flush=True)
    best = max(results, key=lambda r: r["mAP@10"] or 0)
    print("BEST", json.dumps(best))
    if args.json:
        with open(args.json, "w", encoding="utf-8") as f:
            json.dump({"results": results, "best": best}, f, ensure_ascii=False, indent=2)


if __name__ == "__main__":
    sys.exit(main())
