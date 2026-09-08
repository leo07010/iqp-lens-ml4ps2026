import os, sys, time, json
import numpy as np
REPO="<HOME>/gsoc2026-deeplense-qml/experiments/13_iqp_turbulence_replication"
sys.path.insert(0, REPO)
import codec, data_d2
D=os.path.dirname(os.path.abspath(__file__))
t0=time.time()
snaps, anchors, holdout = data_d2.load_d2("mid")
a_z,b_z = data_d2.get_z_range("mid")
ts = sorted(snaps.keys())
print("loaded", len(ts), "snaps in %.1fs"%(time.time()-t0), "a_z,b_z=",a_z,b_z)
fields = np.stack([snaps[t].field for t in ts]).astype(np.float32)
pools  = np.stack([snaps[t].pool_codes for t in ts]).astype(np.int32)
np.save(D+"/gt_fields.npy", fields)
np.save(D+"/pools.npy", pools)
np.save(D+"/ts.npy", np.array(ts,dtype=np.int64))
json.dump({"a_z":a_z,"b_z":b_z,"anchors":list(map(int,anchors)),"holdout":list(map(int,holdout))}, open(D+"/meta_prep.json","w"))
print("pools shape", pools.shape, "fields", fields.shape, "%.1fs"%(time.time()-t0))
