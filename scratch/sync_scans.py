import sys
import asyncio
import os

sys.path.insert(0, 'backend')
from database import scans_col
from ml.inference import run_inference

async def main():
    scans = await scans_col.find({}).to_list(100)
    print(f"Syncing {len(scans)} scans in database...")
    for s in scans:
        fpath = s.get('file_path')
        orig_fn = s.get('original_filename', '')
        scan_id = s.get('scan_id_string')
        if fpath and os.path.exists(fpath):
            res = run_inference(fpath, model_type='multiclass', original_filename=orig_fn)
            print(f"Scan {scan_id} ({orig_fn}) -> {res['prediction']} (Risk: {res['risk_score']}%)")
            await scans_col.update_one(
                {'_id': s['_id']},
                {'$set': {
                    'prediction': res['prediction'],
                    'conf_cn': res['confidence_cn'],
                    'conf_mci': res['confidence_mci'],
                    'conf_ad': res['confidence_ad'],
                    'risk_score': res['risk_score'],
                    'status': 'analyzed'
                }}
            )

asyncio.run(main())
