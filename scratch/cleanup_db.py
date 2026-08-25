import sys
import asyncio
import os

sys.path.insert(0, 'backend')
from database import scans_col, patients_col
from ml.inference import run_inference

async def main():
    all_scans = await scans_col.find({}).to_list(100)
    print(f"Total scans in db: {len(all_scans)}")
    for s in all_scans:
        fpath = s.get('file_path', '')
        # Check backend relative path
        real_path = os.path.join('backend', fpath) if not os.path.exists(fpath) else fpath
        if not os.path.exists(real_path):
            print(f"Removing unlinked scan: {s.get('scan_id_string')}")
            await scans_col.delete_one({'_id': s['_id']})
        else:
            orig_fn = s.get('original_filename', '')
            res = run_inference(real_path, model_type='multiclass', original_filename=orig_fn)
            print(f"Scan {s.get('scan_id_string')} ({orig_fn}) -> {res['prediction']}")
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

    print("All scans synchronized cleanly!")

asyncio.run(main())
