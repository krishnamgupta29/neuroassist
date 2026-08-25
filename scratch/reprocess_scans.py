import asyncio
import os
import sys

sys.path.insert(0, os.path.abspath('backend'))
from database import scans_col, init_db
from ml.inference import run_inference

async def reprocess_all():
    await init_db()
    scans = await scans_col.find({}).to_list(100)
    print(f"Reprocessing {len(scans)} scans in MongoDB...")
    for s in scans:
        fp = s.get('file_path')
        if not fp:
            continue
        if not os.path.exists(fp):
            alt = os.path.join('backend', fp)
            if os.path.exists(alt):
                fp = alt
        orig = s.get('original_filename', '')
        scan_id = s.get('scan_id_string')
        if os.path.exists(fp):
            try:
                res = run_inference(fp, original_filename=orig)
                pred = res['prediction']
                risk = res['risk_score']
                await scans_col.update_one(
                    {'_id': s['_id']},
                    {'$set': {
                        'prediction': pred,
                        'confidence_cn': res['confidence_cn'],
                        'confidence_mci': res['confidence_mci'],
                        'confidence_ad': res['confidence_ad'],
                        'conf_cn': res['confidence_cn'],
                        'conf_mci': res['confidence_mci'],
                        'conf_ad': res['confidence_ad'],
                        'risk_score': risk,
                        'urgency': res['urgency'],
                        'biomarker_hippocampal': res['biomarkers']['hippocampal_atrophy'],
                        'biomarker_amyloid': res['biomarkers']['amyloid_plaque_load'],
                        'biomarker_ventricle': res['biomarkers']['ventricle_enlargement'],
                        'brain_regions': res['brain_regions'],
                        'model_used': 'multiclass',
                        'status': 'analyzed'
                    }}
                )
                print(f"  {scan_id} ({orig}): -> {pred} (Risk: {risk}%)")
            except Exception as e:
                print(f"  Failed {scan_id}: {e}")
        else:
            print(f"  File not found for {scan_id}: {fp}")

if __name__ == '__main__':
    asyncio.run(reprocess_all())
