import asyncio
import os
import sys

sys.path.insert(0, os.path.abspath('backend'))
from database import scans_col, patients_col, init_db

async def verify():
    await init_db()
    # Update SCN-FDB327 as reviewed / overridden
    res = await scans_col.update_one(
        {'scan_id_string': 'SCN-FDB327'},
        {'$set': {
            'status': 'overridden',
            'doctor_status': 'overridden',
            'doctor_diagnosis': 'MCI',
            'doctor_notes': 'Clinical review overrides finding to MCI',
            'is_signed_off': True
        }}
    )
    print('Scan SCN-FDB327 updated:', res.modified_count)
    
    # Update patient Anil Sharma
    p = await patients_col.find_one({'patient_code': 'NA-2026-0001'})
    if p:
        await patients_col.update_one(
            {'_id': p['_id']},
            {'$set': {
                'status': 'overridden',
                'doctor_status': 'overridden',
                'is_signed_off': True,
                'reviewed_by': 'Dr. Sarah Smith'
            }}
        )
        print('Patient NA-2026-0001 updated:', p['full_name'])

if __name__ == '__main__':
    asyncio.run(verify())
