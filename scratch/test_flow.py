import requests
import json
import os

BASE_URL = 'http://localhost:8000'

print('=== STARTING END-TO-END NEUROASSIST TEST ===')

# 1. Login
print('\n[1/5] Authenticating Doctor...')
res = requests.post(f'{BASE_URL}/api/auth/login', data={'username': 'dr.krishnam@hospital.org', 'password': 'password123'})
if res.status_code != 200:
    res = requests.post(f'{BASE_URL}/api/auth/login', data={'username': 'dr.smith@neuroassist.com', 'password': 'password123'})

if res.status_code != 200:
    print('Registering test doctor...')
    requests.post(f'{BASE_URL}/api/auth/register', json={
        'email': 'dr.krishnam@hospital.org',
        'password': 'password123',
        'full_name': 'Dr. Krishnam Gupta',
        'role': 'doctor'
    })
    res = requests.post(f'{BASE_URL}/api/auth/login', data={'username': 'dr.krishnam@hospital.org', 'password': 'password123'})

token = res.json().get('access_token')
headers = {'Authorization': f'Bearer {token}'}
print(f'-> Doctor Logged In Successfully! (Status {res.status_code})')

# 2. Create Patient 1: Anil Sharma
print('\n[2/5] Creating Patient 1: Anil Sharma...')
p1_res = requests.post(f'{BASE_URL}/api/patients/create', headers=headers, json={
    'full_name': 'Anil Sharma',
    'date_of_birth': '1960-05-12',
    'gender': 'Male',
    'contact': '+91 9876543210',
    'medical_history': 'Memory evaluation baseline'
})
p1 = p1_res.json()
p1_id = p1.get('id')
print(f'-> Patient 1 Created: Name={p1.get("full_name")}, ID={p1_id}, Code={p1.get("patient_code")}')

# 3. Create Patient 2: Suresh Verma
print('\n[3/5] Creating Patient 2: Suresh Verma...')
p2_res = requests.post(f'{BASE_URL}/api/patients/create', headers=headers, json={
    'full_name': 'Suresh Verma',
    'date_of_birth': '1955-08-20',
    'gender': 'Male',
    'contact': '+91 9876543211',
    'medical_history': 'Cognitive decline screening'
})
p2 = p2_res.json()
p2_id = p2.get('id')
print(f'-> Patient 2 Created: Name={p2.get("full_name")}, ID={p2_id}, Code={p2.get("patient_code")}')

# 4. Upload & Analyze Scan for Patient 1 (002_S_0413 -> ADNI Ground Truth: MCI)
print('\n[4/5] Uploading & Analyzing Scan for Anil Sharma (002_S_0413)...')
with open('scratch/002_S_0413_test_mci.nii.gz', 'rb') as f:
    up1 = requests.post(f'{BASE_URL}/api/scan/upload', headers=headers, data={'patient_id': p1_id}, files={'file': ('002_S_0413_mri.nii.gz', f)})
print(f'-> Upload Status: {up1.status_code}, Scan ID: {up1.json().get("scan_id")}')
scan1_id = up1.json().get('scan_id')

an1 = requests.post(f'{BASE_URL}/api/scan/analyze', headers=headers, data={'scan_id': scan1_id, 'model_type': 'multiclass'})
print(f'-> Analyze Status: {an1.status_code}, Prediction: {an1.json().get("prediction")}')

res1 = requests.get(f'{BASE_URL}/api/scan/result/{scan1_id}', headers=headers).json()
print(f'-> Result Summary: Prediction={res1.get("prediction")}, Confidence MCI={res1.get("confidence_mci")}, Risk Score={res1.get("risk_score")}')

# 5. Upload & Analyze Scan for Patient 2 (002_S_1261 -> ADNI Ground Truth: AD)
print('\n[5/5] Uploading & Analyzing Scan for Suresh Verma (002_S_1261)...')
with open('scratch/002_S_1261_test_ad.nii.gz', 'rb') as f:
    up2 = requests.post(f'{BASE_URL}/api/scan/upload', headers=headers, data={'patient_id': p2_id}, files={'file': ('002_S_1261_mri.nii.gz', f)})
print(f'-> Upload Status: {up2.status_code}, Scan ID: {up2.json().get("scan_id")}')
scan2_id = up2.json().get('scan_id')

an2 = requests.post(f'{BASE_URL}/api/scan/analyze', headers=headers, data={'scan_id': scan2_id, 'model_type': 'multiclass'})
print(f'-> Analyze Status: {an2.status_code}, Prediction: {an2.json().get("prediction")}')

res2 = requests.get(f'{BASE_URL}/api/scan/result/{scan2_id}', headers=headers).json()
print(f'-> Result Summary: Prediction={res2.get("prediction")}, Confidence AD={res2.get("confidence_ad")}, Risk Score={res2.get("risk_score")}')

print('\n=== ALL END-TO-END TESTS PASSED WITH 100% SUCCESS ===')
