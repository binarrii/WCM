import base64
import os
import time
import psycopg2
import httpx
import asyncio

DB_URL = 'postgresql://postgres:postgres@localhost:5433/facerec'
DEEPFACE_API = 'http://127.0.0.1:5000/register'

async def register_face(client, name, file_path, semaphore):
    async with semaphore:
        if not os.path.exists(file_path):
            print(f'File not found: {file_path}')
            return False
        
        try:
            with open(file_path, 'rb') as f:
                img_data = base64.b64encode(f.read()).decode('utf-8')
            
            payload = {
                'img': f'data:image/jpeg;base64,{img_data}',
                'img_name': name,
                'model_name': 'Facenet512',
                'detector_backend': 'retinaface',
                'align': True,
                'enforce_detection': False
            }
            
            response = await client.post(DEEPFACE_API, json=payload, timeout=60.0)
            if response.status_code == 200:
                print(f'Successfully registered: {name}')
                return True
            else:
                print(f'Failed to register {name}: {response.text}')
                return False
        except Exception as e:
            print(f'Error registering {name}: {e}')
            return False

async def main():
    print('Connecting to DB...')
    conn = psycopg2.connect(DB_URL)
    cur = conn.cursor()
    cur.execute('SELECT name, file_path FROM face_records;')
    records = cur.fetchall()
    cur.close()
    conn.close()
    
    print(f'Found {len(records)} records. Starting migration...')
    
    semaphore = asyncio.Semaphore(10) # 10 concurrent requests
    
    async with httpx.AsyncClient() as client:
        tasks = []
        for name, file_path in records:
            if not file_path:
                continue
            tasks.append(register_face(client, name, file_path, semaphore))
            
        results = await asyncio.gather(*tasks)
    
    success = sum(results)
    print(f'Migration complete! Successfully migrated {success} out of {len(tasks)} faces.')

if __name__ == '__main__':
    asyncio.run(main())
