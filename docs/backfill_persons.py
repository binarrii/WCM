import uuid
from pathlib import Path
from sqlalchemy import select
from wcm_facerec.database import get_session, Person, FaceRecord

def backfill():
    session = get_session()
    try:
        # Get all face records without person_id
        stmt = select(FaceRecord).where(FaceRecord.person_id == None)
        records = session.scalars(stmt).all()
        print(f"Found {len(records)} records without person_id.")
        
        batch_size = 500
        count = 0
        
        for r in records:
            # Infer category type from file_path
            category = "其它"
            if r.file_path:
                if "落马官员" in r.file_path:
                    category = "落马官员"
                elif "劣迹艺人" in r.file_path:
                    category = "劣迹艺人"
                elif "时政敏感" in r.file_path:
                    category = "时政敏感"
                elif "地图" in r.file_path:
                    category = "其它"
                elif "旗帜" in r.file_path:
                    category = "其它"
            
            # Create Person
            person = Person(
                id=uuid.uuid4(),
                name=r.name,
                occupation=None,
                type_=category,
                remarks=None
            )
            session.add(person)
            
            # Link FaceRecord
            r.person_id = person.id
            count += 1
            
            # Commit in batches
            if count % batch_size == 0:
                session.commit()
                print(f"Processed {count} / {len(records)} records...")
                
        session.commit()
        print(f"Successfully backfilled {count} records!")
        
    except Exception as e:
        session.rollback()
        print(f"Error occurred during backfill: {e}")
    finally:
        session.close()

if __name__ == "__main__":
    backfill()
