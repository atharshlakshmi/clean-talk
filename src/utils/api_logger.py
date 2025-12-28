from pathlib import Path
import csv
import uuid
from datetime import datetime

project_root = Path(__file__).parent.parent.parent

class APILogger():
      def __init__(self):
            self.headers = ['request_id', 'timestamp', 'endpoint', 'prompt', 'response', 'status']
            self.filename = project_root / 'reports/api_log.csv'
            
            if not Path(self.filename).exists(): 
                with open(self.filename, mode='w', newline='') as f:
                    writer = csv.DictWriter(f, fieldnames=self.headers)
                    writer.writeheader()
            
      def log(self, endpoint: str, prompt: str, response: dict, status: str, request_id: str = None):
            if request_id is None:
                  request_id = str(uuid.uuid4())[:8]
            
            api_log = {
                  'request_id': request_id,
                  'timestamp': datetime.now().isoformat(),
                  'endpoint': endpoint,
                  'prompt': prompt,
                  'response': response,
                  'status': status
            }
            
            with open(self.filename, mode='a', newline='') as f:
                writer = csv.DictWriter(f, fieldnames=self.headers)
                writer.writerow(api_log)

