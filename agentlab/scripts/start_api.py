from pathlib import Path
import sys
BASE=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(BASE))
import uvicorn
if __name__=='__main__': uvicorn.run('apps.api.main:app',host='127.0.0.1',port=8000,reload=True)
