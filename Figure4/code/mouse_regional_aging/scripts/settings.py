from pathlib import Path
import os
import numpy as np,pandas as pd
ROOT=Path(__file__).resolve().parents[1]
PAPER=ROOT.parents[3]
BRAIN=Path(os.environ.get('ANALYSIS_DATA_DIR',PAPER/'analysis_data'))/'Figure4/aging'
RESULTS=Path(os.environ.get('FIGURE4_OUTPUT_DIR', PAPER/'review_output/Figure4'))/'results/mouse_regional_aging'
RESULTS.mkdir(parents=True,exist_ok=True)
CACHE=RESULTS/'cache'
CACHE.mkdir(exist_ok=True)
(RESULTS/'tables').mkdir(exist_ok=True)
(RESULTS/'figures').mkdir(exist_ok=True)
import json
CFG=json.loads((ROOT/'config/analysis.json').read_text())
def load(species,ct):
 assert species=='Mouse'
 p=CACHE/('Mouse_'+ct.replace(' ','_'))
 meta=pd.read_csv(str(p)+'.csv',dtype={'donor':str,'barcode':str});x=np.load(str(p)+'_embedding.npy',mmap_mode='r')
 return meta,x,None,None
