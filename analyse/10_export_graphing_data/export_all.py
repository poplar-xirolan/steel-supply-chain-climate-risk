"""Create new figure inputs in result/graphing_inputs without modifying graphing."""
from pathlib import Path
import sys
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'library'))
from steel_supply.export_inputs import export_all

if __name__=='__main__':export_all()
