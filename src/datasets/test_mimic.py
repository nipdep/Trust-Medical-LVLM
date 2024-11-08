import sys 
sys.path.append('../')
from  mimic import Mimic as M

mimic_dataset = M("mimic")
print(type(mimic_dataset[0:5]))