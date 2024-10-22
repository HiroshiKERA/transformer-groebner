import sys 
sys.path.append('/app/src')
import os 
from tqdm import tqdm
from loader.data import DictDataset
from joblib import Parallel, delayed
load('src/dataset/symbolic_utils.sage')

data_origin = 'data2/gb_dataset_n=2_field=GF7/data.train.lex.raw'

try:
    with open(data_origin, "r") as f:
        data = f.read().splitlines()
except:
    raise FileNotFoundError

input_texts = [line.split(":")[0].strip() for line in data]
target_texts = [line.split(":")[1].strip() for line in data]

dataset = DictDataset(input_texts, target_texts)

def get_raw_dataset(dataset, ring):
    raw_dataset = []
    for sample in dataset:
        Fstr, Gstr = sample['input'], sample['target']
        F, G = [], []
        
        for fstr in Fstr.split(' [SEP] '):
            f = sequence_to_poly(fstr, ring)
            F.append(f)
        
        for gstr in Gstr.split(' [SEP] '):
            g = sequence_to_poly(gstr, ring)
            G.append(g)
            
        raw_dataset.append((F, G))

    return raw_dataset

ring =  PolynomialRing(GF(7), 'x', 2, order='lex')
# raw_dataset = get_raw_dataset(dataset, ring)


path = 'data2/shape/shape_n=2_field=GF7'
os.makedirs(path, exist_ok=True)


def ont_step(Fstr, Gstr):
    F = [ring(f) for f in Fstr.split('[SEP]')]
    G = [ring(g) for g in Gstr.split('[SEP]')]
    
    Fnew = [poly_to_sequence(f, ring) for f in F]
    Gnew = [poly_to_sequence(f, ring) for f in G]
    
    Fnew = ' [SEP] '.join(Fnew)
    Gnew = ' [SEP] '.join(Gnew)

    return f'{Fnew} : {Gnew}\n'

new_dataset = []
# for sample in tqdm(dataset):
#     Fstr, Gstr = sample['input'], sample['target']

#     # print(Fstr)
    
#     # for fi in Fstr.split('[SEP]'):
#     #     # print(fi)
#     #     print(ring(fi))
    
#     F = [ring(f) for f in Fstr.split('[SEP]')]
#     G = [ring(g) for g in Gstr.split('[SEP]')]
    
#     Fnew = [poly_to_sequence(f, ring) for f in F]
#     Gnew = [poly_to_sequence(f, ring) for f in G]
    
#     Fnew = ' [SEP] '.join(Fnew)
#     Gnew = ' [SEP] '.join(Gnew)
    
    
#     new_dataset.append(f'{Fnew} : {Gnew}\n')
#     # with open(f'{path}/shape_n=2_field=GF7.train.lex.infix', 'w') as f:
#     #     f.write(new_dataset)
    
new_dataset = Parallel(n_jobs=-1, verbose=True, backend="multiprocessing")([delayed(ont_step)(Fstr, Gstr) for Fstr, Gstr in zip(input_texts, target_texts)])

with open(f'{path}/shape_n=2_field=GF7.train.lex.infix', 'w') as f:
    f.writelines(new_dataset)
    