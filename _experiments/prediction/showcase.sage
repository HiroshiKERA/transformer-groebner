load('src_dev/loader/data.py')
load('src_dev/data/symbolic_utils.sage')
from pprint import pprint
import os 
import numpy as np 
import itertools as it
import argparse

def get_parser():
    """
    Generate a parameters parser.
    """
    # parse parameters
    parser = argparse.ArgumentParser(description="Language transfer")

    # path
    parser.add_argument("--data_path", type=str, default="data/")
    parser.add_argument("--load_path", type=str, default="results/")
    parser.add_argument("--save_path", type=str, default="results/")
    parser.add_argument("--correct_file", type=str, default="prediction_correct.txt")
    parser.add_argument("--incorrect_file", type=str, default="prediction_incorrect.txt")

    # setup
    parser.add_argument("--data_encoding", type=str, default="prefix")
    parser.add_argument("--term_order", type=str, default="lex")
    parser.add_argument("--field", type=str, default='QQ', help='QQ or FP with some integer P (e.g., F7).')
    parser.add_argument("--num_variables", type=int, default=2)

    # evaluation parameters
    parser.add_argument("--num_prints", type=int, default=10)


    return parser


def _support_eq(f1, f2):
    return f1.monomials() == f2.monomials()

def support_eq(F1, F2):
    # return np.all([_support_eq(f1, f2) for f1, f2 in it.product(F1, F2)])
    return len(F1) == len(F2) and np.all([_support_eq(f1, f2) for f1, f2 in zip(F1, F2)])

def is_reduced_GB(G):
    if not ideal(G).basis_is_groebner(): return False 
    
    for i, g1 in enumerate(G):
        for j, g2 in enumerate(G):
            if i == j: continue
            lm = g1.lm()
            if np.any([term.reduce([lm]) == 0 for term in g2.monomials()]):
                return False 
            
    return True 


def showcase(num_var, field, term_order, data_encoding, data_path, save_dir, verbose=True):
    
    testset = load_data(data_path, 
                        extensions=['test'], 
                        encoding=f'{term_order}.raw', 
                        do_shuffle=[False], 
                        return_dataloader=False)
    split_rational = data_encoding == 'prefix'
    
    filename_correct = os.path.join(save_dir, 'prediction_correct.txt')
    with open(filename_correct, "r") as f:
        correct_samples_text = f.read().splitlines()

    filename_incorrect = os.path.join(save_dir, 'prediction_incorrect.txt')
    with open(filename_incorrect, "r") as f:
        incorrect_samples_text = f.read().splitlines()
        
    ring = PolynomialRing(field, num_var, 'x', order=term_order)    

    correct_Gs = {}
    for correct_sample_text in correct_samples_text:
        idx, gs_text = correct_sample_text.split(' >> ')
        idx = int(idx)

        gs_text = gs_text.split(' [SEP] ')
        G = [prefix_to_poly(g_text, ring, split_rational=split_rational) for g_text in gs_text]
        correct_Gs[idx] = G

    incorrect_Gs = {}
    n_invalids = 0
    for incorrect_sample_text in incorrect_samples_text:
        idx, gs_text = incorrect_sample_text.split(' >> ')
        idx = int(idx)
        gs_text = gs_text.split(' [SEP] ')

        try:
            G = [prefix_to_poly(g_text, ring, split_rational=split_rational) for g_text in gs_text]
        except:
            print(idx, gs_text)
            print()
            # exit()

        n_invalids += sum([g is None for g in G])
        incorrect_Gs[idx] = [g for g in G if g is not None]
        
    print(f'#invalid polynomials = {n_invalids}')

    stats_list_correct = []
    stats_list_incorrect = []
    stats_list = []
    support_hit = 0
    for i, sample in enumerate(testset):
        fs_text = sample['input'].split(' [SEP] ')
        gs_text = sample['target'].split(' [SEP] ')
        
        F = [ring(f_text) for f_text in fs_text]
        G = [ring(g_text) for g_text in gs_text]
        
        G_pred = correct_Gs[i] if i in correct_Gs else incorrect_Gs[i]
        
        # print(G)
        # print(G_pred)
        # print()
        support_hit = support_eq(G, G_pred)
        
        stats = {'index': int(i),
                 'hit': int(i in correct_Gs),
                 'support_hit': int(support_hit),
                 'linear_dimension': int(len(ideal(G).normal_basis())) if field != RR else 0,  # error for RR
                 'size_F': int(len(F)),
                 'total_num_terms_F': int(sum([len(f.monomials()) for f in F])),
                 'max_num_terms_F': int(np.max([len(f.monomials()) for f in F])),
                 'min_num_terms_F': int(np.min([len(f.monomials()) for f in F])),
                 'max_deg_F': int(np.max([f.total_degree() for f in F])),
                 'min_deg_F': int(np.min([f.total_degree() for f in F])),
                 'size_G': int(len(G)),
                 'total_num_terms_G': int(sum([len(g.monomials()) for g in G])),
                 'max_num_terms_G': int(np.max([len(g.monomials()) for g in G])),
                 'min_num_terms_G': int(np.min([len(g.monomials()) for g in G])),
                 'max_deg_G': int(np.max([g.total_degree() for g in G])),
                 'min_deg_G': int(np.min([g.total_degree() for g in G])),
                 }

        if i in correct_Gs:
            stats_list_correct.append(stats)
        else:
            stats_list_incorrect.append(stats)
        
        stats_list.append(stats)
        
    summary = {}
    for key in stats_list[0]:
        vals = [stats[key] for stats in stats_list]
        summary[key] = np.array(vals)
        
    num_sampels = len(stats_list)
    acc = sum(summary['hit']) / num_sampels
    support_acc = sum(summary['support_hit']) / num_sampels
    print(f'accuracy = {acc:.3f}, support accucary = {support_acc:.3f}, #samples = {num_sampels}')
    
    print('-----------------------')
    for key in summary:
        if key in ('index', 'hit', 'size_G'): continue
        cc = np.corrcoef(summary['hit'], y=summary[key])[0,1]
        print(f'{key}: {cc:.3f}')
    print('-----------------------')    

def mylatex(f):
    fstr = str(f)
    for i in range(5):
        fstr = fstr.replace(f'*x{i}', f' x{i}')
    for i in range(5):
        fstr = fstr.replace(f'x{i}', f'x_{i}')
        
    return fstr
    

def print_case(num_var, field, term_order, data_encoding, data_path, save_dir, case='incorrect', cmp_style='FG', verbose=True, num_prints=5, latex_form=False):
    assert(case in ('correct', 'incorrect'))
    
    print(f'== {case} cases (showing {num_prints} samples) =======')
    
    split_rational = data_encoding == 'prefix'
    
    testset = load_data(data_path, extensions=['test'], encoding=f'{term_order}.raw', do_shuffle=[False], return_dataloader=False)
        
    filename_incorrect = os.path.join(save_dir, f'prediction_{case}.txt')
    with open(filename_incorrect, "r") as f:
        case_samples_text = f.read().splitlines()

    ring = PolynomialRing(field, num_var, 'x', order=term_order)    

    case_Gs = {}
    n_invalids = 0
    for case_sample_text in case_samples_text:
        idx, gs_text = case_sample_text.split(' >> ')
        idx = int(idx)
        gs_text = gs_text.split(' [SEP] ')
        # gs_text = [prefix_to_infix(g_text.split(), return_empty_for_invalid=True) for g_text in gs_text]
        # gs_text = [g_text.replace('**', '^') for g_text in gs_text]
        # G = [ring(g_text) for g_text in gs_text if g_text]
        G = [prefix_to_poly(g_text, ring, split_rational=split_rational) for g_text in gs_text]
        
        n_invalids += sum([g is None for g in G])
        case_Gs[idx] = [g for g in G if g is not None]

    # print(f'#invalid polynomials = {n_invalids}')

    support_acc = 0
    tot = 0
    print(f'[PRB]')
    print(f'[ANS] ')
    print(f'[PRD] ')
    print('')
    print(ring)
    for k, (i, Ginc) in enumerate(case_Gs.items()):
        sample = testset[i]
        fs_text = sample['input'].split(' [SEP] ')
        gs_text = sample['target'].split(' [SEP] ')
        
        F = [ring(f_text) for f_text in fs_text]
        G = [ring(g_text) for g_text in gs_text]
        
        support_acc += int(support_eq(G, Ginc))
        tot += 1
        
        if latex_form:
            FF = [mylatex(ring(f)) for f in F]
            GG = [mylatex(ring(g)) for g in G]
            GGinc = [mylatex(ring(g)) for g in Ginc]

            if cmp_style == 'FG':            
                s = max(len(FF), len(GG))
                for j in range(s):
                    f = FF[j] if j < len(FF) else ''
                    g = GG[j] if j < len(GG) else ''
                    
                    f = f'$f_{j+1}$ = ${f}$' if f else f
                    g = f'& $g_{j+1}$ = ${g}$ ' if g else g
                    if j == 0:
                        s = ' \multirow{' + str(s) + '}{*}{' + str(i) + '} & ' +  f + g +  '\\\\'
                    else:
                        s = '\t\t & ' +  f + g +  '\\\\'
                    print(s)
                print('\\hline')
                
                # pprint(F)
                print('')
                
            if cmp_style == 'GG':
                s = max(len(GGinc), len(GG))
                for j in range(s):
                    g = GG[j] if j < len(GG) else ''
                    ginc = GGinc[j] if j < len(GGinc) else ''
                    
                    # f = f'$f_{j+1}$ = ${f}$' if f else f
                    g = f'$g_{j+1} = {g}$ ' if g else g
                    ginc = f"& $g\'_{j+1} = {ginc}$ " if ginc else ginc
                    if j == 0:
                        s = ' \multirow{' + str(s) + '}{*}{' + str(i) + '} & ' +  g + ginc +  '\\\\'
                    else:
                        s = '\t\t & ' +  g + ginc +  '\\\\'
                    print(s)
                print('\\hline')
                
                # pprint(F)
                print('')
                
                    
                
                        
        else:
            pprint(F)
            pprint(G)
            pprint(Ginc)
            print('')
        
        if num_prints > 0 and k > num_prints: break 
    
    support_acc /= 1.0 * tot
    print(f'support accuracy = {support_acc:.3f}')
    
    
def check_reducity(num_var,field, term_order, save_dir, data_path, verbose=True):
    testset = load_data(data_path, extensions=['test'], encoding=f'{term_order}.raw', do_shuffle=[False], return_dataloader=False)
        
    filename_correct = os.path.join(save_dir, 'prediction_correct.txt')
    
    with open(filename_correct, "r") as f:
        correct_samples_text = f.read().splitlines()

    filename_incorrect = os.path.join(save_dir, 'prediction_incorrect.txt')
    with open(filename_incorrect, "r") as f:
        incorrect_samples_text = f.read().splitlines()

    ring = PolynomialRing(field, num_var, 'x', order=term_order)    

    correct_Gs = {}
    for correct_sample_text in correct_samples_text:
        idx, gs_text = correct_sample_text.split(' >> ')
        idx = int(idx)
        gs_text = gs_text.split(' [SEP] ')
        G = [prefix_to_poly(g_text, ring, split_rational=split_rational) for g_text in gs_text]
        correct_Gs[idx] = G

    incorrect_Gs = {}
    n_invalids = 0
    for incorrect_sample_text in incorrect_samples_text:
        idx, gs_text = incorrect_sample_text.split(' >> ')
        idx = int(idx)
        gs_text = gs_text.split(' [SEP] ')
        G = [prefix_to_poly(g_text, ring, split_rational=split_rational) for g_text in gs_text]                
        n_invalids += sum([g is None for g in G])
        incorrect_Gs[idx] = [g for g in G if g is not None]

    print(f'#invalid polynomials = {n_invalids}')

    stats_list_correct = []
    stats_list_incorrect = []
    stats_list = []
    support_hit = 0
    GB_in_incorrect = 0
    reducedGB_in_incorrect = 0
    eqideal_in_incorrect = 0

    GB_in_correct = 0
    reducedGB_in_correct = 0
    eqideal_in_correct = 0

    for i, sample in enumerate(testset):
        fs_text = sample['input'].split(' [SEP] ')
        gs_text = sample['target'].split(' [SEP] ')
        
        F = [ring(f_text) for f_text in fs_text]
        G = [ring(g_text) for g_text in gs_text]
        
        GB = ideal(G).groebner_basis(algorithm='libsingular:std')
        assert(len(G) == len(GB) and np.all([g in G for g in GB]))
        assert(is_reduced_GB(G))
        
        if i in correct_Gs:
            Gc = correct_Gs[i]
            redGc = ideal(Gc).groebner_basis(algorithm='libsingular:std')
            GB_in_correct += int(ideal(Gc).basis_is_groebner())
            # reducedGB_in_correct += int(len(Gc) == len(G) and np.all([g in Gc for g in G]))
            reducedGB_in_correct += int(is_reduced_GB(Gc))
            eqideal_in_correct += int(ideal(G) == ideal(Gc))

        if i in incorrect_Gs:
            Ginc = incorrect_Gs[i]
            redGinc = ideal(Ginc).groebner_basis(algorithm='libsingular:std')
            GB_in_incorrect += int(ideal(Ginc).basis_is_groebner())
            # reducedGB_in_incorrect += int(len(Ginc) == len(G) and np.all([g in Ginc for g in G]))
            reducedGB_in_incorrect += int(is_reduced_GB(Ginc))
            eqideal_in_incorrect += int(ideal(G) == ideal(Ginc))
        
    print(f'GBs in corrected predictions: [{GB_in_correct}/{len(correct_Gs)}]')
    print(f'Reduced GBs in corrected predictions: [{reducedGB_in_correct}/{len(correct_Gs)}]')
    print(f'Same ideal to F: [{eqideal_in_correct}/{len(correct_Gs)}]')
    print('')
    print(f'GBs in incorrected predictions: [{GB_in_incorrect}/{len(incorrect_Gs)}]')
    print(f'Reduced GBs in incorrected predictions: [{reducedGB_in_incorrect}/{len(incorrect_Gs)}]')
    print(f'Same ideal to F: [{eqideal_in_incorrect}/{len(incorrect_Gs)}]')
    
def main():
    parser = get_parser()
    args = parser.parse_args()

    if args.field  == 'QQ':
        field = QQ
    if args.field  == 'RR':
        field = RR
    if args.field[0] == 'F' and args.field[1:].isdigit():
        field = GF(int(args.field[1:]))

    print(os.path.basename(args.data_path))
    showcase(args.num_variables, field, args.term_order, args.data_encoding, args.data_path, args.save_path, verbose=True)

    print('\n\n---- printing latex codes for tables ------------------\n')

    print('### sucessful cases (F-G view)')
    print_case(args.num_variables, field, args.term_order, args.data_encoding, args.data_path, args.save_path, 
               case='correct', cmp_style='FG', verbose=True, latex_form=True, num_prints=args.num_prints)
    print('\n\n\n')

    print('### sucessful cases (G-G view)')
    print_case(args.num_variables, field, args.term_order, args.data_encoding, args.data_path, args.save_path, 
               case='correct', cmp_style='GG', verbose=True, latex_form=True, num_prints=args.num_prints)
    print('\n\n\n')

    print('### failure cases (F-G view)')
    print_case(args.num_variables, field, args.term_order, args.data_encoding, args.data_path, args.save_path, 
               case='incorrect', cmp_style='FG', verbose=True, latex_form=True, num_prints=args.num_prints)
    print('\n\n\n')

    print('### failure cases (G-G view)')
    print_case(args.num_variables, field, args.term_order, args.data_encoding, args.data_path, args.save_path, 
               case='incorrect', cmp_style='GG', verbose=True, latex_form=True, num_prints=args.num_prints)


if __name__ == '__main__':
    main()
    

# def main(term_order):
#     char = 7
#     iterator = range(2, 3)
#     # term_order = 'degrevlex'
#     # other_ext = f'_char={char}_PE2000'
#     other_ext = f'_field=QQ_ep16'
#     load_dirs = [f'results/shape_gb_{term_order}/gb_dataset_n={n}{other_ext}' for n in iterator]
#     # data_paths = [f'data/gb_dataset_n={n}_char={char}_ex/data' for n in iterator]
#     # save_dirs = [f'results/prediction_{term_order}_char={char}_ex/{os.path.basename(load_dir)}' for load_dir in load_dirs]
#     data_paths = [f'data/gb_dataset_n={n}_field=QQ/data' for n in iterator]
#     save_dirs = [f'results/prediction_{term_order}_field=QQ/{os.path.basename(load_dir)}' for load_dir in load_dirs]

#     case = ('correct', 'incorrect')[1]
#     cmp_style = ('FG', 'GG')[0]
    
#     print_case(n, char, term_order, save_dir, data_path, case=case, cmp_style=cmp_style, verbose=True, latex_form=True, num_prints=20)

#     for n, data_path, save_dir in zip(iterator, data_paths, save_dirs):
#         print(data_path)
#         # os.makedirs(save_dir, exist_ok=True)
#         # print(os.path.basename(save_dir))
#         showcase(n, char, term_order, save_dir, data_path, verbose=True)
#         # print_case(n, char, term_order, save_dir, data_path, case=case, cmp_style=cmp_style, verbose=True, latex_form=True, num_prints=20)
#         # check_reducity(n, char, term_order, save_dir, data_path, verbose=True)
#         print('\n\n')


# if __name__ == '__main__':
#     import sys 
#     term_order = sys.argv[1]
#     main(term_order)

