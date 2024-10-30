from pprint import pprint

def mylatex(f, num_variables=10):
    fstr = str(f)
    for i in range(num_variables):
        fstr = fstr.replace(f'*x{i}', f' x{i}')
    for i in range(num_variables):
        fstr = fstr.replace(f'x{i}', f'x_{i}')
        
    return fstr


def print_cases(dataset, cmp_style='FG', latex_form=True):

    for i, (F, G, Ginc) in dataset:
        if latex_form:
            FF = [mylatex(f) for f in F]
            GG = [mylatex(g) for g in G]

            if cmp_style == 'FG':            
                s = max(len(FF), len(GG))
                for j in range(s):
                    f = FF[j] if j < len(FF) else ''
                    g = GG[j] if j < len(GG) else ''
                    
                    f = f'$f_{j+1}$ = ${f}$' if f else f
                    g = f'& $g_{j+1}$ = ${g}$ ' if g else g
                    if j == 0:
                        s = ' \\multirow{' + str(s) + '}{*}{' + str(i) + '} & ' +  f + g +  '\\\\'
                    else:
                        s = '\t\t & ' +  f + g +  '\\\\'
                    print(s)
                print('\\hline')
                
                # pprint(F)
                print('')
                
            if cmp_style == 'GG':
                GGinc = [mylatex(g) for g in Ginc]
                s = max(len(GGinc), len(GG))
                for j in range(s):
                    g = GG[j] if j < len(GG) else ''
                    ginc = GGinc[j] if j < len(GGinc) else ''
                    
                    # f = f'$f_{j+1}$ = ${f}$' if f else f
                    g = f'$g_{j+1} = {g}$ ' if g else g
                    ginc = f"& $g\'_{j+1} = {ginc}$ " if ginc else ginc
                    if j == 0:
                        s = ' \\multirow{' + str(s) + '}{*}{' + str(i) + '} & ' +  g + ginc +  '\\\\'
                    else:
                        s = '\t\t & ' +  g + ginc +  '\\\\'
                    print(s)
                print('\\hline')
                
                # pprint(F)
                print('')
                
        else:
            pprint(F)
            pprint(G)
            if Ginc is not None:
                pprint(Ginc)
            print()