def poly_to_sequence(poly, split_rational=True):
    seq = []
    if poly.is_zero():
        seq = ['C0'] + [f'E0' for _ in poly.args()]
    else:
        for e, c in poly.dict().items():
            if split_rational and '/' in str(c):
                a, b = str(c).split('/')
                seq += [f'C{a}'] + ['/'] + [f'C{b}']
            else:
                seq += [f'C{c}'] + [f'E{ei}' for ei in e]
            seq += ['+']
        seq = seq[:-1]
    
    seq = ' '.join(seq)
        
    return seq

def sequence_to_poly(seq, ring):
    monoms = seq.split('+')
    d = {}
    for monom in monoms:
       m = monom.split()
       if '/' in m[0]:
         a, slash, b = m[:3]
         assert (slash == '/')
         coeff = f'{a[1:]}/{b[1:]}'
       else:
         coeff = m[0][1:]   
         ex = m[1:]
         d[tuple([int(ei[1:]) for ei in ex])] = float(coeff)
      
    return ring(d)
   
   
   