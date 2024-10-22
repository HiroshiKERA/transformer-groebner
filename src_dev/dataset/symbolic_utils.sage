def poly_to_sequence(poly, split_rational=True):
    seq = []
    if poly.is_zero():
        seq = ['C0'] + [f'E0' for _ in poly.args()]
    else:
        d = poly.dict()
        for e, c in lex_sort(d):
            if split_rational and '/' in str(c):
                a, b = str(c).split('/')
                seq += [f'C{a}'] + ['/'] + [f'C{b}']
            else:
                seq += [f'C{c}']
            seq += [f'E{ei}' for ei in e]
            seq += ['+']
        seq = seq[:-1]
    
    seq = ' '.join(seq)
        
    return seq

def sequence_to_poly(seq, ring):
    field = ring.base_ring()
    
    monoms = seq.split('+')
    d = {}
    for monom in monoms:
        m = monom.split()
       
        if '/' in monom:
            a, slash, b = m[:3]
            assert (slash == '/')
            coeff = f'{a[1:]}/{b[1:]}'
            ex = m[3:]
        else:
            coeff = m[0][1:]
            ex = m[1:]
        
        d[tuple([int(ei[1:]) for ei in ex])] = field(coeff)
    
    # if seq != poly_to_sequence(ring(d)):
    #     print(seq)
    #     print(ring(d))
    #     assert(seq == poly_to_sequence(ring(d)))
    
    return ring(d)
   

def get_field(field_name):
    if field_name == 'QQ':
        field = QQ
    elif field_name == 'RR':
        field = RR
    elif field_name == 'ZZ':
        field = ZZ
    elif field_name[:2] == 'GF':
        order = int(field_name[2:])
        field = GF(order)
    else:
        raise ValueError(f'Unknown field {field_name}')
        
    return field

def sort_key(item):
    return tuple(-x for x in item)

def lex_sort(poly_dict):
    '''
    SageMath bug
    PolynomirlRing with RR coefficients does not return dict with exponents sorted by predescribed term order
    
    example: 
    [If coefficients are in ZZ]
    sage: PolynomialRing(ZZ, 'x, y', order='lex').random_element().dict()
    {(2, 0): -1, (1, 1): -4, (0, 2): -1, (0, 1): 3, (0, 0): 1}  # lex order

    [If coefficients are in RR]
    sage: PolynomialRing(RR, 'x, y', order='lex').random_element().dict()
    {(0, 0): -0.0942223927825381,  # not lex order
    (1, 0): 0.106429415327920,
    (0, 1): -0.768209166717910,
    (1, 1): -0.981378351714658,
    (0, 2): 0.0746351239911487}
    '''
    
    return sorted(poly_dict.items(), key=lambda x: sort_key(x[0]))