import itertools as it 
from time import time 
import os 
import numpy as np 
from joblib import Parallel, delayed
import sys 
import sage.misc.randstate as randstate
import joblib
from pprint import pprint
# randstate.set_random_seed(os.getpid())
# load('src/dataset/symbolic_utils.sage')
load('src/dataset/polynomial_sampler.sage')

# np.random.seed((os.getpid() * int(time())) % 123456789)


class Dataset_Builder_Groebner():
    def __init__(self, ring, config):

        self.ring = ring
        self.config = config

        self.ring               = ring 
        self.coeff_bound        = config['coeff_bound'] if not ring.base_ring().is_finite() else 0
        self.max_coeff_F        = config['max_coeff_F'] if not ring.base_ring().is_finite() else 0
        self.max_coeff_G        = config['max_coeff_G'] if not ring.base_ring().is_finite() else 0
        self.num_bound_F        = config['num_bound_F'] if ring.base_ring() == QQ else 0 
        self.num_bound_G        = config['num_bound_G'] if ring.base_ring() == QQ else 0 
        self.max_degree_F       = config['max_degree_F']
        self.max_degree_G       = config['max_degree_G']
        self.max_num_terms_F    = config['max_num_terms_F']
        self.max_num_terms_G    = config['max_num_terms_G']
        self.max_size_F         = config['max_size_F']
        self.num_duplicants     = config['num_duplicants']
        self.density            = config['density']
        self.gb_type            = config['gb_type']
        
        self.degree_sampling    = config['degree_sampling']
        self.term_sampling      = config['term_sampling']
        

    def random_shape_gb(self, ring, max_coeff=0, num_bound=0, max_degree=0, max_num_terms=0, degree_sampling='uniform', term_sampling='uniform', gb_type='shape', seed=100, strictly_conditioned=True):
        '''
        G = (x_0 - g1(x_{n-1}), x_1 - g2(x_{n-1}), ..., h(x_{n-1}))
        '''
        ## joblib with multiprocesssing cannot use identical random states at the begining.
        randstate.set_random_seed()
        
        ts = time()
        
        ring, field, x, num_vars = self.ring, self.ring.base_ring(), self.ring.gens(), ring.ngens()

        uring = PolynomialRing(field, 1, x[-1], order='lex')
        
        conditions = {'max_degree'      : max_degree, 
                      'min_degree'      : 1,
                      'max_num_terms'   : max_num_terms, 
                      'max_coeff'       : max_coeff,
                      'num_bound'       : num_bound,
                      'nonzero_instance': True,}

        psampler = Polynomial_Sampler(uring, 
                                      degree_sampling=degree_sampling, 
                                      term_sampling=term_sampling, 
                                      strictly_conditioned=strictly_conditioned, 
                                      conditions=conditions)

        h = psampler.sample(num_samples=1)[0]
        h = to_monic(h, uring)
        
        conditions = {'max_degree'      : h.total_degree()-1, 
                      'min_degree'      : 0,
                      'max_num_terms'   : max_num_terms, 
                      'max_coeff'       : max_coeff,
                      'num_bound'       : num_bound,
                      'nonzero_instance': True,}
        
        psampler = Polynomial_Sampler(uring, 
                                      degree_sampling=degree_sampling, 
                                      term_sampling=term_sampling, 
                                      strictly_conditioned=strictly_conditioned, 
                                      conditions=conditions)

        G = psampler.sample(num_samples=1, size=(num_vars-1, 1))[0]
        G = G.stack(matrix(uring, 1, 1, [h])).change_ring(ring) 
        X = matrix(ring, num_vars, 1, (*x[:-1], 0))
        G = G + X
        
        runtime = time() - ts 
        
        return G, runtime, self.get_system_stat(G)

    def random_shape_gbs(self, num_samples, *args, **kwargs):
        
        # Gs = [self.random_shape_gb(*args, **kwargs) for i in range(num_samples)]
        results = Parallel(n_jobs=self.n_jobs, backend="multiprocessing", verbose=True)(
            delayed(self.random_shape_gb)(*args, **kwargs) for i in range(num_samples)
            )

        Gs, runtimes, stats = zip(*results)
        return Gs, list(runtimes), list(stats)
        
        
    def random_cauchy_gb(self, max_degree, max_num_terms=None, seed=100):
        ## joblib with multiprocesssing cannot use identical random states at the begining.
        randstate.set_random_seed()
         
        ring = self.ring 
        field = self.ring.base_ring()
        num_vars = len(ring.gens())
        
        if field == RR:
            ext_ring = PolynomialRing(QQ, 2*num_vars, 'x', order='lex')
            # sub_ring = PolynomialRing(RR, num_vars, 'x', order='lex')
        else:
            ext_ring = PolynomialRing(field, 2*num_vars, 'x', order='lex')
            # sub_ring = PolynomialRing(field, num_vars, 'x', order='lex')

        xs = ext_ring.gens()[:num_vars]
        ys = ext_ring.gens()[num_vars:]
        
        ts = time()
        
        f1 = prod([xs[0] - yi for yi in ys])
        F = [f1]
        if num_vars > 1:
            f2 = (f1(x0=xs[1]) - f1) / (xs[1] - xs[0])
            F.append(expand(f2))
        if num_vars > 2:
            f3 = (f2(x1=xs[2], x0=xs[0]) - f2) / (xs[2] - xs[1])
            F.append(expand(f3))
        if num_vars > 3:
            f4 = (f3(x2=xs[3], x1=xs[1], x0=xs[0]) - f3) / (xs[3] - xs[2])
            F.append(expand(f4))
        if num_vars > 4:
            f5 = (f4(x3=xs[4], x2=xs[2], x1=xs[1], x0=xs[0]) - f4) / (xs[4] - xs[3])
            F.append(expand(f5))
                
        if field == QQ:
            a = [ring.base_ring().random_element(num_bound=self.num_bound_G) for _ in range(num_vars)]
        elif field == RR:
            a = [ring.base_ring().random_element(min=-self.max_coeff_G, max=self.max_coeff_G) for _ in range(num_vars)]
        else:
            a = [ring.base_ring().random_element() for _ in range(num_vars)]
            
        for i in range(num_vars):
            F[i] = F[i].subs(dict(zip(ys, a)))
        
        try: 
            G = matrix(ext_ring, num_vars, 1, F)
            G = G.change_ring(self.ring)
        except:
            assert(field == RR)
            G = matrix(ideal(F).ring(), num_vars, 1, F)
        
        runtime = time() - ts 
        
        return G, runtime, self.get_system_stat(G)
    
    def random_cauchy_gbs(self, num_samples, *args, **kwargs):
        runtimes = []
        Gs = []
        
        results = Parallel(n_jobs=self.n_jobs, backend="multiprocessing", verbose=True)(
            delayed(self.random_cauchy_gb)(*args, **kwargs) for i in range(num_samples)
            )

        Gs, runtimes, stats = zip(*results)
        return Gs, list(runtimes), list(stats)

    # self, ring, max_coeff=0, num_bound=0, max_degree=0, max_num_terms=0, seed=100
    def random_non_gb(self, SG, max_coeff=0, num_bound=0, coeff_bound=100, max_degree=0, max_num_terms=0, density=1.0, max_size=0, degree_sampling='uniform', term_sampling='uniform', seed=100, max_iter=100, strictly_conditioned=True):
        '''
        SG: gb in shape position
        max_size: maximum size of target system (> num_vars)
        '''
        
        randstate.set_random_seed()
    
        conditions = {'max_degree'      : max_degree, 
                      'min_degree'      : 1,
                      'max_num_terms'   : max_num_terms, 
                      'max_coeff'       : max_coeff,
                      'num_bound'       : num_bound,
                      'nonzero_instance': True,}

        psampler = Polynomial_Sampler(self.ring, 
                                      degree_sampling=degree_sampling, 
                                      term_sampling=term_sampling, 
                                      strictly_conditioned=strictly_conditioned, 
                                      conditions=conditions)
        
        for i in range(max_iter):
            ts = time()
            num_vars = SG.nrows()
            m = randint(0, max_size-num_vars) + num_vars
            
            A = psampler.sample(num_samples=1, size=(m, num_vars), density=density, matrix_type='unimoduler_upper_triangular')[0]
            U = psampler.sample(num_samples=1, size=(m, m), matrix_type='unimoduler_upper_triangular')[0]
            P = random_permutation_matrix(m) 

            F = U * P * A * SG

            runtime = time() - ts 
            
            if not self.ring.base_ring().is_finite():
                if max_num_of_matrix(F) <= coeff_bound:
                    break
                else:
                    print('Non-GB has too large coefficient -> retry', flush=True)
            else:
                break

        return F, runtime, self.get_system_stat(F)
    
    def random_non_gbs(self, SGs, *args, **kwargs):
        
        results = Parallel(n_jobs=self.n_jobs, backend="multiprocessing", verbose=True)(delayed(self.random_non_gb)(SG, *args, **kwargs) for i, SG in enumerate(SGs))

        Fs, runtimes, stats = zip(*results)

        return Fs, list(runtimes), list(stats)

    def compute_gb(self, F, algorithm=''):
        ts = time()
        G = ideal(F).groebner_basis(algorithm=algorithm)
        runtime = time() - ts 
        return G, runtime

    def check_gb(self, G):
        return ideal(G).basis_is_groebner()

    def check_same_ideal(self, F, G):
        return ideal(F) == ideal(G)

    def run(self, num_samples, run_check=False, timing=True, n_jobs=-1, timing_n=-1, strictly_conditioned=True, degree_sampling='uniform', term_sampling='uniform'):
        self.n_jobs = n_jobs
        max_size_F = self.max_size_F
        max_degree_F = self.max_degree_F
        max_degree_G = self.max_degree_G
        max_num_terms_F = self.max_num_terms_F
        max_num_terms_G = self.max_num_terms_G
        density = self.density
        # gb_type = self.gb_type
        
        if self.gb_type == 'shape':
            SGs, sg_runtimes, stats_G = self.random_shape_gbs(num_samples, 
                                                              self.ring, 
                                                              max_coeff     = self.max_coeff_G, 
                                                              num_bound     = self.num_bound_G, 
                                                              max_degree    = self.max_degree_G, 
                                                              max_num_terms = self.max_num_terms_G,
                                                              strictly_conditioned = strictly_conditioned,
                                                              degree_sampling = degree_sampling,
                                                              term_sampling = term_sampling)
        if self.gb_type == 'cauchy':
            SGs, sg_runtimes, stats_G = self.random_cauchy_gbs(num_samples, 
                                                               max_degree_G, 
                                                               max_num_terms=max_num_terms_G)
        
        if self.num_duplicants > 1:
            SGs = list(sum(zip(*it.repeat(SGs, self.num_duplicants)), ()))
            sg_runtimes = list(sum(zip(*it.repeat(sg_runtimes, self.num_duplicants)), ()))

        Fs, f_runtimes, stats_F = self.random_non_gbs(SGs, 
                                                      coeff_bound   = self.coeff_bound,
                                                      max_coeff     = self.max_coeff_F, 
                                                      num_bound     = self.num_bound_F, 
                                                      max_degree    = self.max_degree_F, 
                                                      max_num_terms = self.max_num_terms_F,
                                                      max_size      = self.max_size_F,
                                                      density       = density,
                                                      strictly_conditioned = strictly_conditioned,
                                                      degree_sampling = degree_sampling,
                                                      term_sampling = term_sampling)
        
        Fs = [F.list() for F in Fs]
        Gs = [SG.list() for SG in SGs]


        f_runtimes = np.array(f_runtimes)
        sg_runtimes = np.array(sg_runtimes)
        bwd_runtimes = f_runtimes + sg_runtimes

        f_runtime, f_runtime_std = np.sum(f_runtimes), np.std(f_runtimes)
        sg_runtime, sg_runtime_std = np.sum(sg_runtimes), np.std(sg_runtimes)
        bwd_runtime, bwd_runtime_std = np.sum(bwd_runtimes), np.std(bwd_runtimes)

        runtime_d = {'bwd_runtime': float(bwd_runtime),
                     'bwd_runtime_std': float(bwd_runtime_std),
                     'runtime_for_Fs': float(f_runtime), 
                     'runtime_for_Fs_std': float(f_runtime_std), 
                     'runtime_for_SGs': float(sg_runtime),
                     'runtime_for_SGs_std': float(sg_runtime_std),
                     'num_samples': int(num_samples)
                     }

        if run_check:
            ret = Parallel(n_jobs=self.n_jobs, backend="multiprocessing", verbose=True)(delayed(self.check_gb)(G) for G in Gs)
            if np.all(list(ret)):
                print('[ok] all bases are Gröbner bases')
            else: 
                print('[NG] some bases are NOT Gröbner bases')
        
        timing_n = timing_n if timing_n > 0 else len(Fs)
        if timing:
            results = Parallel(n_jobs=self.n_jobs, backend="multiprocessing", verbose=True)(delayed(self.compute_gb)(F) for F in Fs[:timing_n])
            _, runtimes = zip(*results)
            runtimes = list(runtimes)
            
            fwd_runtime = np.sum(np.array(runtimes))
            fwd_runtime_std = np.std(np.array(runtimes))
            runtime_d['fwd_runtime'] = float(fwd_runtime)
            runtime_d['fwd_runtime_std'] = float(fwd_runtime_std)
            
        _f_runtimes, _sg_runtimes, _bwd_runtimes = f_runtimes[:timing_n], sg_runtimes[:timing_n], bwd_runtimes[:timing_n]
        _f_runtime, _sg_runtime, _bwd_runtime = np.sum(_f_runtimes), np.sum(_sg_runtimes), np.sum(_bwd_runtimes)
        
        print(f'---- timing (using {timing_n} samples)-------------')
        print(f'backward generation | {_bwd_runtime:.5f} [sec] ({_sg_runtime:.5f} + {_f_runtime:.5f}) --- {_bwd_runtime/timing_n:.5f} [sec/sample]')
        if timing:
            print(f'  foward generation | {fwd_runtime:.5f} [sec] --- {fwd_runtime/timing_n:.5f} [sec/sample]')
        print('-------------------------')
            
        
        self.dataset = list(zip(Fs, Gs))        
        self.runtime_stat = runtime_d
        self.stats = (stats_F, stats_G)
        
        return self

    def get_system_stat(self, P):
        '''
        P: set of polynomials (column vector with polynomial entries)
        '''
        if not isinstance(P, list): P = P.list()
        size = len(P)
        degrees = [int(p.total_degree()) for p in P]
        num_monoms = [int(len(p.monomials())) for p in P]

        # max_coeff = max([max([abs(c) for c in p.coefficients()]) for p in P])
        field = P[0].base_ring()
        coeffs = it.chain(*[p.coefficients() for p in P])
        if field == QQ: coeffs = [c.numer() for c in coeffs] + [c.denom() for c in coeffs]
        if field.characteristic() == 0: coeffs = [abs(c) for c in coeffs]
        max_coeff = max(coeffs)
        
        stat = {'size': int(size), 
                'degrees': degrees,
                'num_monoms': num_monoms,
                'min_degree': float(np.min(degrees)),
                'min_num_monoms': float(np.min(num_monoms)),
                'max_degree': float(np.max(degrees)),
                'max_num_monoms': float(np.max(num_monoms)),
                'total_num_monoms': float(np.sum(degrees)),
                'max_coeff': int(max_coeff),
                'is_GB': int(self.check_gb(P)) if field is not RR else int(-1)
                }
        
        return stat 
    
 

def summarize_stats(stats, metric=['mean', 'std', 'max', 'min', 'median']):
    summary = {}
    for k in stats[0]:
        if isinstance(stats[0][k], list): continue
        if 'mean' in metric:
            summary[f'{k}_mean'] = float(np.mean([item[k] for item in stats]))
        if 'median' in metric:    
            summary[f'{k}_median'] = float(np.median([item[k] for item in stats]))
        if 'std' in metric:    
            summary[f'{k}_std'] = float(np.std([item[k] for item in stats]))
        if 'max' in metric:    
            summary[f'{k}_max'] = float(np.max([item[k] for item in stats]))
        if 'min' in metric:    
            summary[f'{k}_min'] = float(np.min([item[k] for item in stats]))
    return summary

def random_permutation_matrix(m):
    randstate.set_random_seed()
    
    perms = Permutations(list(1..m))
    P = matrix(Permutation(perms.random_element()))
    return P 

def to_monic(p, ring):
    lm_expoenent = p.lm().exponents()[0]
    d = p.dict()
    d[lm_expoenent] = 1
    return ring(d)
    
    
def get_symmetric_polynomials(n, ring):
    e = SymmetricFunctions(ring.base_ring()).e()
    return [e([i]).expand(n) for i in range(1, n+1)]
    

def term_order_change(ring, new_ring, data):
    input_text, target_text = data.split(":")
    input_text = input_text.strip()
    target_text = target_text.strip()
    
    F = [new_ring(fstr) for fstr in input_text.split(' [SEP] ')]
    G = [ring(gstr) for gstr in target_text.split(' [SEP] ')]
    G = ideal(G).transformed_basis(algorithm='fglm', other_ring=new_ring)
    
    return F, G

