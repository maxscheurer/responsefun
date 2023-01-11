from itertools import product 
import numpy as np
from responsefun.symbols_and_labels import *
import string

#taken from pyscf
ELEMENTS = [
    'X',  # Ghost
    'H' , 'He', 'Li', 'Be', 'B' , 'C' , 'N' , 'O' , 'F' , 'Ne',
    'Na', 'Mg', 'Al', 'Si', 'P' , 'S' , 'Cl', 'Ar', 'K' , 'Ca',
    'Sc', 'Ti', 'V' , 'Cr', 'Mn', 'Fe', 'Co', 'Ni', 'Cu', 'Zn',
    'Ga', 'Ge', 'As', 'Se', 'Br', 'Kr', 'Rb', 'Sr', 'Y' , 'Zr',
    'Nb', 'Mo', 'Tc', 'Ru', 'Rh', 'Pd', 'Ag', 'Cd', 'In', 'Sn',
    'Sb', 'Te', 'I' , 'Xe', 'Cs', 'Ba', 'La', 'Ce', 'Pr', 'Nd',
    'Pm', 'Sm', 'Eu', 'Gd', 'Tb', 'Dy', 'Ho', 'Er', 'Tm', 'Yb',
    'Lu', 'Hf', 'Ta', 'W' , 'Re', 'Os', 'Ir', 'Pt', 'Au', 'Hg',
    'Tl', 'Pb', 'Bi', 'Po', 'At', 'Rn', 'Fr', 'Ra', 'Ac', 'Th',
    'Pa', 'U' , 'Np', 'Pu', 'Am', 'Cm', 'Bk', 'Cf', 'Es', 'Fm',
    'Md', 'No', 'Lr', 'Rf', 'Db', 'Sg', 'Bh', 'Hs', 'Mt', 'Ds',
    'Rg', 'Cn', 'Nh', 'Fl', 'Mc', 'Lv', 'Ts', 'Og',
]

NUC = dict(((i,x) for i,x in enumerate(ELEMENTS)))
NUC_2 = dict(((x, i) for i,x in enumerate(ELEMENTS)))

available_ops = {
            'electric': 'DIPLEN',
            'magnetic': "ANGMOM",
            'nabla': 'DIPVEL'
            }

ABC = list(string.ascii_uppercase)


def generate_response_section(operators, omegas, tda = False):
    input_dalton = []
    #check if op is available
    assert len(operators) == len(omegas)
    for i in operators:
        assert i in available_ops
    input_dalton.append('**RESPONSE\n')

    if tda == True:
        input_dalton.append('.TDA\n')


    #OPERATORS
    if len(operators) == 4:
        input_dalton.append('*CUBIC\n')
    if len(operators) == 3:
        input_dalton.append('*QUADRA\n')
    if len(operators) == 2:
        input_dalton.append('*LINEAR\n')

    for i, val in enumerate(operators):
        input_dalton.append(f'.{ABC[i]}PROP\n X{available_ops[operators[i]]}\n')
        input_dalton.append(f'.{ABC[i]}PROP\n Y{available_ops[operators[i]]}\n')
        input_dalton.append(f'.{ABC[i]}PROP\n Z{available_ops[operators[i]]}\n')

    #FREQUENCIES
    if len(operators) != 2:
        for i, val in enumerate(omegas):
            if i > 0:
                input_dalton.append(f'.{ABC[i]}FREQ\n 1 \n {val[-1]}\n')
    else:
        for i, val in enumerate(omegas):
            if i > 0:
                input_dalton.append(f'.FREQUE\n 1\n {val[-1]}\n')
    return input_dalton


def generate_cas_input(state, sos_object, basisset, cas_space, inactive_orbitals, electrons, multiplicity ,*, 
        hf = True, mp2 = False, symmetry = 1, gauge_origin = None,
        omegas = [(w_o, w_1+w_2+w_3), (w_1, 0.0), (w_2, 0), (w_3,0)]        
        ):
    operators = [op.op_type for op in sos_object.operators]
    input_dalton = ['*DALTON INPUT\n', '.DIRECT\n', '.RUN RESPONSE\n', '**WAVE FUNCTIONS\n']

    #wave function section
    
    if hf == True:
        input_dalton.append('.HF\n')
    if mp2 == True:
        input_dalton.append('.MP2\n')

    #configuration input
    input_dalton.append('*CONFIGURATION INPUT\n')
    input_dalton.append(f'.SPIN MULTIPLICITY\n {multiplicity}\n')
    input_dalton.append(f'.SYMMETRY\n {symmetry}\n')
    input_dalton.append(f'.CAS SPACE\n {cas_space}\n')
    input_dalton.append(f'.INACTIVE ORBITALS\n {inactive_orbitals}\n')
    input_dalton.append(f'.ELECTRONS\n {electrons}\n')
    
    #gauge origin selection
    if not gauge_origin == None and isinstance(gauge_origin, str):
        input_dalton.append(f'**INTEGRALS\n.GAUGEO\n {gauge_origin}\n')
    elif not gauge_origin == None and isinstance(gauge_origin, list):
        input_dalton.append(f'**INTEGRALS\n.GAUGEO\n {gauge_origin[0]}, {gauge_origin[1]}, {gauge_origin[2]},\n')

    response = generate_response_section(operators, omegas)
    input_dalton += response

    input_dalton.append('**END OF INPUT')

    with open('cas_input.dal', 'w') as dal_file:
        dal_file.writelines(input_dalton)

    generate_mol(state, basisset)

def generate_cis_input(state, sos_object, basisset ,*, 
        gauge_origin = None,
        omegas = [(w_o, w_1+w_2+w_3), (w_1, 0.0), (w_2, 0), (w_3,0)]        
        ):
    operators = [op.op_type for op in sos_object.operators]
    input_dalton = ['*DALTON INPUT\n', '.DIRECT\n', '.RUN RESPONSE\n', '**WAVE FUNCTIONS\n', '.HF\n', '*SCF INPUT\n', '.THRESH\n1.0D-08\n']

    #gauge origin selection
    if not gauge_origin == None and isinstance(gauge_origin, str):
        input_dalton.append(f'**INTEGRALS\n.GAUGEO\n {gauge_origin}')
    elif not gauge_origin == None and isinstance(gauge_origin, list):
        input_dalton.append(f'**INTEGRALS\n.GAUGEO\n {gauge_origin[0]}, {gauge_origin[1]}, {gauge_origin[2]},')

    response = generate_response_section(operators, omegas, tda = True)
    input_dalton += response

    input_dalton.append('**END OF INPUT')

    with open('cas_input.dal', 'w') as dal_file:
        dal_file.writelines(input_dalton)

    generate_mol(state, basisset)

def generate_cc_input(state, sos_object, basisset, method,
         omegas = [(w_o, w_1+w_2+w_3), (w_1, 0.0), (w_2, 0), (w_3,0)],
         gauge_origin = None
        ):
    """
    state : adcc state object
    sos_object :  sos obeject (responsefun class)
    basis set: str, 
    method :  str, cc2, ccsd, cc3 etc..
    omegas : wie vroher
    """
    # assess operators and check if they are available
    operators = [op.op_type for op in sos_object.operators]
    #CHECK OPERATORS
    for i, val in enumerate(operators):
        assert val in available_ops
    
    input_dalton = ['*DALTON INPUT\n', '.RUN WAVEFUNCTIONS\n', '**INTEGRALS\n']
    #Integral section
    for i in sos_object.operator_types:
        if i in available_ops:
            input_dalton.append(f'.{available_ops[i]}\n')

    #gauge origin selection
    if not gauge_origin == None and isinstance(gauge_origin, str):
        input_dalton.append(f'**INTEGRALS\n.GAUGEO\n {gauge_origin}')
    elif not gauge_origin == None and isinstance(gauge_origin, list):
        input_dalton.append(f'**INTEGRALS\n.GAUGEO\n {gauge_origin[0]}, {gauge_origin[1]}, {gauge_origin[2]},')

    
    #WAVEFUNTION SECTION
    input_dalton.append(f"""**WAVE FUNCTIONS
.CC
*SCF INPUT
.THRESH
1.0D-12
*CC INPUT
.{method}
.THRENR
1.0D-8
.THRLEQ
1.0D-8
.MAXRED
800
.MXLRV
800""")

    #RESPONSE SECTION
    if len(operators) == 4:
        input_dalton.append('*CCCR\n')
        input_dalton.append(f'.MIXFRE\n{len(omegas)-1}\n')
    elif len(operators) == 3:
        input_dalton.append('*CCQR\n')
        input_dalton.append(f'.MIXFRE\n{len(omegas)-1}\n')
    elif len(operators) == 2:
        input_dalton.append('*CCLR\n')
        input_dalton.append('FREQUE\n')
    else:
        raise NotImplementedError('Just linear, quadratic and cubic responses!')
    for i, val in enumerate(omegas):
        if i > 0:
            input_dalton.append(f'{val[-1]}\n')


    combs = list(product('XYZ', repeat = len(omegas)))
    combinations = []
    if len(operators) == 4:
        for i in combs:
            x = f'{i[0]}{available_ops[operators[0]]} {i[1]}{available_ops[operators[1]]} {i[2]}{available_ops[operators[2]]} {i[3]}{available_ops[operators[3]]}\n'
            combinations.append(x)
    if len(operators) == 3:
        for i in combs:
            x = f'{i[0]}{available_ops[operators[0]]} {i[1]}{available_ops[operators[1]]} {i[2]}{available_ops[operators[2]]} \n'
            combinations.append(x)
    if len(operators) == 2:
        for i in combs:
            x = f'{i[0]}{available_ops[operators[0]]} {i[1]}{available_ops[operators[1]]} \n'
            combinations.append(x)


    input_dalton.append('.OPERAT\n')
    input_dalton += combinations

    input_dalton.append('**END OF INPUT')

    with open('cc_input.dal', 'w') as dal_file:
        dal_file.writelines(input_dalton)

    generate_mol(state, basisset)


def generate_mol(state, basisset,*, charge=0):
    charges = state.reference_state.nuclear_charges
    atoms = []
    for i in charges:
        atoms.append(NUC[int(i)])
    atomtypes = set(atoms)
    coordinates = np.array(state.reference_state.coordinates)
    coordinates = coordinates.reshape((len(atoms),3))
    molecule = ''
    for i in atomtypes:
        if atoms.count(i) > 1:
            molecule += f'{i}{atoms.count(i)}'
        else:
            molecule += f'{i}'

    mol_input = [f"""BASIS
{basisset}
DALTON INPUT 
{molecule} - {basisset}
AtomTypes={len(atomtypes)} Charge={charge} NoSymmetry\n"""]

    for i in atomtypes:
        mol_input.append(f"""Atoms={atoms.count(i)} Charge={NUC_2[i]} \n""")
        for j, val  in enumerate(atoms):
            if i == val:
                 mol_input.append(f'{i} {coordinates[j][0]} {coordinates[j][1]} {coordinates[j][2]}\n')
    with open('molecule.mol', 'w') as mol_file:
        mol_file.writelines(mol_input)

if __name__ == '__main__':


    import adcc
    from sympy import symbols
    from responsefun.symbols_and_labels import *
    from responsefun.response_operators import (
            DipoleOperator,
            TransitionFrequency
    )
    from responsefun.sum_over_states import (
            TransitionMoment,
            SumOverStates
    )
    from responsefun.evaluate_property import (
            evaluate_property_isr,
            evaluate_property_sos_fast
    )
    from responsefun.adcc_properties import AdccProperties
    from pyscf import gto, scf

    mol = gto.M(
        atom="""
        C    0.0  0.0     0.0
        O    0.0  0.0     1.12832
        """,
        unit="angstrom",
        basis = "sto-3g"
    )
    mol.build()
    scfres = scf.RHF(mol)
    scfres.run(conv_tol = 1e-13)


    #run adc in adcc

    state = adcc.adc1(scfres, n_singlets=21)
    #sos object erstellen!

    #gamma term 1
    firstterm =  TransitionMoment(O, nabla_a, n) * TransitionMoment(n, nabla_b, m) * TransitionMoment(m, nabla_c, p) * TransitionMoment(p, nabla_d, O) / ((w_n - w_o) * (w_m - w_2 - w_3) * (w_p - w_3))
    gamma_sos = SumOverStates(firstterm, [n,m,p], perm_pairs = [(nabla_a, -w_o), (nabla_b, w_1), (nabla_c, w_2), (nabla_d, w_3)])
    generate_cas_input(state, gamma_sos, 'STO-3G', 12, 5, 12,1 ,omegas= [(w_o, w_1+w_2+w_3), (w_1, 0.0), (w_2, 0), (w_3,0)])

#    generate_cas_input(12, 5,12, 1,         operators = ['electric', 'electric', 'magnetic'],
#        omegas = [(w_o, w_1+w_2+w_3), (w_1, 0.0), (w_2, 0.0)]
#)
    #generate_cc_input('cc2')
    #generate_mol('COO','cc-pVTZ')
