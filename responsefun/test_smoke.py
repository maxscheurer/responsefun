"""Test smoke for responsefun package by running a simple RIXS calculation."""

def test_smoke_rixs():
    from scipy.constants import physical_constants
    from pyscf import gto, scf
    import adcc
    from responsefun.symbols_and_labels import (
        op_a, op_b, f, n, O, gamma, w_n, w
    )
    from responsefun.SumOverStates import TransitionMoment
    from responsefun.evaluate_property import evaluate_property_isr

    Hartree = physical_constants["hartree-electron volt relationship"][0]

    # run SCF in PySCF
    mol = gto.M(
        atom="""
        O 0 0 0
        H 0 0 1.795239827225189
        H 1.693194615993441 0 -0.599043184453037
        """,
        unit="Bohr",
        basis="sto-3g",
    )
    scfres = scf.RHF(mol)
    scfres.kernel()

    # run ADC calculation using adcc
    state = adcc.adc2(scfres, n_singlets=5)

    # compute RIXS tensor within the rotating wave approximation
    rixs_sos_expr = (
        TransitionMoment(f, op_a, n) * TransitionMoment(n, op_b, O) / (w_n - w - 1j*gamma)
    )
    rixs_tens = evaluate_property_isr(
        state, rixs_sos_expr, [n], omegas=[(w, 534.74/Hartree)], gamma_val=0.124/Hartree, final_state=(f, 2)
    )
    print(rixs_tens)