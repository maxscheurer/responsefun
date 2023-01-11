import numpy as np
import zarr
import re
from scipy.constants import physical_constants
from scipy.special import binom

Hartree = physical_constants["hartree-electron volt relationship"][0]
dip_au = physical_constants["atomic unit of electric dipole mom."][0]
debye_in_au = 1/(2.9979e29)*1/(dip_au)

def read_dalton_cc2(outfile):
    with open(f'{outfile}') as file_1:
        lines = file_1.readlines()
        file_1.close()

    for i, line in enumerate(lines):
        if ".NCCEXC" in line:
            n_ex = lines[i +1].split()
            n_ex = int(n_ex[0])
    excitation_energies = np.zeros((n_ex))
    trans_dip = np.zeros((n_ex,3))
    trans_dip_mag = np.zeros((n_ex, 3))
    s2s = np.zeros((n_ex,n_ex, 3))
    s2s_mag = np.zeros((n_ex,n_ex, 3))
    gs_mag_dipole_moment = np.zeros((3))

    for i, line in enumerate(lines):
        if 'CC2        Excitation energies' in line:
            a = i + 4 
            for i, val in enumerate(lines[a:a+n_ex]):
                val = val.split()
                excitation_energies[i] = float(val[5])
        if 'CC2   Right transition moments in atomic units' in line:
            a = i+ 3
            x = lines[a:a + 3 * n_ex]
            for f in x:
                bla = f.split()
                if 'X' in bla[2]:
                    trans_dip[int(bla[0])-1][0] = float(bla[-1])
                if 'Y' in bla[2]:
                    trans_dip[int(bla[0])-1][1] = float(bla[-1])
                if 'Z' in bla[2]:
                    trans_dip[int(bla[0])-1][2] = float(bla[-1])
            b = a + 3 *n_ex
            x = lines[b:b + 3 * n_ex]
            for f in x:
                bla = f.split()
                if 'X' in bla[2]:
                    trans_dip_mag[int(bla[0])-1][0] = float(bla[-1])
                if 'Y' in bla[2]:
                    trans_dip_mag[int(bla[0])-1][1] = float(bla[-1])
                if 'Z' in bla[2]:
                    trans_dip_mag[int(bla[0])-1][2] = float(bla[-1])

        if 'Transition moments between excited states in atomic units(R):' in line:
            a = i + 3
            end =  2*6 *binom(n_ex,2)
            x = lines[a: a + int(end)]
            for f in x:
                bla = f.split()
                if 'XDIPLEN' in bla[6]:
                    s2s[int(bla[4])-1][int(bla[2])-1][0] = float(bla[-1])
                if 'YDIPLEN' in bla[6]:
                    s2s[int(bla[4])-1][int(bla[2])-1][1] = float(bla[-1])
                if 'ZDIPLEN' in bla[6]:
                    s2s[int(bla[4])-1][int(bla[2])-1][2] = float(bla[-1])
                if 'XA' in bla[6]:
                    s2s_mag[int(bla[4])-1][int(bla[2])-1][0] = float(bla[-1])
                if 'YA' in bla[6]:
                    s2s_mag[int(bla[4])-1][int(bla[2])-1][1] = float(bla[-1])
                if 'ZA' in bla[6]:
                    s2s_mag[int(bla[4])-1][int(bla[2])-1][2] = float(bla[-1])

        if 'Electronic contribution to operator' in line:
            a = i + 4
            if 'XANGMOM' in lines[a]:
                gs_mag_dipole_moment[0] = float(lines[a].split()[-1])
            if 'YANGMOM' in lines[a+4]:
                gs_mag_dipole_moment[1] = float(lines[a].split()[-1])
            if 'ZANGMOM' in lines[a+8]:
                gs_mag_dipole_moment[2] = float(lines[a].split()[-1])


    trans_dip_mag = 0.5 * trans_dip_mag
    s2s_mag = 0.5 * s2s_mag
    state_dips = np.zeros((n_ex,3))
    for i in range(n_ex):
        state_dips[i] = s2s[i][i]
    state_mag_dips = np.zeros((n_ex, 3))
    for i in range(n_ex):
        state_mag_dips[i] = s2s_mag[i][i]

    for i in range(n_ex):
        for j in range(n_ex):
            s2s[j,i] =  s2s[i,j]
            s2s_mag[j,i ] = - s2s_mag[i,j]


    zarr_storage = f'{outfile}.zarr'
    z = zarr.open(f'{outfile}.zarr', mode = 'w')
    z['ground_state/magnetic_dipole_moment'] = gs_mag_dipole_moment
    #z['ground_state/dipole_moment'] = gs_dipole_moment
    #z['ground_state/energy'] = gs_energy
    z['excitation_energy_uncorrected'] = excitation_energies
    z['excited_state/state_dipole_moment'] = state_dips
    z['excited_state/state_magnetic_dipole_moment'] = state_mag_dips
    z['excited_state/transition_dipole_moment'] = trans_dip
    z['excited_state/transition_magnetic_dipole_moment'] = trans_dip_mag
    z['s2s_transition_dipole_moment'] = s2s
    z['s2s_transition_magnetic_dipole_moment'] = s2s_mag

    return zarr_storage


def read_dalton_cis(outfile, outfile_s2s=None):
    with open(f'{outfile}') as file_1:
        lines_1 = file_1.readlines()
        file_1.close()
    excitation_energies = []
    for i, line_1 in enumerate(lines_1):
        if '@ Excitation energy ' in line_1:
            excitation_energies.append(line_1.split())
    excitation_energies = np.asarray(excitation_energies)
    excitation_energies = np.delete(excitation_energies, [0,1,2,3,5], axis =1)
    excitation_energies = excitation_energies.astype(float)
    n_ex = excitation_energies.size
    excitation_energies = np.reshape(excitation_energies, (n_ex))
    trans_dip = np.zeros((n_ex, 3))
    trans_mag_dip = np.zeros((n_ex, 3))
    trans_nabla = np.zeros((n_ex, 3))
    for i, line_1 in enumerate(lines_1):
        if  '@ Excited state no:' in line_1:
            x = line_1.split()
            excited_state = int(x[4])-1
            if 'XDIPLEN' in lines_1[i+8]:
                line = lines_1[i+9].split()
                trans_dip[excited_state][0] = float(line[-2])
            if 'YDIPLEN' in lines_1[i+11]:
                line = lines_1[i+12].split()
                trans_dip[excited_state][1] = float(line[-2])
            if 'ZDIPLEN' in lines_1[i+14]:
                line = lines_1[i+15].split()
                trans_dip[excited_state][2] = float(line[-2])
            if 'XANGMOM' in lines_1[i+17]:
                line = lines_1[i+18].split()
                trans_mag_dip[excited_state][0] = float(line[-1])
            if 'YANGMOM' in lines_1[i+20]:
                line = lines_1[i+21].split()
                trans_mag_dip[excited_state][1] = float(line[-1])
            if 'ZANGMOM' in lines_1[i+23]:
                line = lines_1[i+24].split()
                trans_mag_dip[excited_state][2] = float(line[-1])
            if 'XDIPVEL' in lines_1[i+8]:
                line = lines_1[i+9].split()
                trans_nabla[excited_state][0] = float(line[-2])
            if 'YDIPVEL' in lines_1[i+11]:
                line = lines_1[i+12].split()
                trans_nabla[excited_state][1] = float(line[-2])
            if 'ZDIPVEL' in lines_1[i+14]:
                line = lines_1[i+15].split()
                trans_nabla[excited_state][2] = float(line[-2])
            if 'XDIPVEL' in lines_1[i+17]:
                line = lines_1[i+18].split()
                trans_nabla[excited_state][0] = float(line[-2])
            if 'YDIPVEL' in lines_1[i+20]:
                line = lines_1[i+21].split()
                trans_nabla[excited_state][1] = float(line[-2])
            if 'ZDIPVEL' in lines_1[i+23]:
                line = lines_1[i+24].split()
                trans_nabla[excited_state][2] = float(line[-2])
            if 'XDIPLEN' in lines_1[i+17]:
                line = lines_1[i+18].split()
                trans_dip[excited_state][0] = float(line[-2])
            if 'YDIPLEN' in lines_1[i+20]:
                line = lines_1[i+21].split()
                trans_dip[excited_state][1] = float(line[-2])
            if 'ZDIPLEN' in lines_1[i+23]:
                line = lines_1[i+24].split()
                trans_dip[excited_state][2] = float(line[-2])
    s2s_dips = np.zeros((n_ex, n_ex, 3))
    s2s_mag_dips = np.zeros((n_ex, n_ex, 3))
    s2s_nabla = np.zeros((n_ex, n_ex, 3))
    if not outfile_s2s == None:
        with open(f'{outfile_s2s}') as file_s2s:
            lines_s2s = file_s2s.readlines()
            file_s2s.close()
        for i, line_s2s in enumerate(lines_s2s):
            if '@ A operator label,    symmetry, spin:      XDIPLEN' in line_s2s:
                from_state = lines_s2s[i+2].split()
                from_state = int(from_state[7])-1
                to_state = lines_s2s[i+1].split()
                to_state = int(to_state[7])-1
                moment = lines_s2s[i+4].split()
                moment = float(moment[-1])
                s2s_dips[from_state][to_state][0] = moment
            if '@ A operator label,    symmetry, spin:      YDIPLEN' in line_s2s:
                from_state = lines_s2s[i+2].split()
                from_state = int(from_state[7])-1
                to_state = lines_s2s[i+1].split()
                to_state = int(to_state[7])-1
                moment = lines_s2s[i+4].split()
                moment = float(moment[-1])
                s2s_dips[from_state][to_state][1] = moment
            if '@ A operator label,    symmetry, spin:      ZDIPLEN' in line_s2s:
                from_state = lines_s2s[i+2].split()
                from_state = int(from_state[7])-1
                to_state = lines_s2s[i+1].split()
                to_state = int(to_state[7])-1
                moment = lines_s2s[i+4].split()
                moment = float(moment[-1])
                s2s_dips[from_state][to_state][2] = moment
            if '@ A operator label,    symmetry, spin:      XANGMOM' in line_s2s:
                from_state = lines_s2s[i+2].split()
                from_state = int(from_state[7])-1
                to_state = lines_s2s[i+1].split()
                to_state = int(to_state[7])-1
                moment = lines_s2s[i+4].split()
                moment = float(moment[-1])
                s2s_mag_dips[from_state][to_state][0] = moment
            if '@ A operator label,    symmetry, spin:      YANGMOM' in line_s2s:
                from_state = lines_s2s[i+2].split()
                from_state = int(from_state[7])-1
                to_state = lines_s2s[i+1].split()
                to_state = int(to_state[7])-1
                moment = lines_s2s[i+4].split()
                moment = float(moment[-1])
                s2s_mag_dips[from_state][to_state][1] = moment
            if '@ A operator label,    symmetry, spin:      ZANGMOM' in line_s2s:
                from_state = lines_s2s[i+2].split()
                from_state = int(from_state[7])-1
                to_state = lines_s2s[i+1].split()
                to_state = int(to_state[7])-1
                moment = lines_s2s[i+4].split()
                moment = float(moment[-1])
                s2s_mag_dips[from_state][to_state][2] = moment
            if '@ A operator label,    symmetry, spin:      XDIPVEL' in line_s2s:
                from_state = lines_s2s[i+2].split()
                from_state = int(from_state[7])-1
                to_state = lines_s2s[i+1].split()
                to_state = int(to_state[7])-1
                moment = lines_s2s[i+4].split()
                moment = float(moment[-1])
                s2s_nabla[to_state][from_state][0] = moment
            if '@ A operator label,    symmetry, spin:      YDIPVEL' in line_s2s:
                from_state = lines_s2s[i+2].split()
                from_state = int(from_state[7])-1
                to_state = lines_s2s[i+1].split()
                to_state = int(to_state[7])-1
                moment = lines_s2s[i+4].split()
                moment = float(moment[-1])
                s2s_nabla[to_state][from_state][1] = moment
            if '@ A operator label,    symmetry, spin:      ZDIPVEL' in line_s2s:
                from_state = lines_s2s[i+2].split()
                from_state = int(from_state[7])-1
                to_state = lines_s2s[i+1].split()
                to_state = int(to_state[7])-1
                moment = lines_s2s[i+4].split()
                moment = float(moment[-1])
                s2s_nabla[to_state][from_state][2] = moment
    state_dips = np.zeros((n_ex,3))
    for i in range(n_ex):
        state_dips[i] = s2s_dips[i][i]
    state_mag_dips = np.zeros((n_ex, 3))
    for i in range(n_ex):
        state_mag_dips[i] = s2s_mag_dips[i][i]
   
    for i in range(n_ex):
        for j in range(n_ex):
            s2s_dips[i, j] =  s2s_dips[j , i]
            s2s_mag_dips[i,j ] = - s2s_mag_dips[j,i]
            s2s_nabla[j,i ] = - s2s_nabla[i,j]
    state_mag_dips = 0.5 * state_mag_dips
    trans_mag_dip = 0.5 * trans_mag_dip
    s2s_mag_dips = 0.5 * s2s_mag_dips
    zarr_storage = f'{outfile}.zarr'
    z = zarr.open(f'{outfile}.zarr', mode = 'w')
    #z['ground_state/dipole_moment'] = gs_dipole_moment
    #z['ground_state/energy'] = gs_energy
    z['excitation_energy_uncorrected'] = excitation_energies
    z['excited_state/state_dipole_moment'] = state_dips
    z['excited_state/state_magnetic_dipole_moment'] = state_mag_dips
    z['excited_state/transition_dipole_moment'] = trans_dip
    z['excited_state/transition_magnetic_dipole_moment'] = trans_mag_dip
    z['excited_state/transition_nabla'] = trans_nabla
    z['s2s_transition_dipole_moment'] = s2s_dips
    z['s2s_transition_magnetic_dipole_moment'] = s2s_mag_dips
    z['s2s_nabla'] = s2s_nabla

    print(type(z))
    return zarr_storage


def qchem_read_fci(outdatei):
    
    count = 0

    with open(outdatei) as f:
        lines = f.readlines()
        f.close

    excitation_energies = []
    transition_dipole_moments = []
    state_dipole_moments = []
    #s2s = np.zeros((n_ex, n_ex, 3))

    for count, line in enumerate(lines):
        #groundstate
        if 'state   1:' in line:
            a = count + 3
            gs_dipole = lines[a].split()
            gs_dipole = np.asarray(gs_dipole)
            gs_dipole = np.delete(gs_dipole, [0,1,3,5,7])
            gs_dipole = gs_dipole.astype(float)
        
        if 'Excitation energy' in line:
            excitation_energies.append(line.split())

        if 'Dipole Moment:' in line:
            state_dipole_moments.append(line.split())

        if 'Trans. Moment:' in line:
            transition_dipole_moments.append(line.split())


    excitation_energies = np.asarray(excitation_energies)
    excitation_energies = np.delete(excitation_energies, [0,1,2,3], axis =1)
    excitation_energies = excitation_energies.astype(float)
    excitation_energies = excitation_energies / Hartree
    n_ex = excitation_energies.size
    excitation_energies = np.reshape(excitation_energies, (n_ex))
    excitation_energies = np.delete(excitation_energies, [0], axis =0)

    state_dipole_moments = np.asarray(state_dipole_moments)
    state_dipole_moments = np.delete(state_dipole_moments, [0,1,3,5, -1], axis = 1)
    state_dipole_moments = state_dipole_moments.astype(float)
    state_dipole_moments = np.delete(state_dipole_moments, [0], axis = 0)

    transition_dipole_moments = np.asarray(transition_dipole_moments)
    transition_dipole_moments = np.delete(transition_dipole_moments, [0,1,3,5, -1], axis = 1)
    transition_dipole_moments = transition_dipole_moments.astype(float)
    
    #TODO: s2s 
    zarr_storage = f'{outdatei}.zarr'
    z = zarr.open(f'{outdatei}.zarr', mode = 'w')
    z['ground_state/dipole_moment'] = gs_dipole
    #z['ground_state/energy'] = gs_energy
    z['excitation_energy_uncorrected'] = excitation_energies
    z['excited_state/state_dipole_moment'] = state_dipole_moments
    z['excited_state/transition_dipole_moment'] = transition_dipole_moments
    #z['s2s_transition_dipole_moment'] = s2s

    return zarr_storage



def qchem_read_ccsd(outdatei, out_datei_2):

    count = 0
    with open(outdatei) as f:
        lines = f.readlines()
        f.close()
    excitation_energies = []

    for count, line in enumerate(lines):
        #groundstate
        if 'Dipole Moment (Debye)' in line: 
            a = count + 1
            gs_dipole = lines[a].split()
            gs_dipole = np.asarray(gs_dipole)
            gs_dipole = np.delete(gs_dipole, [0,2,4])
            gs_dipole = gs_dipole.astype(float)
            gs_dipole = gs_dipole  * debye_in_au

        if 'Excitation energy =' in line:
            excitation_energies.append(line.split())
    
    excitation_energies = np.asarray(excitation_energies)
    excitation_energies = np.delete(excitation_energies, [0,1,2,3,4,5,6,7,9], axis =1)
    excitation_energies = excitation_energies.astype(float)
    n_ex = excitation_energies.size
    excitation_energies = excitation_energies / Hartree
    excitation_energies = np.reshape(excitation_energies, (n_ex))

    transition_dipole_moments = np.zeros((n_ex,3))
    state_dipole_moments = []
    s2s = np.zeros((n_ex, n_ex, 3))

    for count, line in enumerate(lines):
        if 'Dipole moment (a.u.):' in line:
            state_dipole_moments.append(line.split())

        if 'State A:' in line:
            for i in range(1,n_ex+1):
                for j in range (1, n_ex+1):
                    if f'{i}/' in line:
                        a = count + 3
                        b = count + 1
                        if f'{j}/' in lines[b]:
                            y = lines[a].split()
                            y = np.asarray(y)
                            y = np.delete(y, [0,1,2,4,6])
                            y[-1] = y[-1].removesuffix(')')
                            y[-2] = y[-2].removesuffix(',')
                            y[-3] = y[-3].removesuffix(',')
                            s2s[i-1][j-1] = y

                    if f'{i}/' in line:
                        a = count + 4
                        b = count + 1
                        if f'{j}/' in lines[b]:
                            y = lines[a].split()
                            y = np.asarray(y)
                            y = np.delete(y, [0,1,2,4,6])
                            y[-1] = y[-1].removesuffix(')')
                            y[-2] = y[-2].removesuffix(',')
                            y[-3] = y[-3].removesuffix(',')
                            s2s[j-1][i-1] = y

    with open(out_datei_2) as f:
        lines = f.readlines()
        f.close()

    for count, line in enumerate(lines):
        if 'State B:' in line:
            for i in range(1,n_ex+1):
                if f'{i}/' in line:
                    a = count + 2
                    y = lines[a].split()
                    y = np.asarray(y)
                    y = np.delete(y, [0,1,2,4,6])
                    y[-1] = y[-1].removesuffix(')')
                    y[-2] = y[-2].removesuffix(',')
                    y[-3] = y[-3].removesuffix(',')
                    transition_dipole_moments[i-1] = y

    state_dipole_moments = np.asarray(state_dipole_moments)
    state_dipole_moments = np.delete(state_dipole_moments, [0,1,2,3,4,6,8], axis = 1)
    for i in range(n_ex):
        state_dipole_moments[i][-1]=state_dipole_moments[i][-1].removesuffix(')')
        state_dipole_moments[i][-2]=state_dipole_moments[i][-2].removesuffix(',')
        state_dipole_moments[i][-3]=state_dipole_moments[i][-3].removesuffix(',')
    state_dipole_moments = state_dipole_moments.astype(float)

    for i in range(0, n_ex):
        s2s[i][i] = state_dipole_moments[i]

    zarr_storage = f'{outdatei}.zarr'
    z = zarr.open(f'{outdatei}.zarr', mode = 'w')
    z['ground_state/dipole_moment'] = gs_dipole
    #z['ground_state/energy'] = gs_energy
    z['excitation_energy_uncorrected'] = excitation_energies
    z['excited_state/state_dipole_moment'] = state_dipole_moments
    z['excited_state/transition_dipole_moment'] = transition_dipole_moments
    z['s2s_transition_dipole_moment'] = s2s

    return zarr_storage


def qchem_read_adc(outdatei):

    count = 0
    with open(outdatei) as f:
        lines = f.readlines()
        f.close()
    excitation_energies = []

    for count, line in enumerate(lines):
        #excitation energies
        if 'excitation energy' in line:
            excitation_energies.append(line.split())

    excitation_energies = np.asarray(excitation_energies)
    excitation_energies = np.delete(excitation_energies, [0,1,2,3,4,6,7], axis =1)
    excitation_energies = excitation_energies.astype(float)
    n_ex = excitation_energies.size
    excitation_energies = np.reshape(excitation_energies, (n_ex))

    transition_dipole_moments = []
    state_dipole_moments = []
    s2s = np.zeros((n_ex, n_ex, 3))
    for count, line in enumerate(lines):
        #GS energy + dipole
        if 'MP(' in line:
            a = count + 3
            f = lines[a].split()
            gs_energy = np.asarray(f)
            gs_energy = np.delete(gs_energy, [0,1,3])
            gs_energy = gs_energy.astype(float)

            b = count +4
            gs_dipole = lines[b].split()
            gs_dipole = np.asarray(gs_dipole)
            gs_dipole = np.delete(gs_dipole, [0,1,2,3])
            gs_dipole[-1] = gs_dipole[-1].removesuffix(']')
            gs_dipole[-2]= gs_dipole[-2].removesuffix(',')
            gs_dipole[-3] = gs_dipole[-3].removesuffix(',')
            gs_dipole = gs_dipole.astype(float)

        #state_dipole_moments
        for i in range(1, n_ex):
            if f'Excited state {i}' in line:
                a = count + 8
                transition_dipole_moments.append(lines[a].split())
                b = count + 16
                state_dipole_moments.append(lines[b].split())

        #s2si dipole moments
        for i in range(1, n_ex+2):
            for j in range(1,n_ex+2):
                if 'Transition from excited state' in line:
                    x = re.sub('[a-zA-z]', '', line)
                    x = re.sub(':','',x)
                    x = re.sub(' +',' ',x)
                    x += '_ '
                    if f'{i} {j}\n_' in x:
                        a = count +5
                        f =lines[a].split()
                        f =np.asarray(f)
                        f = np.delete(f, [0,1,2,3,4])
                        f[-1] = f[-1].removesuffix(']')
                        f[-2] = f[-2].removesuffix(',')
                        f[-3] = f[-3].removesuffix(',')
                        f = f.astype(float)
                        s2s[i-1][j-1] = f

    transition_dipole_moments = np.asarray(transition_dipole_moments)
    transition_dipole_moments = np.delete(transition_dipole_moments, [0,1,2,3,4], axis =1)
    for i in range(n_ex):
        transition_dipole_moments[i][-1]=transition_dipole_moments[i][-1].removesuffix(']')
        transition_dipole_moments[i][-2]=transition_dipole_moments[i][-2].removesuffix(',')
        transition_dipole_moments[i][-3]=transition_dipole_moments[i][-3].removesuffix(',')
    transition_dipole_moments = transition_dipole_moments.astype(float)

    state_dipole_moments = np.asarray(state_dipole_moments)
    state_dipole_moments = np.delete(state_dipole_moments, [0,1,2,3], axis = 1)
    for i in range(n_ex):
        state_dipole_moments[i][-1]=state_dipole_moments[i][-1].removesuffix(']')
        state_dipole_moments[i][-2]=state_dipole_moments[i][-2].removesuffix(',')
        state_dipole_moments[i][-3]=state_dipole_moments[i][-3].removesuffix(',')
    state_dipole_moments = state_dipole_moments.astype(float)
    for i in range(0, n_ex):
        s2s[i][i] = state_dipole_moments[i]
    for i in range(0,n_ex):
        for l in range(0,n_ex):
            s2s[l][i]=s2s[i][l]
    #s2s = np.reshape(s2s, (n_ex*n_ex*3))
    zarr_storage = f'{outdatei}.zarr'
    z = zarr.open(f'{outdatei}.zarr', mode = 'w')
    z['ground_state/dipole_moment'] = gs_dipole
    z['ground_state/energy'] = gs_energy
    z['excitation_energy_uncorrected'] = excitation_energies
    z['excited_state/state_dipole_moment'] = state_dipole_moments
    z['excited_state/transition_dipole_moment'] = transition_dipole_moments
    z['s2s_transition_dipole_moment'] = s2s

    return zarr_storage


def qchem_read_tddft(outdatei):
    count = 0
    with open(outdatei) as f:
        lines = f.readlines()
        f.close()
    excitations = []


    for count, line in enumerate(lines):
    #excitation energies auslesen
        if 'excitation energy (eV) =' in line:
            excitations.append(line.split())
    excitation_energies = np.asarray(excitations)
    excitation_energies = np.delete(excitation_energies, [0,1,2,3,4,5,6],axis = 1)
    excitation_energies = excitation_energies.astype(float)
    excitation_energies = excitation_energies / Hartree
    n_ex = excitation_energies.size
    excitation_energies = np.reshape(excitation_energies, (n_ex))


    lenght = int(0.5 * (n_ex -1)*n_ex)
    for count, line in enumerate(lines):
        #state dipole moments
        if 'Electron Dipole Moments of Singlet Excited State' in line:
            a = count + 4
            f = lines[a:a+n_ex]
            array = np.empty(shape = [0,3])
            for line in f:
                x = line.split()
                y = np.array(x)
                y = np.delete(y, 0)
                y = y.astype(float)
                array =np.append(array, y)
            state_dipole_moments = np.reshape(array, (n_ex,3))

        #transition state dipole moments
        if 'Transition Moments Between Ground and Singlet Excited States' in line:
            x = count +4
            f = lines[x:x+n_ex]
            array = np.empty(shape = [0,3])
            for line in f:
                x = line.split()
                y = np.array(x)
                y = np.delete(y, [0,1,5])
                y = y.astype(float)
                array = np.append(array,y)
            transition_dipole_moments =np.reshape(array, (n_ex,3))

        #state to state transition dipole moments
        if 'Transition Moments Between Singlet Excited States' in line:
            x = count + 4
            f = lines[x:x+lenght]
            array = np.empty(shape = [0,0,3])
            for line in f:
                x = line.split()
                y = np.array(x)
                y = np.delete(y, [5])
                y = y.astype(float)
                array = np.append(array,y)
            state_to_state =np.reshape(array, (lenght,5))

        #ground state energy
        if 'Total energy in the final basis' in line:
            f = line.split()
            array = np.array(f)
            array = np.delete(array ,[0,1,2,3,4,5,6,7])
            gs_energy = array.astype(float)

        #ground state dipole moment
        if 'Electron Dipole Moments of Ground State' in line:
            x = count + 4
            f = lines[x].split()
            y = np.array(f)
            y = np.delete(y, 0)
            gs_dipole_moment = y.astype(float)


    s2s_tdms = np.zeros((n_ex,n_ex,3))
    for i in range(0, n_ex):
        s2s_tdms[i][i] = state_dipole_moments[i]

    for i in range(0,lenght):
        for j in range(0,n_ex):
            if state_to_state[i][0] == j+1:
                bla =int(0.5*((n_ex-2)*(n_ex-3)-(n_ex-2-j)*(n_ex-j-3))+j-1)
                s2s_tdms[j][i-bla][0] = state_to_state[i][2]
                s2s_tdms[j][i-bla][1] = state_to_state[i][3]
                s2s_tdms[j][i-bla][2] = state_to_state[i][4]

    for i in range(0,n_ex):
        for l in range(0,n_ex):
            s2s_tdms[l][i]=s2s_tdms[i][l]

    zarr_storage = f'{outdatei}.zarr'
    z = zarr.open(f'{outdatei}.zarr', mode = 'w')
    z['ground_state/dipole_moment'] = gs_dipole_moment
    z['ground_state/energy'] = gs_energy
    z['excitation_energy_uncorrected'] = excitation_energies
    z['excited_state/state_dipole_moment'] = state_dipole_moments
    z['excited_state/transition_dipole_moment'] = transition_dipole_moments
    z['s2s_transition_dipole_moment'] = s2s_tdms

    return zarr_storage


if __name__ == "__main__":

    out_datei = '/export/home/fschneid/Masterarbeit/Dalton/nabla/cis_molecule.out'
    #out_datei_2 = '/export/home/fschneid/Masterarbeit/Dalton/CO2_S2S/cis_molecule.out'
    x = read_dalton_cis(out_datei, None)

