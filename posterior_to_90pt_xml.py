import sys
import numpy as np
import lal as lal
import lalsimulation as lalsim
import glue
from glue.ligolw import ilwd
from glue.ligolw import ligolw
from glue.ligolw import lsctables
from glue.ligolw import utils
from glue.ligolw.utils import process
from pycbc import pnutils

approx = lalsim.IMRPhenomPv2
posterior = np.genfromtxt('/home/spxiwh/TMP/allIsp_post.dat',names=True)
sorted_posterior = \
    posterior[np.argsort(posterior['logprior']+posterior['logl'])][::-1]
num_points = len(sorted_posterior)
end_ninety = int(num_points * 0.9 + 0.5)

def _empty_row(obj):
    """Create an empty sim_inspiral or sngl_inspiral row where the columns have
    default values of 0.0 for a float, 0 for an int, '' for a string. The ilwd
    columns have a default where the index is 0.
    """

    # check if sim_inspiral or sngl_inspiral
    if obj == lsctables.SimInspiral:
        row = lsctables.SimInspiral()
        cols = lsctables.SimInspiralTable.validcolumns
    else:
        row = lsctables.SnglInspiral()
        cols = lsctables.SnglInspiralTable.validcolumns

    # populate columns with default values
    for entry in cols.keys():
        if cols[entry] in ['real_4','real_8']:
            setattr(row,entry,0.)
        elif cols[entry] == 'int_4s':
            setattr(row,entry,0)
        elif cols[entry] == 'lstring':
            setattr(row,entry,'')
        elif entry == 'process_id':
            row.process_id = ilwd.ilwdchar("sim_inspiral:process_id:0")
        elif entry == 'simulation_id':
            row.simulation_id = ilwd.ilwdchar("sim_inspiral:simulation_id:0")
        elif entry == 'event_id':
            row.event_id = ilwd.ilwdchar("sngl_inspiral:event_id:0")
        else:
            raise ValueError("Column %s not recognized." %(entry) )

    return row

# Start constructing my XML file

outdoc = ligolw.Document()
outdoc.appendChild(ligolw.LIGO_LW())

# Replace with actual opts dictionary
opts_dict = {}
# create process table
proc_id = process.register_to_xmldoc(
                    outdoc, sys.argv[0], opts_dict,
                    comment="", ifos=['H1L1'],
                    version=glue.git_version.id,
                    cvs_repository=glue.git_version.branch,
                    cvs_entry_time=glue.git_version.date).process_id

# create sim_inspiral table
sim_table = lsctables.New(lsctables.SimInspiralTable,
                          columns=lsctables.SimInspiralTable.validcolumns)
outdoc.childNodes[0].appendChild(sim_table)
# create sngl_inspiral table
sngl_table = lsctables.New(lsctables.SnglInspiralTable,
                           columns=lsctables.SnglInspiralTable.validcolumns)
outdoc.childNodes[0].appendChild(sngl_table)

for i in xrange(end_ninety):
    sngl = _empty_row(lsctables.SnglInspiral)
    sim = _empty_row(lsctables.SimInspiral)
    m1 = sorted_posterior[i]['m1']
    sngl.mass1 = m1
    sim.mass1 = m1
    m2 = sorted_posterior[i]['m2']
    sngl.mass2 = m2
    sim.mass2 = m2
    mc, eta = pnutils.mass1_mass2_to_mchirp_eta(m1, m2)
    mtot = m1 + m2
    sngl.mchirp = mc
    sngl.eta = eta
    sngl.mtotal = mtot
    sim.mchirp = mc
    sim.eta = eta
    dist = sorted_posterior[i]['distance']
    sim.distance = dist
    fmin = sorted_posterior[i]['flow']
    sim.f_lower = fmin
    # Canot store this in XML files!!
    fref = sorted_posterior[i]['f_ref']
    theta_jn = sorted_posterior[i]['theta_jn']
    phi_jl = sorted_posterior[i]['phi_jl']
    tilt1 = sorted_posterior[i]['tilt1']
    tilt2 = sorted_posterior[i]['tilt2']
    phi12 = sorted_posterior[i]['phi12']
    a1 = sorted_posterior[i]['a1']
    a2 = sorted_posterior[i]['a2']
    spins = lalsim.SimInspiralTransformPrecessingNewInitialConditions\
        (theta_jn, phi_jl, tilt1, tilt2, phi12, a1, a2, m1, m2, fref)
    sim.spin1x = spins[1]
    sim.spin1y = spins[2]
    sim.spin1z = spins[3]
    sim.spin2x = spins[4]
    sim.spin2y = spins[5]
    sim.spin2z = spins[6]
    sngl.spin1x = spins[1]
    sngl.spin1y = spins[2]
    sngl.spin1z = spins[3]
    sngl.spin2x = spins[4]
    sngl.spin2y = spins[5]
    sngl.spin2z = spins[6]


    inclination = sorted_posterior[i]['iota']
    phi_orb = sorted_posterior[i]['phi_orb']
    sim.coa_phase = phi_orb
    sim.inclination = inclination
    # Hacky column time -> These match what sbank is doing internally
    sngl.alpha5 = phi_orb
    sngl.alpha3 = inclination
    # PE doesn't store this. XML cannot write this.
    amp_order = -1
    ph_order = -1
    time = sorted_posterior[i]['time']
    ra = sorted_posterior[i]['ra']
    dec = sorted_posterior[i]['dec']
    psi = sorted_posterior[i]['psi']
    sim.latitude = ra
    sim.longitude = dec
    sim.polarization = psi
    # WARNING: Sbank actually stores *theta* in alpha1 (\pi/2 - latitude)
    #          Be careful about conversions with this
    sngl.alpha1 = ra
    sngl.alpha2 = dec
    sngl.alpha4 = psi
    sim.geocent_end_time = time
    sngl.end_time = time
    sim_table.append(sim)
    sngl_table.append(sngl)

utils.write_filename(outdoc, 'pe_posteriors.xml')
