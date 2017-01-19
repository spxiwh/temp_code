# Copyright (C) 2011  Nickolas Fotopoulos, Stephen Privitera
#
# This program is free software; you can redistribute it and/or modify it
# under the terms of the GNU General Public License as published by the
# Free Software Foundation; either version 2 of the License, or (at your
# option) any later version.
#
# This program is distributed in the hope that it will be useful, but
# WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the GNU General
# Public License for more details.
#
# You should have received a copy of the GNU General Public License along
# with this program; if not, write to the Free Software Foundation, Inc.,
# 51 Franklin Street, Fifth Floor, Boston, MA  02110-1301, USA.

from __future__ import division

from time import strftime
from collections import deque
import numpy as np
import sys, os
import h5py

from scipy.interpolate import UnivariateSpline
from glue.ligolw import ligolw
from glue.ligolw import lsctables
from glue.ligolw import table
from glue.ligolw import utils
from glue.ligolw import ilwd
from glue.ligolw.utils import process as ligolw_process
from lal import REAL8FrequencySeries

from optparse import OptionParser

#from sbank import git_version FIXME
from lalinspiral.sbank.bank import Bank
from lalinspiral.sbank.tau0tau3 import proposals
from lalinspiral.sbank.psds import noise_models, read_psd, get_PSD
from lalinspiral.sbank import waveforms
from lalinspiral import CreateSBankWorkspaceCache
from lalinspiral import InspiralSBankComputeMatch, InspiralSBankComputeRealMatch, InspiralSBankComputeMatchMaxSkyLoc, InspiralSBankComputeMatchMaxSkyLocNoPhase

import lal
import lalsimulation as lalsim

class ContentHandler(ligolw.LIGOLWContentHandler):
    pass
lsctables.use_in(ContentHandler)

def parse_command_line():

    parser = OptionParser(usage = "")

    #
    # waveform options
    #
    parser.add_option("--approximant", choices=waveforms.waveforms.keys(), metavar='|'.join(waveforms.waveforms.keys()), default=None, help="Required. Specify the approximant to use for waveform generation.")

    #
    # mass parameter options
    #
    parser.add_option("--mass1-min",help="Required. Set minimum mass of the first component.", type="float", metavar="MASS")
    parser.add_option("--mass1-max",help="Required. Set maximum mass of the first component.", type="float", metavar="MASS")
    parser.add_option("--mass2-min",help="Set minimum mass of the second component. If not specified, the mass limits provided on the first component will be assumed for the second component.", type="float", metavar="MASS")
    parser.add_option("--mass2-max",help="Set maximum mass of the second component. If not specified, the mass limits provided on the first component will be assumed for the second component.", type="float", metavar="MASS")
    parser.add_option("--mtotal-min", help="Set minimum total mass of the system.", type="float", metavar="MASS")
    parser.add_option("--mtotal-max", help="Set maximum total mass of the system.",  type="float", metavar="MASS")
    parser.add_option("--mratio-min", dest="qmin", help="Set minimum allowed mass ratio of the system (convention is that q=m1/m2).", metavar="RATIO", type="float", default=1.0)
    parser.add_option("--mratio-max", dest="qmax", help="Set maximum allowed mass ratio of the system (convention is that q=m1/m2).", metavar="RATIO", type="float")
    parser.add_option("--mchirp-min", help="Deprecated. Set minimum chirp-mass of the system (in solar masses)", type="float")
    parser.add_option("--mchirp-max", help="Deprecated. Set maximum chirp-mass of the system (in solar masses)", type="float")


    #
    # spin parameter options
    #
    parser.add_option("--spin1-min", help="Set minimum allowed value for the spin of the first component. If spins are aligned, this parameter is interpreted as the projection of the spin vector along the orbital angualr momentum and can be positive or negative. If the spins are not aligned, this parameter is interpreted as the magnitude of the spin vector and must be positive.", type="float", default = None, metavar="SPIN")
    parser.add_option("--spin1-max", help="Set maximum allowed value for the spin of the first component.", type="float", default = None, metavar="SPIN")
    parser.add_option("--spin2-min", help="Set minimum allowed value for the spin of the second component. If not specified, the spin2 limits will equal the spin1 limits.", type="float", default = None, metavar="SPIN")
    parser.add_option("--spin2-max", help="Set maximum allowed value for the spin of the second component.", type="float", default = None, metavar="SPIN")
    parser.add_option("--aligned-spin", action="store_true", default=False, help="Only generate templates whose spins are parallel to the orbital angular momentum.")
    parser.add_option("--ns-bh-boundary-mass", type=float, metavar="MASS", help="Use spin bounds based on whether the object is a black hole or a neutron star. Objects with mass smaller (larger) than the given value are considered NSs (BHs) and use spin bounds given by --ns-spin-{min,max} (--bh-spin-{min,max}) rather than --spin{1,2}-{min,max}.")
    parser.add_option("--bh-spin-min", type=float, metavar="SPIN", help="Minimum spin for black holes when using --ns-bh-boundary-mass.")
    parser.add_option("--bh-spin-max", type=float, metavar="SPIN", help="Maximum spin for black holes when using --ns-bh-boundary-mass.")
    parser.add_option("--ns-spin-min", type=float, metavar="SPIN", help="Minimum spin for neutron stars when using --ns-bh-boundary-mass.")
    parser.add_option("--ns-spin-max", type=float, metavar="SPIN", help="Maximum spin for neutron stars when using --ns-bh-boundary-mass.")

    #
    # initial condition options
    #
    parser.add_option("--seed", help="Set the seed for the random number generator used by SBank for waveform parameter (masss, spins, ...) generation.", metavar="INT", default=1729, type="int")
    parser.add_option("--pe-samples", metavar="FILE",
                      help="Filtered PE posteriors.")

    #
    # noise model options
    #
    parser.add_option("--noise-model", choices=noise_models.keys(), metavar='|'.join(noise_models.keys()), default="aLIGOZeroDetHighPower", help="Choose a noise model for the PSD from a set of available analytical model.")
    parser.add_option("--reference-psd", help="Read PSD from an xml file instead of using analytical noise model. The PSD is assumed to be infinite beyond the maximum frequency contained in the file. This effectively sets the upper frequency cutoff to that frequency, unless a smaller frequency is given via --fhigh-max.", metavar="FILE")
    parser.add_option("--instrument", metavar="IFO", help="Specify the instrument from input PSD file for which to generate a template bank.")

    #
    # match calculation options
    #
    parser.add_option("--flow", type="float", help="Required. Set the low-frequency cutoff to use for the match caluclation.")
    parser.add_option("--match-min", help="Keep points within this.", type="float", default=0.95)
    #parser.add_option("--convergence-threshold", metavar="N", help="Set the criterion for convergence of the stochastic bank. The code terminates when there are N rejected proposals for each accepted proposal, averaged over the last ten acceptances. Default 1000.", type="int", default=1000)
    #parser.add_option("--max-new-templates", metavar="N", help="Use this option to force the code to exit after accepting a specified number N of new templates. Note that the code may exit with fewer than N templates if the convergence criterion is met first.", type="int", default=float('inf'))
    parser.add_option("--cache-waveforms", default = False, action="store_true", help="A given waveform in the template bank will be used many times throughout the bank generation process. You can save a considerable amount of CPU by caching the waveform from the first time it is generated; however, do so only if you are sure that storing the waveforms in memory will not overload the system memory.")
    parser.add_option("--coarse-match-df", type="float", default=None, help="If given, use this value of df to quickly test if the mismatch is less than 4 times the minimal mismatch. This can quickly reject points at high values of df, that will not have high overlaps at smaller df values. This can be used to speed up the sbank process.")
    parser.add_option("--iterative-match-df-max", type="float", default=None, help="If this option is given it will enable sbank using larger df values than 1 / data length when computing overlaps. Sbank will then compute a match at this value, and at half this value, if the two values agree to 0.1% the value obtained will be taken as the actual value. If the values disagree the match will be computed again using a df another factor of 2 smaller until convergence or a df of 1/ data_length, is reached.")
    parser.add_option("--fhigh-max", type="float", default=None, help="If given, generate waveforms and compute matches only to this frequency. The number will be rounded up to the nearest power of 2.")
    parser.add_option("--neighborhood-size", metavar="N", default = 0.25, type="float", help="Specify the window size in seconds to define \"nearby\" templates used to compute the match against each proposed template. The neighborhood is chosen symmetric about the proposed template; \"nearby\" is defined using the option --neighborhood-type. The default value of 0.25 is *not a guarantee of performance*. Choosing the neighborhood too small will lead to larger banks (but also higher bank coverage).")
    parser.add_option("--neighborhood-param", default="tau0", choices=["tau0","dur"], help="Choose how the neighborhood is sorted for match calculations.")

    #
    # output options
    #
    parser.add_option("--output-filename", default=None, help="Required. Name for output template bank. May not clash with seed bank.")
    parser.add_option("--verbose", default=False,action="store_true", help="Be verbose and write diagnostic information out to file.")

    opts, args = parser.parse_args()

    #
    # check for required arguments
    #
    for opt in ("flow", "match_min", "output_filename"):
        if getattr(opts, opt) is None:
            parser.error("--%s is required" % opt.replace("_", "-"))

    if opts.qmin < 1:
        parser.error("Mass ratio is assumed to be >= 1.")

    numeric_spin_opt_presence = [getattr(opts, x + '_' + y) is not None \
                                 for x in ['spin1', 'spin2'] for y in ['min', 'max']]
    all_numeric_spin_opts = False not in numeric_spin_opt_presence
    any_numeric_spin_opts = True in numeric_spin_opt_presence

    nsbh_spin_opt_presence = [getattr(opts, x + '_' + y) is not None \
                              for x in ['bh_spin', 'ns_spin'] for y in ['min', 'max']]
    all_nsbh_spin_opts = False not in nsbh_spin_opt_presence
    any_nsbh_spin_opts = True in nsbh_spin_opt_presence

    if any_numeric_spin_opts and any_nsbh_spin_opts:
        parser.error("conflicting specification of spin bounds")
    if any_nsbh_spin_opts and opts.ns_bh_boundary_mass is None:
        parser.error("NSBH spin bounds require --ns-bh-boundary-mass")
    if opts.ns_bh_boundary_mass is not None and not any_nsbh_spin_opts:
        parser.error("--ns-bh-boundary-mass requires NSBH spin bounds (--bh-spin-* etc)")

    if all_numeric_spin_opts:
        if not -1 <= opts.spin1_min <= opts.spin1_max <=1:
            parser.error("unphysical spin1 bounds: [%.2f, %.2f]" % (opts.spin1_min, opts.spin1_max))
        if not -1 <= opts.spin2_min <= opts.spin2_max <=1:
            parser.error("unphysical spin2 bounds: [%.2f, %.2f]" % (opts.spin2_min, opts.spin2_max))
    elif all_nsbh_spin_opts:
        if not -1 <= opts.bh_spin_min <= opts.bh_spin_max <= 1:
            parser.error("unphysical BH spin bounds: [%.2f, %.2f]" % (opts.bh_spin_min, opts.bh_spin_max))
        if not -1 <= opts.ns_spin_min <= opts.ns_spin_max <= 1:
            parser.error("unphysical NS spin bounds: [%.2f, %.2f]" % (opts.ns_spin_min, opts.ns_spin_max))
    else:
        # default spin bounds
        if opts.spin1_min is None:
            opts.spin1_min = -1
        if opts.spin1_max is None:
            opts.spin1_max = 1
        if opts.spin2_min is None:
            opts.spin2_min = opts.spin1_min
        if opts.spin2_max is None:
            opts.spin2_max = opts.spin1_max


    return opts, args


#
# begin main
#
opts, args = parse_command_line()

print "PARSED COMMAND LINE"

#
# choose noise model
#
if opts.reference_psd is not None:

    if opts.reference_psd.endswith(".txt") or opts.reference_psd.endswith(".txt.gz") or opts.reference_psd.endswith(".dat"):
        # assume psd file is a two-column ASCII formatted file
        data = np.loadtxt(opts.reference_psd)
        f_orig, psddata = data[:,0], data[:,1]

    elif opts.reference_psd.endswith(".xml") or opts.reference_psd.endswith(".xml.gz"):
        # assume psd file is formatted as a LIGOLW XML
        psddict = read_psd(opts.reference_psd)
        if opts.instrument:
            psd = psddict[opts.instrument]
        elif len(psddict.keys()) == 1:
            psd = psddict[psddict.keys()[0]]
        else:
            raise ValueError("More than one PSD found in file %s. Specify which you want with --instrument." % opts.reference_psd)
        f_orig = psd.f0 + np.arange(len(psd.data.data)) * psd.deltaF
        psddata = psd.data.data

    # cut off upper frequency content as requested by user for better
    # computational performance
    f_max_orig = max(f_orig)
    if opts.fhigh_max:
        if opts.fhigh_max > f_max_orig:
            print >> sys.stderr, "Warning: requested fhigh-max (%.3f Hz) exceeds limits of PSD (%.3f Hz). Using PSD limit instead!" \
                    % (opts.fhigh_max, f_max_orig)
            opts.fhigh_max = float(f_max_orig)
    else:
        print >> sys.stderr, "Warning: fhigh-max not specified, using maximum frequency in the PSD (%.3f Hz)" \
                % f_max_orig
        opts.fhigh_max = float(f_max_orig)

    interpolator = UnivariateSpline(f_orig, np.log(psddata), s=0)

    # spline extrapolation may lead to unexpected results,
    # so set the PSD to infinity above the max original frequency
    noise_model = lambda g: np.where(g < f_max_orig, np.exp(interpolator(g)), np.inf)

else:
    noise_model = noise_models[opts.noise_model]

print "Interpolated noise model"

#
# initialize the bank
#
bank_pe = Bank(noise_model, opts.flow, False, opts.cache_waveforms, opts.neighborhood_size, opts.neighborhood_param, coarse_match_df=opts.coarse_match_df, iterative_match_df_max=opts.iterative_match_df_max, fhigh_max=opts.fhigh_max)

bank_nr = Bank(noise_model, opts.flow, False, opts.cache_waveforms, opts.neighborhood_size, opts.neighborhood_param, coarse_match_df=opts.coarse_match_df, iterative_match_df_max=opts.iterative_match_df_max, fhigh_max=opts.fhigh_max)
        

# prepare a new XML document
xmldoc = ligolw.Document()
xmldoc.appendChild(ligolw.LIGO_LW())
tbl = lsctables.New(lsctables.SnglInspiralTable)
xmldoc.childNodes[-1].appendChild(tbl)

# initialize random seed
np.random.mtrand.seed(opts.seed)

#
# prepare process table with information about the current program
#
opts_dict = dict((k, v) for k, v in opts.__dict__.iteritems() if v is not False and v is not None)
process = ligolw_process.register_to_xmldoc(xmldoc, "lalapps_cbc_sbank",
    opts_dict, version="no version",
    cvs_repository="sbank", cvs_entry_time=strftime('%Y/%m/%d %H:%M:%S'))

# Prepare the input points
#waveforms.waveforms[opts.approximant]
class InternalTemplate(waveforms.PrecessingSpinTemplate):
    approximant = opts.approximant
    def _get_dur(self):
       # Hackiness to the max!!!
       self.__class__ = waveforms.waveforms[opts.approximant]
       dur = waveforms.waveforms[opts.approximant]._get_dur(self)
       self.__class__ = InternalTemplate
       return dur

    def _get_f_final(self):
       # Hackiness to the max!!!
       self.__class__ = waveforms.waveforms[opts.approximant]
       f_final = waveforms.waveforms[opts.approximant]._get_f_final(self)
       self.__class__ = InternalTemplate
       return f_final

pe_points_doc = utils.load_filename(opts.pe_samples, contenthandler=ContentHandler, gz=opts.pe_samples.endswith('.gz'))
trial_sngls = lsctables.SnglInspiralTable.get_table(pe_points_doc)
# Assume all times are the same
epoch_time = trial_sngls[0].get_end()
for t in trial_sngls:
    curr_tmplt = InternalTemplate.from_sngl(t, bank=bank_pe)
    bank_pe.insort(curr_tmplt)
    

#
# populate params dictionary to be passed to the generators
#

params = {'mass1': (opts.mass1_min, opts.mass1_max),
          'mass2': (opts.mass2_min, opts.mass2_max),
          'mtotal': (opts.mtotal_min, opts.mtotal_max),
          'mratio': (opts.qmin, opts.qmax),
          'mchirp': (opts.mchirp_min, opts.mchirp_max)}

if opts.ns_bh_boundary_mass is not None:
    params['ns_bh_boundary_mass'] = opts.ns_bh_boundary_mass
    params['bh_spin'] = (opts.bh_spin_min, opts.bh_spin_max)
    params['ns_spin'] = (opts.ns_spin_min, opts.ns_spin_max)
else:
    params['spin1'] = (opts.spin1_min, opts.spin1_max)
    params['spin2'] = (opts.spin2_min, opts.spin2_max)

# get the correct generator for the chosen approximant
proposal = proposals[opts.approximant](opts.flow, InternalTemplate, bank_nr,
                                       **params)

print "Internal setup complete."

#
# main working loop
#

# Hardcoded
df = 0.5
PSD = get_PSD(df, bank_nr.flow, opts.fhigh_max, noise_model)
ifo_H1 = lalsim.DetectorPrefixToLALDetector('H1')
ifo_L1 = lalsim.DetectorPrefixToLALDetector('L1')

def generate_hplus_hcross(tmplt):
    hp, hc, cross = tmplt.get_whitened_normalized_comps(df, PSD=PSD)
    fp_H1, fc_H1 = lal.ComputeDetAMResponse\
        (ifo_H1.response, tmplt.theta, tmplt.phi, tmplt.psi,
         lal.GreenwichMeanSiderealTime(epoch_time))
    nfac = (fp_H1**2 + fc_H1**2 + 2. * fp_H1 * fc_H1 * cross)**0.5
    fp_H1 = fp_H1 / nfac
    fc_H1 = fc_H1 / nfac
    fp_L1, fc_L1 = lal.ComputeDetAMResponse\
        (ifo_L1.response, tmplt.theta, tmplt.phi, tmplt.psi,
         lal.GreenwichMeanSiderealTime(epoch_time))
    nfac = (fp_L1**2 + fc_L1**2 + 2. * fp_L1 * fc_L1 * cross)**0.5
    fp_L1 = fp_L1 / nfac
    fc_L1 = fc_L1 / nfac
    tmp_H1 = fp_H1 * hp.data.data[:] + fc_H1*hc.data.data[:]
    tmp_L1 = fp_L1 * hp.data.data[:] + fc_L1*hc.data.data[:]
    hp.data.data = tmp_H1
    hc.data.data = tmp_L1
    tmplt.h_H1 = hp
    tmplt.h_L1 = hc

# Generate and store PE waveforms

print "Generating all PE posterior waveforms."

for waveform in bank_pe:
    generate_hplus_hcross(waveform)

print "DONE"

print "Beginning main loop."
count = 0
for tmplt in proposal:
    if count > 100000:
        break
    count += 1
    if not count % 100:
        print "Processed %d trials." % count
        print "%d accepted points for NR" % len(bank_nr)
    generate_hplus_hcross(tmplt)
    
    # Check if proposal agrees really well with any of the pe points.
    for point in bank_pe:
        match_H1 = InspiralSBankComputeMatch(tmplt.h_H1, point.h_H1,
                                             bank_pe._workspace_cache[0])
        match_L1 = InspiralSBankComputeMatch(tmplt.h_L1, point.h_L1,
                                             bank_pe._workspace_cache[0])
        #print match_H1, match_L1
        if match_H1 > 1.0001 or match_L1 > 1.0001:
            raise ValueError()
        if match_H1 > 0.99 and match_L1 > 0.99:
            #print "Point %d matches PE posterior." % count
            break
    else:
        continue

    for point in bank_nr:
        # NOTE: *NOT* maximimizing over inclination. Assuming that orbital
        #       phase becomes a frequency-domain constant phase offset, this
        #       is not true for higher-order modes, and not strictly true for
        #       generic 2,2 mode signals
        point_hp, point_hc, point_cross = \
            point.get_whitened_normalized_comps(df, PSD=PSD)
        match_H1 = InspiralSBankComputeMatchMaxSkyLoc\
            (point_hp, point_hc, point_cross, tmplt.h_H1,
             bank_nr._workspace_cache[0], bank_nr._workspace_cache[1])
        match_L1 = InspiralSBankComputeMatchMaxSkyLoc\
            (point_hp, point_hc, point_cross, tmplt.h_H1,
             bank_nr._workspace_cache[0], bank_nr._workspace_cache[1])
        if match_H1 > 0.99 and match_L1 > 0.99:
            #print "Point %d is already covered by NR" % count
            break
    else:
        print "Point %d matches NR waveforms and is being added" % count
        tmplt._wf_hp = {}
        tmplt._wf_hc = {}
        bank_nr.insort(tmplt)

for tmplt in bank_nr:
    row = tmplt.to_sngl()
    row.event_id = ilwd.ilwdchar('sngl_inspiral:event_id:%d' %(len(tbl),))
    row.ifo = 'H1L1'
    row.process_id = process.process_id
    tbl.append(row)


# write out the document
if opts.output_filename.endswith(('.xml', '.xml.gz')):
    ligolw_process.set_process_end_time(process)
    utils.write_filename(xmldoc, opts.output_filename,
                         gz=opts.output_filename.endswith("gz"))
