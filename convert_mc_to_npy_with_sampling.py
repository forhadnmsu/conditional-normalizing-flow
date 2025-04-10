import ROOT
import numpy as np

# Load the histogram for rejection sampling
file_hist = ROOT.TFile("/Users/spin/RUS_Extended_MC/Exp_Data/hist_xf_mass.root", "READ")
hist_2D = file_hist.Get("hist_xf_mass")
hist_1D = file_hist.Get("hist_xf")
hist_2D.Scale(1.0/hist_2D.Integral())
hist_1D.Scale(1.0/hist_1D.Integral())
max_weight = hist_2D.GetMaximum()
max_weight_1D = hist_1D.GetMaximum()


print("max weight  for 2d:", max_weight)
print("max weight  for 1d:", max_weight_1D)

file_comb = ROOT.TFile("/Users/spin/RUS_Extended_MC/Comb/sampling/MC_RejectionSampled.root")
#file_comb = ROOT.TFile("/Users/spin/RUS_Extended_MC/Comb/allfiles/combine_mc_comb_march12.root")
#file_comb = ROOT.TFile("/Users/spin/RUS_Extended_MC/Comb/MC_Comb_Dump_Combined_Filtered_March8.root")
tree_comb = file_comb.Get("tree")

def save_to_npy_with_rejection(tree, filename):
    """Extracts muon and dimuon kinematics, applies rejection sampling, and saves filtered data."""
    data_list = []

    for event in tree:
        if len(event.rec_dimu_mass) == 0:
            continue

        mass_dimu = event.rec_dimu_mass[0]
        xf_dimu = event.rec_dimu_xf[0]

        # Apply selection cuts
        if not ( 5 < event.rec_dimu_pz[0] < 120):
            continue

        mass_dimu, pt_dimu, phi_dimu = 0, 0, 0
        px_dimu, py_dimu, pz_dimu = 0, 0, 0
        x1_dimu, x2_dimu, xf_dimu = 0, 0, 0

        # Extract event kinematics
        if len(event.rec_dimu_mass) > 0:
            px, py, pz, mass = event.rec_dimu_px[0], event.rec_dimu_py[0], event.rec_dimu_pz[0], event.rec_dimu_mass[0]
            dimu = ROOT.TLorentzVector()
            dimu.SetXYZM(px, py, pz, mass)
            mass_dimu, pt_dimu, phi_dimu = dimu.M(), dimu.Pt(), dimu.Phi()
            px_dimu, py_dimu, pz_dimu  = px, py, pz
            x1_dimu, x2_dimu, xf_dimu = event.rec_dimu_x1[0], event.rec_dimu_x2[0], event.rec_dimu_xf[0]


        if not (0 < pt_dimu < 5 and 5 < pz_dimu < 120 and mass_dimu < 8.0 and 0 < xf_dimu< 1.0 ):
            continue

        # Get the bin number for (xf, mass)
        bin_x = hist_2D.GetXaxis().FindBin(mass_dimu)
        bin_y = hist_2D.GetYaxis().FindBin(xf_dimu)
        weight = hist_2D.GetBinContent(bin_x, bin_y)


        bin_xf=hist_1D.FindBin(xf_dimu)
        weight_1D = hist_1D.GetBinContent(bin_xf)

        #weight_1D = hist_1D.GetBinContent(bin_x, bin_y)

        if weight <= 0:
            weight = 1e-6  # Assign a very small probability


        if weight_1D<=0:
            weight_1D=1e-6

        random_threshold = np.random.uniform(0, max_weight)
        random_threshold_1D = np.random.uniform(0, max_weight_1D)
        #if weight_1D < random_threshold_1D:
        #    continue  # Reject the event
    
        random_value = np.random.uniform(0, 1)

        #if random_value > (weight_1D / max_weight_1D):
        #    continue

        event_data = np.array([px_dimu, py_dimu, pz_dimu, x2_dimu, xf_dimu, mass_dimu, pt_dimu, phi_dimu], dtype=np.float32)
        if not np.isnan(event_data).any():
            data_list.append(event_data)
    # Save accepted events
    data_array = np.array(data_list, dtype=np.float32)
    np.save(filename, data_array)
save_to_npy_with_rejection(tree_comb, "mc_comb_data.npy")
