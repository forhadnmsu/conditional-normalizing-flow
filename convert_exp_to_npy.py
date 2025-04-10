import ROOT
import numpy as np

#file_exp = ROOT.TFile("Filtered_Data_Exp_March9.root")
file_exp = ROOT.TFile("../Data_GoodRuns_March05.root", "READ")
tree_exp = file_exp.Get("tree")

def save_to_npy_with_selections(tree, filename):
    """Extracts and filters muon kinematics before saving as a NumPy array."""
    data_list = []

    # Define a 2D histogram for pz vs mass_dimu
    hist_2D = ROOT.TH2D("hist_xf_mass", "2D Histogram of xf vs Mass",
                    50, 0.0, 8.,  # 50 bins for pz in range [5, 120] GeV
                    50, 0, 1)    # 50 bins for mass_dimu in range [0, 5] GeV

    hist_1D = ROOT.TH1D("hist_xf", "1D Histogram of xf ",50, 0.0, 1.0)

    for event in tree:
        # Initialize muon variables
        px_mup, py_mup, pz_mup = 0, 0, 0
        px_mum, py_mum, pz_mum = 0, 0, 0
        mass_dimu, pt_dimu, phi_dimu = 0, 0, 0
        px_dimu, py_dimu, pz_dimu = 0, 0, 0
        x1_dimu, x2_dimu, xf_dimu = 0, 0, 0

        # Assign first found mu+ and mu-
        #for i in range(len(event.mu_plus_vtx_px)):
        #    px1, py1, pz1 = event.mu_plus_vtx_px[i], event.mu_plus_vtx_py[i], event.mu_plus_vtx_pz[i]
        #    px2, py2, pz2 = event.mu_minus_vtx_px[i], event.mu_minus_vtx_py[i], event.mu_minus_vtx_pz[i]
        #    px_mup, py_mup, pz_mup, px_mum, py_mum, pz_mum = px1, py1, pz1, px2, py2, pz2

        if len(event.rec_dimu_mass) > 0:
            px, py, pz, mass = event.rec_dimu_px[0], event.rec_dimu_py[0], event.rec_dimu_pz[0], event.rec_dimu_mass[0]
            dimu = ROOT.TLorentzVector()
            dimu.SetXYZM(px, py, pz, mass)
            mass_dimu, pt_dimu, phi_dimu = dimu.M(), dimu.Pt(), dimu.Phi()
            px_dimu, py_dimu, pz_dimu  = px, py, pz
            x1_dimu, x2_dimu, xf_dimu = event.rec_dimu_x1[0], event.rec_dimu_x2[0], event.rec_dimu_xf[0]

        if not (-10 < px_dimu < 10 and
                -10 < py_dimu < 10 and
                0 < mass_dimu < 8 and
                0 < xf_dimu < 1 and
                0 < pz_dimu < 120):
                continue

#        # Apply selection cuts
#        if not (0 < pt_dimu < 5 and
#                5 < pz_dimu < 120 and
#                -10 < px_mup < 10 and -10 < py_mup < 10 and 0 < pz_mup < 120 and
#                -10 < px_mum < 10 and -10 < py_mum < 10 and 0 < pz_mum < 120):
#            continue  # Skip event if any cut fails

        # Remove specific dimuon events
        #if  (67 < event.rec_dimu_pz[0] < 74 and 2.9 < event.rec_dimu_mass[0] < 3.3):
        #    continue  # Skip event

        # Convert event data to NumPy array
        event_data = np.array([px_dimu, py_dimu, pz_dimu, x2_dimu, xf_dimu, mass_dimu, pt_dimu, phi_dimu], dtype=np.float32)
        
        # Ensure no NaN values
        if not np.isnan(event_data).any():
            data_list.append(event_data)
            hist_1D.Fill(xf_dimu)
            hist_2D.Fill(mass_dimu, xf_dimu)

    # Convert to NumPy array
    data_array = np.array(data_list, dtype=np.float32)
    output_file = ROOT.TFile("hist_xf_mass.root", "RECREATE")
    hist_2D.Write()
    hist_1D.Write()
    output_file.Close()

    # Print min/max values for each feature
    feature_names = [ "px_dimu", "py_dimu", "pz_dimu", "x2_dimu", "xf_dimu", "mass_dimu", "pt_dimu", "phi_dimu"]
    for i, feature in enumerate(feature_names):
        print(f"{feature}: min={data_array[:, i].min()}, max={data_array[:, i].max()}")

    # Save as npy file
    np.save(filename, data_array)
    print(f"Saved {filename} with {len(data_list)} events.")

# Apply function with selection cuts
save_to_npy_with_selections(tree_exp, "exp_data_filtered.npy")

