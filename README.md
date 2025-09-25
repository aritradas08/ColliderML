# ColliderML

The function convert_calo_hits converts the Calorimeter information from ROOT format to .h5 format. It takes in the root file, the desired h5 file, the number of events and the required calo region as input arguments. The default value of calo_type is "All".
An example function call is:
convert_calo_hits("edm4hep.root", "hcal_hits.h5", n_events=5, calo_type="HCal")

The function convert_mcparticles converts the MCParticles information from ROOT format to .h5 format. It takes in the root file, the desired h5 file and the number of events as input arguments.
An example function call is:
convert_mcparticles("edm4hep.root","MCParticles_v1",n_events = 5)

The function root_to_h5_tracker converts the Tracker information from ROOT format to .h5 format. It takes in the root file, the desired h5 file, the number of events and the detector region as input arguments.
An example function call and how to specify the detector region is shown as follows:
detectors = {
    "PixelBarrelReadout": 1,
    "PixelEndcapReadout": 2,
    "LongStripBarrelReadout": 3,
    "ShortStripBarrelReadout": 4
}

root_to_h5_tracker("edm4hep.root", "tracker_hits.h5", num_events=5, selected_detectors=detectors)

The function inspect_h5_file is used to inspect the created h5 file to cross check its contents. It takes in the name of h5 file, the number of events we want to check, and the number of entries for each event we want to check.
An example function call is:
inspect_h5_file("tracker_hits.h5",n_events=3,n_hits=3)
