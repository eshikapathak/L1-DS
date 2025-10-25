Python SEDS Training on LASA DatasetThis project implements the Stable Estimator of Dynamical Systems (SEDS) algorithm in Python, as described in the paper "Learning Stable Nonlinear Dynamical Systems With Gaussian Mixture Models" (Khansari & Billard, 2011).Setup1. DependenciesYou'll need the following Python libraries. You can install them using pip:pip install numpy scipy scikit-learn matplotlib pyLasaDataset
numpy: For numerical operations.scipy: For constrained optimization (scipy.optimize.minimize).scikit-learn: For the initial Gaussian Mixture Model (GMM) estimation.matplotlib: For plotting the results.pyLasaDataset: To load the handwriting data.2. File StructureEnsure your file structure looks like this:.
├── seds_train.py
├── seds_core.py
├── lasa_data.py
└── README.md
3. Run the TrainingOnce the dependencies are installed, you can run the main training script with optional command-line arguments.Default command:python seds_train.py
This will run with the defaults (Sshape, 6 Gaussians).Custom command:You can specify the shape, number of Gaussians (k), time step (dt), and simulation steps.python seds_train.py --shape "GShape" -k 8 --dt 0.005
To see all options, run:python seds_train.py --help
