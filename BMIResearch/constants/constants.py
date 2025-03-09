ORIGINAL_DATASET_PATH = 'drive/MyDrive/BmiResearch/data/datasets/original/a_walk_in_the_park'
PREPROCESSED_DATASET_PATH = 'drive/MyDrive/BmiResearch/data/datasets/preprocessed/a_walk_in_the_park'
RESULT_TESTS_PATH = 'drive/MyDrive/BmiResearch/data/results/a_walk_in_the_park'
BASE_DATASET_PATH = 'drive/MyDrive/BmiResearch/data/datasets/preprocessed/a_walk_in_the_park/baseline_chunked_data'
MODELS = 'drive/MyDrive/BmiResearch/models'

EEG_CHANNELS = ['F5', 'F3', 'Fz', 'F4', 'F6', 'FC3', 'FC1', 'FC2', 'FC4',
                'C5', 'C3', 'C1', 'Cz', 'C2', 'C4', 'C6',
                'CP5', 'CP3', 'CP1', 'CPz', 'CP2', 'CP4', 'CP6',
                'P5', 'P3', 'Pz', 'P4', 'P6',
                'PO3', 'PO4',
                'O1', 'Oz', 'O2']

FOUR_CLASSES_ENCODING = {'RightTO': 0, 'RightHS': 1, 'LeftTO': 2, 'LeftHS': 3}

# EOG_CHANNELS = ['Fp1', 'Fp2', 'F9', 'F10']

# PIPELINE
SIGNAL_PROCESSING = ['SLF', 'ASR', 'NOSP'] #  'Hinf', 'WT', 
FEATURE_EXTRACTION = ['ICA', 'CSP', 'NOFE'] # 'SFT', 'DWT', 
CLASSIFICATION = ['LDA', 'CNN', 'LSTM', 'SVM']

# PREPROCESSING TESTS
LOW_BAND = [0.5, 1., 1.5, 2., 2.5]
HIGH_BAND = [40.]
FREQUENCY = [500]
NORMALIZATION = [True]

# Common parameters
low_filter = 1
high_filter = 20
freq = 500
minutes_for_test = 2
window_size = 100
overlap = 80

# ASR 
asr_cutoff=20


