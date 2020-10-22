USER = 'christian'

if USER == 'christian':
    DO_NOT_COPY = []
    # dataset
    NAT_IM_BASE = 'data/natural_images'
    IMFX_IM_BASE = 'data/immunfix_images'
    IMFX_TEST_BASE = 'data/immunfix_testsets'

    EXP_FOLDER = './experiments'

    PRINT_FREQUENCY = 100

    # default training values
    LR = 0.001
    WD = 0.00001
    BS = 16
    MOM = 0.9
    CS = 224
    MT = 'resnet18'
    OPT = 'Adam'
    
    # default options
    in_dim = 3
    out_dim = 7
    dataset = "imfx_im"

DO_NOT_COPY += [EXP_FOLDER]
