NOMINAL_BW0 = 100 # kg

NOMINAL_BETA = 0.24 # -

NOMINAL_GAMMA_FM = 3.2 # EE rate of fat mass (kcal/kg/day)

NOMINAL_GAMMA_FFM = 22  # EE rate of fat-free mass (kcal/kg/day)

NOMINAL_ALPHA = 1 # Relationship between change in lean and fat mass

NOMINAl_NFM = 180 # energy cost of fat mass turnover (kcal/kg)

NOMINAL_NFFM = 230 # energy cost of fat-free mass turnover (kcal/kg)

NOMINAL_RHO_FM = 9300 # energy density of fat mass (kcal/kg)

NOMINAL_RHO_FFM = 1100 # energy cost of fat free mass (kcal/kg)

NOMINAL_RHO = (NOMINAl_NFM + NOMINAL_RHO_FM + NOMINAL_ALPHA*NOMINAL_NFFM + NOMINAL_ALPHA*NOMINAL_RHO_FFM) / (1 + NOMINAL_ALPHA - NOMINAL_BETA - NOMINAL_ALPHA*NOMINAL_BETA)

NOMINAL_K = (NOMINAL_GAMMA_FM + NOMINAL_ALPHA*NOMINAL_GAMMA_FFM)/ (1 + NOMINAL_ALPHA)

NOMINAL_EI0 = 2000 # kcal
NOMINAL_EE0 = 0 # kcal / kg
NOMINAL_UGE0 = 0

NOMINAL_PARAMS = [
                    NOMINAL_BW0,
                    NOMINAL_BETA,
                    NOMINAL_EI0,
                    NOMINAL_EE0,
                    NOMINAL_RHO,
                    NOMINAL_K
                ]



















