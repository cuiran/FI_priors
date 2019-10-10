
def funcpip_L1(flatpip_L1, priors):
    # assuming single causal SNP
    # compute functional PIP based on derived relations between prior ratios and pip ratios
    # inputs are both arrays of the same length
    i = next((i for i,x in enumerate(flatpip_L1) if x), None)
    a = flatpip_L1[i]
    p = priors[i]
    funcpip_unscaled = np.divide(flatpip_L1,a)*(np.divide(priors,p))
    return funcpip_unscaled/sum(funcpip_unscaled)
