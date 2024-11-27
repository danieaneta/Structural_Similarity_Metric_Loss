import math


def SSIM_Loss_Func(mean_a, mean_b, var_a, var_b, covariance):
    k_1 = 0.01
    k_2 = 0.03
    L = 255
    constant_1 = (k_1 * L) ** 2
    constant_2 = (k_2 * L) ** 2
    constant_3 = (k_2 * L) ** 2 / 2
    l = ((2 * mean_a * mean_b) + constant_1) / ((mean_a ** 2) + (mean_b ** 2) + constant_1)
    c = ((2 * covariance) + constant_2) / (var_a + var_b + constant_2)
    s = (covariance + constant_3) / (math.sqrt(var_a) * math.sqrt(var_b) + constant_3)
    return l * c * s
    
if __name__ == "__main__":
    ssim = SSIM_Loss_Func(120, 115, 100, 90, 80)
    print(ssim)