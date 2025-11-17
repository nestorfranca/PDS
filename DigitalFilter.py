# coding: utf-8

# Importando bibliotecas necessárias
import numpy as np
import matplotlib.pyplot as plt


class DigitalFilter:
    def __init__(self):
        # Paraâmetros gerais do filtro
        self.order = None
        self.epsilon = None
        self.omega_c = None

        # Parámetros do filtro protótipo
        self.num_prototype = None
        self.den_prototype = None
        self.gain_prototype = None

        # Parámetros do filtro analógico
        self.num_analog = None
        self.den_analog = None
        self.gain_analog = None

        # Parámetros do filtro digital
        self.num_digital = None
        self.den_digital = None
        self.gain_digital = None

    # ✅
    # ============================================================
    # Função para arredondar para cima com casas decimais específicas
    def ceil_decimal(self, valor, casas_decimais):
        """
        Implementation of the ceiling function with specific decimal places

        Parameters:
        ======================
        Inputs:
        ======================
            valor : float
                The value to be rounded up
            casas_decimais : int
                The number of decimal places to round up to

        ======================
        Returns:
        ======================
            rounded_value : float
                The rounded up value
        ======================
        """

        fator = 10**casas_decimais
        return np.ceil(valor * fator) / fator

    # ============================================================

    # ✅
    # ============================================================
    # Função para converter a função de transferência em zeros, polos e ganho
    def tf2zpk(self, num, den):
        """
        Implementation of the transfer function to zero-pole-gain conversion

        Parameters:
        ======================
        Inputs:
        ======================
            num : array
                The numerator coefficients of the filter polynomial
            den : array
                The denominator coefficients of the filter polynomial

        ======================
        Returns:
        ======================
            zeros : array
                The zeros of the filter
            poles : array
                The poles of the filter
            gain : float
                The gain of the filter
        ======================
        """

        zeros = np.roots(num)
        poles = np.roots(den)

        gain = num[0] / den[0]

        return zeros, poles, gain

    # ============================================================

    # ✅
    # ============================================================
    # Função para pré-warping de frequência
    def pre_warping(self, omega, Ts):
        """
        Implementation of the pre-warping frequency for bilinear transformation

        Parameters:
        ======================
        Inputs:
        ======================
            omega : float
                The analog frequency (rad/s)
            Ts : float
                The sampling period (s)

        ======================
        Returns:
        ======================
            warped_omega : float
                The pre-warped frequency (rad/s)
        """
        # return np.tan(omega * Ts / 2)
        return (2/Ts) * np.tan(omega * Ts / 2)

    # ============================================================

    # ✅
    # ============================================================
    # Parametros do filtro
    def filter_params(
        self, omega_p, omega_s, alpha_p, alpha_s, type="lowpass", Ts=None, warping=False
    ):
        """
        Implementation of the calculation of the prototype filter parameters

        Parameters:
        ======================
        Inputs:
        ======================
            omega_p : float
                Passband frequency (rad/s)
            omega_s : float
                Stopband frequency (rad/s)
            alpha_p : float
                Passband attenuation (dB)
            alpha_s : float
                Stopband attenuation (dB)
            type : {'bandpass', 'low', 'high', 'bandstop'}, optional
                Type of the filter:
                    - low-pass   : 'lowpass'
                    - high-passs : 'highpass'
                    - band-pass  : 'bandpass'
                    - band-stop  : 'bandstop'
            Ts : float, optional
                The sampling period (s)
            warping : bool, optional
                Whether to apply pre-warping of frequencies for bilinear transformation

        ======================
        Returns:
        ======================
            parameters : array
                The prototype filter parameters [Omega_p, Omega_s, alpha_p, alpha_s]
        ======================
        """

        parameters = []

        # Converte para array, caso seja passado um valor único
        omega_p = np.atleast_1d(omega_p)
        omega_s = np.atleast_1d(omega_s)

        # Pre-warping das frequências:
        if warping and Ts != None:
            Omega_p = self.pre_warping(omega_p, Ts)
            Omega_s = self.pre_warping(omega_s, Ts)
        else:
            Omega_p = omega_p
            Omega_s = omega_s

        # ==================================================
        # Configuração dos valores para o filtro protótipo:

        if Omega_p.shape != Omega_s.shape:
            raise ValueError(
                "As bandas de passagem e rejeição precisam ter a mesma quantidade de valores, 1 ou 2."
            )

        if Omega_p.shape[0] == 1:
            if type == "lowpass":
                Omega_prot_p = Omega_p[0]
                Omega_prot_s = Omega_s[0]

            elif type == "highpass":
                Omega_prot_p = 1
                Omega_prot_s = Omega_p[0] / Omega_s[0]

            else:
                raise ValueError("Não foi possível detectar o tipo de filtro")

        elif Omega_p.shape[0] == 2:
            prod_wp = Omega_p[0] * Omega_p[1]
            sub_wp = Omega_p[1] - Omega_p[0]

            if type == "bandpass":
                Omega_prot_p = 1
                Omega_prot_s = min(
                    ((prod_wp - Omega_s[0] ** 2) / (Omega_s[0] * sub_wp)),
                    ((Omega_s[1] ** 2 - prod_wp) / (Omega_s[1] * sub_wp)),
                )

            elif type == "bandstop":
                Omega_prot_p = 1
                Omega_prot_s = min(
                    ((Omega_s[0] * sub_wp) / (prod_wp - Omega_s[0] ** 2)),
                    ((Omega_s[1] * sub_wp) / (Omega_s[1] ** 2 - prod_wp)),
                )

            else:
                raise ValueError("Não foi possível detectar o tipo de filtro")

        else:
            raise ValueError(
                "É preciso que as bandas de passagem e rejeição possuam 1 ou 2 valores"
            )

        parameters = np.array([Omega_prot_p, Omega_prot_s, alpha_p, alpha_s])

        return parameters

    # ============================================================

    # ✅
    # ============================================================
    # Função para calcular a ordem e o parâmetro de ripple do Chebyshev
    def order_cheby(self, parameter):
        """
        Implementation of the calculation of the minimum order and ripple factor for Chebyshev filter

        Parameters:
        ======================
        Inputs:
        ======================
            parameter : array
                The prototype filter parameters [Omega_p, Omega_s, alpha_p, alpha_s]

        ======================
        Returns:
        ======================
            K : int
                Minimum order of the filter
            epsilon : float (only for Chebyshev)
                Ripple factor
        ======================
        """
        # Parametros do filtro:
        Omega_p = parameter[0]
        Omega_s = parameter[1]
        alpha_p = parameter[2]
        alpha_s = parameter[3]

        # Cálculo da Ordem Mínima do Filtro:
        K = np.ceil(
            np.arccosh(np.sqrt((10 ** (alpha_s / 10) - 1) / (10 ** (alpha_p / 10) - 1)))
            / (np.arccosh(Omega_s / Omega_p))
        )

        # Cálculo do parâmetro de controle de ondulação da banda:
        epsilon = np.sqrt(10 ** (alpha_p / 10) - 1)

        # Armazenando a ordem do filtro
        self.order = K
        self.epsilon = epsilon

        return K, epsilon

    # ===========================================================

    # ✅
    # ============================================================
    # Função para calcular a ordem e a frequência de corte do Butterworthnt
    def order_butter(self, parameter, oc_type="mean"):
        """
        Implementation of the calculation of the minimum order and cutoff frequency for Butterworth filter

        Parameters:
        ======================
        Inputs:
        ======================
            parameter : array
                The prototype filter parameters [Omega_p, Omega_s, alpha_p, alpha_s]
            oc_type : {'mean', 'min', 'max'}, optional
                Type of cutoff frequency calculation:
                    - mean : average value between min and max cutoff frequencies
                    - min  : minimum cutoff frequency
                    - max  : maximum cutoff frequency

        ======================
        Returns:
        ======================
            K : int
                Minimum order of the filter
            Omega_c : float (only for Butterworth)
                Cutoff frequency (rad/s)
        ======================
        """
        # Parametros do filtro:
        Omega_p = parameter[0]
        Omega_s = parameter[1]
        alpha_p = parameter[2]
        alpha_s = parameter[3]

        # Cálculo da Ordem Mínima do Filtro:
        K = np.log10((10 ** (alpha_s / 10) - 1) / (10 ** (alpha_p / 10) - 1)) / (
            2 * np.log10(Omega_s / Omega_p)
        )
        K = np.ceil(K)

        # Cálculo da Frequência de corte (média dos intervamos máximo e mínimo):
        Omega_c_min = Omega_p / (10 ** (alpha_p / 10) - 1) ** (1 / (2 * K))
        Omega_c_max = Omega_s / (10 ** (alpha_s / 10) - 1) ** (1 / (2 * K))

        # Critério de Passagem:
        if oc_type == "min":
            Omega_c = Omega_c_min
        # Critério de Rejeição:
        elif oc_type == "max":
            Omega_c = Omega_c_max
        # Critério do valor médio:
        elif oc_type == "mean":
            Omega_c = (Omega_c_max + Omega_c_min) / 2

        # Armazenando a ordem do filtro
        self.order = K
        self.omega_c = Omega_c

        return K, Omega_c

    # ============================================================

    # ✅
    # ============================================================
    # Implementação dos protótipos do butterworth
    def prototype_lp_butter(self, order):
        """
        Implementation of the low-pass filter prototype

        Parameters:
        ======================
        Inputs:
        ======================
            order : int
                The order of the filter (positive integer)

        ======================
        Returns:
        ======================
            numerador : array
                The numerator coefficients of the filter polynomial
            denominador : array
                The denominator coefficients of the filter polynomial
            gain_butter: float
                The gain of the filter protoype
        ======================
        """

        # Cálculo do Omega_C:
        Omega_c = 1  # 1 rad/s

        # Polos do Filtro:
        i = np.arange(1, order + 1)  # [1, 2, ..., K] :. k = order
        polos = 1j * Omega_c * np.exp(1j * np.pi * (2 * i - 1) / (2 * order))
        # polos = -Omega_c*np.sin(np.pi * (2*i - 1) / (2*K)) + 1j*Omega_c * np.cos(np.pi * (2*i - 1) / (2*K))

        # Gerando o polinômio do denominador:
        denominador = np.real(np.poly(polos))

        # Gerando o polinômio do numerador:
        """ 
            OBS: O numpy, quando detecta que o vetor só possui 1 valor, 
            ele converte o vetor para um valor numérico.
            O np.atleast_1d() serve para forçar o numpy a criar um vetor com apenas 1 valor 
        """
        numerador = np.atleast_1d(denominador[-1])

        gain_butter = 1

        # Parâmetros do filtro protótipo:
        self.num_prototype = numerador
        self.den_prototype = denominador
        self.gain_prototype = gain_butter

        return numerador, denominador, gain_butter

    # ============================================================

    # ✅
    # ============================================================
    # Implementação dos protótipos do chebyshev
    def prototype_lp_cheby(self, order, epsilon):
        """
        Implementation of the low-pass filter prototype

        Parameters:
        ======================
        Inputs:
        ======================
            order : int
                The order of the filter (positive integer)
            epsilon : float
                The ripple factor of the filter

        ======================
        Returns:
        ======================
            numerador : array
                The numerator coefficients of the filter polynomial
            denominador : array
                The denominator coefficients of the filter polynomial
            gain_cheby: float
                The gain of the filter protoype
        ======================
        """

        # Freq. do final da Banda de Passagem:
        Omega_p = 1  # 1 rad/s

        # Polos do Filtro:
        i = np.arange(1, order + 1)  # [1, 2, ..., K]      :. k = order

        # polos analogicos do chebyshev
        polos = -Omega_p * np.sin(np.pi * (2 * i - 1) / (2 * order)) * np.sinh(
            np.arcsinh(1 / epsilon) / order
        ) + 1j * Omega_p * np.cos(np.pi * (2 * i - 1) / (2 * order)) * np.cosh(
            np.arcsinh(1 / epsilon) / order
        )

        gain_cheby =  1 / np.sqrt(1 + epsilon**2) if order % 2 == 0 else 1

        # Gerando o polinômio do denominador analógico:
        denominador = np.real(np.poly(polos))

        # Gerando o polinômio do numerador:
        """ 
            OBS: O numpy, quando detecta que o vetor só possui 1 valor, 
            ele converte o vetor para um valor numérico.
            O np.atleast_1d() serve para forçar o numpy a criar um vetor com apenas 1 valor 
        """
        numerador = np.atleast_1d(gain_cheby * np.real(np.prod(-polos)))

        # Parâmetros do filtro protótipo:
        self.num_prototype = numerador
        self.den_prototype = denominador
        self.gain_prototype = gain_cheby

        return numerador, denominador, gain_cheby

    # ============================================================

    # ✅
    # ============================================================
    # Função para transformar o filtro protótipo em filtro passa-baixa com frequência de corte Omega_c
    def low_pass(self, order, Omega_c, epsilon=None, filter_type="butter"):
        """
        Implementation of the low-pass to low-pass frequency transformation

        Parameters:
        ======================
        Inputs:
        ======================
            order : int
                The order of the filter (positive integer)
            Omega_c : float (only for Butterworth)
                The cutoff frequency (rad/s)
            epsilon : float, optional (only for Chebyshev)
                The ripple factor of the filter (only for Chebyshev)
            filter_type : {'butter', 'cheby'}, optional
                Type of the prototype filter:
                    - Butterworth : 'butter'
                    - Chebyshev   : 'cheby'

        ======================
        Returns:
        ======================
            new_num : array
                The numerator coefficients of the transformed filter polynomial
            new_den : array
                The denominator coefficients of the transformed filter polynomial
            gain_n : float
                The gain of the transformed filter
        ======================
        """
        Omega_c = np.atleast_1d(Omega_c)
        # Obtém os coeficientes do filtro protótipo:
        if filter_type == "butter":
            num, den, gain_n = self.prototype_lp_butter(order)

        elif filter_type == "cheby" and epsilon != None:
            num, den, gain_n = self.prototype_lp_cheby(order, epsilon)

        # Ordem dos polinômios:
        N = len(den) - 1  # denominador
        M = len(num) - 1  # numerador

        # novo denominador:
        new_den = np.array([den[i] * (1 / Omega_c[0]) ** (N - i) for i in range(N + 1)])

        # novo numerador:
        new_num = np.array([num[i] * (1 / Omega_c[0]) ** (M - i) for i in range(M + 1)])

        # ajusta o ganhos dos coeficientes:
        new_num = new_num / new_den[0]
        new_den = new_den / new_den[0]

        # Parâmetros do filtro passa-baixa:
        self.num_analog = new_num
        self.den_analog = new_den
        self.gain_analog = gain_n

        return new_num, new_den, gain_n

    # ============================================================

    # ============================================================
    # Função para transformar o filtro protótipo em filtro passa-alta com frequência de corte
    def high_pass(self, order, Omega_p, Omega_c=None, epsilon=None, filter_type="butter"):
        """
        Implementation of the low-pass to high-pass frequency transformation

        Parameters:
        ======================
        Inputs:
        ======================
            order : int
                The order of the filter (positive integer)
            Omega_c : float
                The cutoff frequency (rad/s)
            epsilon : float, optional (only for Chebyshev)
                The ripple factor of the filter (only for Chebyshev)
            filter_type : {'butter', 'cheby'}, optional
                Type of the prototype filter:
                    - Butterworth : 'butter'
                    - Chebyshev   : 'cheby'

        ======================
        Returns:
        ======================
            new_num : array
                The numerator coefficients of the transformed filter polynomial
            new_den : array
                The denominator coefficients of the transformed filter polynomial
        ======================
        """

        # Obtém os coeficientes do filtro protótipo:
        if filter_type == 'butter' and Omega_c != None:
            att_num, att_den, gain_n = self.low_pass(
                order, Omega_c, filter_type=filter_type
            )

        elif filter_type == "cheby" and epsilon != None:
            att_num, att_den, gain_n = self.prototype_lp_cheby(order, epsilon)

        # Extrai os polos e zeros:
        polos = np.roots(np.array(att_den))
        zeros = np.roots(att_num)

        # Ordem do denominador e numerador:
        N = len(polos)
        M = len(zeros)

        # novos polos:
        new_polos = Omega_p / polos

        # novos zeros:
        new_zeros = np.zeros(int(N))

        # novo ganho:
        new_ganho = gain_n

        # novo numerador:
        new_num = new_ganho * np.real(np.poly(new_zeros))
        # novo denominador:
        new_den = np.poly(new_polos)

        # ajusta o ganhos dos coeficientes:
        new_den = new_den / new_den[0]
        new_num = gain_n * new_num / new_den[0]

        # Parâmetros do filtro passa-alta:
        self.num_analog = new_num
        self.den_analog = new_den
        self.gain_analog = new_ganho

        return new_num, new_den

    # ============================================================

    # ============================================================
    # Função para transformar o filtro protótipo em filtro passa-banda com frequências de corte omega_1 e omega_2
    def band_pass(self, order, Omega_p, Omega_c=None, epsilon=None, filter_type="butter"):
        """
        Implementation of the low-pass to band-pass frequency transformation

        Parameters:
        ======================
        Inputs:
        ======================
            order : int
                The order of the filter (positive integer)
            Omega_c : array
                The cutoff frequencies [omega_1, omega_2] (rad/s)
            epsilon : float, optional (only for Chebyshev)
                The ripple factor of the filter (only for Chebyshev)
            filter_type : {'butter', 'cheby'}, optional
                Type of the prototype filter:
                    - Butterworth : 'butter'
                    - Chebyshev   : 'cheby'

        ======================
        Returns:
        ======================
            new_num : array
                The numerator coefficients of the transformed filter polynomial
            new_den : array
                The denominator coefficients of the transformed filter polynomial
        ======================
        """

        # Obtém os coeficientes do filtro protótipo:
        if filter_type == "butter" and Omega_c != None:
            # num, den, gain_n = self.prototype_lp_butter(order)
            att_num, att_den, gain_n = self.low_pass(
                order, Omega_c, filter_type=filter_type
            )

        elif filter_type == "cheby" and epsilon != None:
            att_num, att_den, gain_n = self.prototype_lp_cheby(order, epsilon)

        # Extrai os polos e zeros:
        polos = np.roots(att_den)
        zeros = np.roots(att_num)

        # Ordem do denominador e numerador:
        N = len(polos)
        M = len(zeros)

        # novos polos:
        P1 = polos * (Omega_p[1] - Omega_p[0]) / 2 + np.sqrt(
            (polos * (Omega_p[1] - Omega_p[0]) / 2) ** 2 - Omega_p[0] * Omega_p[1]
        )
        P2 = polos * (Omega_p[1] - Omega_p[0]) / 2 - np.sqrt(
            (polos * (Omega_p[1] - Omega_p[0]) / 2) ** 2 - Omega_p[0] * Omega_p[1]
        )

        new_polos = np.concatenate((P1, P2))

        # novos zeros:
        new_zeros = np.zeros(int(N))

        # novo ganho:
        new_ganho = gain_n

        # novo denominador:
        new_den = np.real(np.poly(new_polos))

        # novo numerador:
        new_num = (
            new_ganho
            * np.real(np.prod(-polos))
            * ((Omega_p[1] - Omega_p[0]) ** N)
            * np.real(np.poly(new_zeros))
        )

        # ajusta o ganhos dos coeficientes:
        new_den = new_den / new_den[0]
        new_num = new_num / new_den[0]

        # Parâmetros do filtro passa-banda:
        self.num_analog = new_num
        self.den_analog = new_den
        self.gain_analog = new_ganho

        return new_num, new_den

    # ============================================================

    # ============================================================
    # Função para transformar o filtro protótipo em filtro rejeita-banda com frequências de corte omega_1 e omega_2
    def band_stop(self, order, Omega_p, Omega_c=None, epsilon=None, filter_type="butter"):
        """
        Implementation of the low-pass to band-stop frequency transformation

        Parameters:
        ======================
        Inputs:
        ======================
            order : int
                The order of the filter (positive integer)
            Omega_c : array
                The cutoff frequencies [omega_1, omega_2] (rad/s)
            epsilon : float, optional (only for Chebyshev)
                The ripple factor of the filter (only for Chebyshev)
            filter_type : {'butter', 'cheby'}, optional
                Type of the prototype filter:
                    - Butterworth : 'butter'
                    - Chebyshev   : 'cheby'

        ======================
        Returns:
        ======================
            new_num : array
                The numerator coefficients of the transformed filter polynomial
            new_den : array
                The denominator coefficients of the transformed filter polynomial
        ======================
        """

        # Obtém os coeficientes do filtro protótipo:
        if filter_type == 'butter' and Omega_c != None:
            # num, den, gain_n = self.prototype_lp_butter(order)
            att_num, att_den, gain_n = self.low_pass(
                order, Omega_c, filter_type=filter_type
            )

        elif filter_type == "cheby" and epsilon != None:
            att_num, att_den, gain_n = self.prototype_lp_cheby(order, epsilon)

        # Extrai os polos e zeros:
        polos = np.roots(att_den)
        zeros = np.roots(att_num)

        # Ordem do denominador e numerador:
        N = len(polos)
        M = len(zeros)

        # novos polos:
        P1 = (Omega_p[1] - Omega_p[0]) / (2 * polos) + np.sqrt(
            ((Omega_p[1] - Omega_p[0]) / (2 * polos)) ** 2 - Omega_p[0] * Omega_p[1]
        )
        P2 = (Omega_p[1] - Omega_p[0]) / (2 * polos) - np.sqrt(
            ((Omega_p[1] - Omega_p[0]) / (2 * polos)) ** 2 - Omega_p[0] * Omega_p[1]
        )

        new_polos = np.concatenate((P1, P2))

        # novos zeros ( Gerando os N zeros em +- sqrt(Omega_p1*Omega_p2) ):
        Z1 = +1j * np.sqrt(Omega_p[0] * Omega_p[1]) * np.ones(N)
        Z2 = -1j * np.sqrt(Omega_p[0] * Omega_p[1]) * np.ones(N)

        new_zeros = np.concatenate((Z1, Z2))

        # novo ganho:
        new_ganho = gain_n

        # novo denominador:
        new_den = np.real(np.poly(new_polos))

        # novo numerador:
        new_num = new_ganho * np.real(np.poly(new_zeros))

        # ajusta o ganhos dos coeficientes:
        new_den = new_den / new_den[0]
        new_num = new_num / new_den[0]

        # Parâmetros do filtro rejeita-banda:
        self.num_analog = new_num
        self.den_analog = new_den
        self.gain_analog = new_ganho

        return new_num, new_den

    # ============================================================

    # ============================================================
    #
    def bilinear_transform(self, num, den, Ts):
        """
        Implementation of the bilinear transformation for analog to digital filter conversion

        Parameters:
        ======================
        Inputs:
        ======================
            num : array
                The numerator coefficients of the analog filter polynomial
            den : array
                The denominator coefficients of the analog filter polynomial
            Ts : float
                The sampling period (s)
        ======================
        Returns:
        ======================
            new_num : array
                The numerator coefficients of the digital filter polynomial
            new_den : array
                The denominator coefficients of the digital filter polynomial
        ======================

        """
        # Extraindo ganho K:
        K = num[0] / den[0]

        # Extraindo os polos e zeros:
        polos = np.roots(den)
        zeros = np.roots(num)

        # Calculando a Ordem dos polinômios:
        N = len(polos)
        M = len(zeros)

        # ==================================================
        # Calculando novo ganho:
        Kn = K * (np.prod((2/Ts) - zeros) / np.prod((2/Ts) - polos))

        # remapeando polos e zeros, no tempo discreto:
        polos_z = (1 + (polos * (Ts/2))) / (1 - (polos * (Ts/2)))
        zeros_z = (1 + (zeros * (Ts/2))) / (1 - (zeros * (Ts/2)))

        # Adicionando zeros:
        for _ in range(N - M):
            zeros_z = np.append(zeros_z, -1)

        # Gerando numerador e denominador do filtro digital:
        den_z = np.real(np.poly(polos_z))
        num_z = Kn * np.real(np.poly(zeros_z))

        # Parâmetros do filtro digital:
        self.num_digital = num_z
        self.den_digital = den_z
        self.gain_digital = Kn

        return num_z, den_z

    # ============================================================

    # ============================================================
    #
    def butterworth_filter(
        self,
        omega_p,
        omega_s,
        alpha_p,
        alpha_s,
        type_response="lowpass",
        oc_type="mean",
        Ts=None,
        warping=False,
    ):
        """
        Implementation of the Butterworth filter

        Parameters:
        ======================
        Inputs:
        ======================
            omega_p : float
                The passband frequency of the filter (Hz)
            omega_s : float
                The stopband frequency of the filter (Hz)
            alpha_p : float
                The passband attenuation of the filter (dB)
            alpha_s : float
                The stopband attenuation of the filter (dB)
            type_response : {'bandpass', 'low', 'high', 'bandstop'}, optional
                Type of the filter:
                    - low-pass   : 'lowpass'
                    - high-passs : 'highpass'
                    - band-pass  : 'bandpass'
                    - band-stop  : 'bandstop'
            oc_type : {'mean', 'min', 'max'}, optional
                Type of cutoff frequency calculation:
                    - mean : average value between min and max cutoff frequencies
                    - min  : minimum cutoff frequency
                    - max  : maximum cutoff frequency
            Ts : float, optional
                The sampling period (s)
            warping : bool, optional
                Whether to apply pre-warping of frequencies for bilinear transformation
        ======================
        Returns:
        ======================
            num : array
                The numerator coefficients of the digital filter polynomial
            den : array
                The denominator coefficients of the digital filter polynomial
        ======================
        """

        filter_type = "butter"
        parametros_prot = self.filter_params(
            omega_p,
            omega_s,
            alpha_p,
            alpha_s,
            type=type_response,
            Ts=Ts,
            warping=warping,
        )

        K, Omega_c = self.order_butter(parameter=parametros_prot, oc_type=oc_type)

        if type_response == "lowpass":
            num, den, _ = self.low_pass(K, Omega_c, filter_type=filter_type)

        elif type_response == "highpass":
            num, den = self.high_pass(K, omega_p, Omega_c, filter_type=filter_type)

        elif type_response == "bandpass":
            num, den = self.band_pass(K, omega_p, Omega_c, filter_type=filter_type)

        elif type_response == "bandstop":
            num, den = self.band_stop(K, omega_p, Omega_c, filter_type=filter_type)

        if Ts == None:
            return num, den
        else:
            return self.bilinear_transform(num, den, Ts)

    # ============================================================

    # ============================================================
    #
    def chebyshev_filter(
        self,
        omega_p,
        omega_s,
        alpha_p,
        alpha_s,
        type_response="lowpass",
        Ts=None,
        warping=False,
    ):
        """
        Implementation of the Chebyshev filter

        Parameters:
        ======================
        Inputs:
        =====================
            omega_p : float
                The passband frequency of the filter (Hz)
            omega_s : float
                The stopband frequency of the filter (Hz)
            alpha_p : float
                The passband attenuation of the filter (dB)
            alpha_s : float
                The stopband attenuation of the filter (dB)
            type_response : {'bandpass', 'low', 'high', 'bandstop'}, optional
                Type of the filter:
                    - low-pass   : 'lowpass'
                    - high-passs : 'highpass'
                    - band-pass  : 'bandpass'
                    - band-stop  : 'bandstop'
            oc_type : {'mean', 'min', 'max'}, optional
                Type of cutoff frequency calculation:
                    - mean : average value between min and max cutoff frequencies
                    - min  : minimum cutoff frequency
                    - max  : maximum cutoff frequency
            Ts : float, optional
                The sampling period (s)
        ======================
        Returns:
        ======================
            num : array
                The numerator coefficients of the digital filter polynomial
            den : array
                The denominator coefficients of the digital filter polynomial
        ======================
        """

        filter_type = "cheby"

        parametros_prot = self.filter_params(
            omega_p,
            omega_s,
            alpha_p,
            alpha_s,
            type=type_response,
            Ts=Ts,
            warping=warping,
        )

        K, epsilon = self.order_cheby(parameter=parametros_prot)

        if type_response == "lowpass":
            num, den, _ = self.low_pass(
                K, omega_p, epsilon=epsilon, filter_type=filter_type
            )

        elif type_response == "highpass":
            num, den = self.high_pass(
                K, omega_p, epsilon=epsilon, filter_type=filter_type
            )

        elif type_response == "bandpass":
            num, den = self.band_pass(
                K, omega_p, epsilon=epsilon, filter_type=filter_type
            )

        elif type_response == "bandstop":
            num, den = self.band_stop(
                K, omega_p, epsilon=epsilon, filter_type=filter_type
            )

        if Ts != None:
            return self.bilinear_transform(num, den, Ts)
        else:
            return num, den

    # ============================================================

    # ============================================================
    # Função para plotar a resposta em frequência do filtro digital projetado
    def plot_response(self, num, den, x_min, x_max, Ts=None, params=[]):
        """
        Implementation of the frequency response plot of the designed digital filter

        Parameters:
        ======================
        Inputs:
        ======================
            num : array
                The numerator coefficients of the digital filter polynomial
            den : array
                The denominator coefficients of the digital filter polynomial
            x_min : float
                The minimum x-axis value for the plot
            x_max : float
                The maximum x-axis value for the plot
            Ts : float
                The sampling period (s)
            params : array
                The parameter of the filter [Omega_p, Omega_s, alpha_p, alpha_s]

        ======================
        Returns:
        ======================
            None
        ======================
        """

        if params:
            print(params)

        # Frequências para análise:
        if Ts == None:
            w = np.arange(x_min, x_max, 0.01)
            H = np.polyval(num, (1j*w))/np.polyval(den, (1j*w))
        
        else:
            w = np.arange(x_min*Ts, x_max*Ts, Ts)
            H = np.polyval(num, np.exp(1j*w))/np.polyval(den, np.exp(1j*w))

        # Plotando a resposta em frequência:
        plt.figure()
        plt.plot(w, 20 * np.log10(np.abs(H)), "b")
        plt.title("Resposta em Frequência do Filtro Digital")
        plt.xlabel("Frequência [Rad/amostra]")
        plt.ylabel("Magnitude [dB]")
        plt.grid()
        plt.tight_layout()
        plt.show()

    # ============================================================
