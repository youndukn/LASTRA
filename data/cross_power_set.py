from data.core import Core
import numpy as np

class CrossPowerSet():

    def __init__(self,summary, abs, nfis, ss1, ss2, power, flux, densities):
        self.abs = abs
        self.nfis = nfis
        self.ss1 = ss1
        self.ss2 = ss2
        self.power = power
        self.flux = flux
        self.densities = densities
        self.summary = summary

        if len(self.abs.get_value_matrix(0)) == 0:
            raise IOError

        if len(self.nfis.get_value_matrix(0)) == 0:
            raise IOError

        if len(self.power.get_value_matrix(0)) == 0:
            raise IOError

        if len(self.flux.get_value_matrix(0)) == 0:
            raise IOError

        input_matrix = []
        output_matrix = []
        flux_matrix = []

        input_matrix.append(self.abs.get_value_matrix(0))
        input_matrix.append(self.abs.get_value_matrix(1))
        input_matrix.append(self.nfis.get_value_matrix(0))
        input_matrix.append(self.nfis.get_value_matrix(1))
        #input_matrix.append(self.ss1.get_value_matrix(0))
        #input_matrix.append(self.ss1.get_value_matrix(1))
        input_matrix.append(self.ss2.get_value_matrix(0))
        #input_matrix.append(self.ss2.get_value_matrix(1))

        output_matrix.append(self.power.get_value_matrix(0))
        output_matrix.append(self.power.get_value_matrix(1))
        output_matrix.append(self.power.get_value_matrix(2))
        output_matrix.append(self.power.get_value_matrix(3))

        flux_matrix.append(self.flux.get_value_matrix(0))
        flux_matrix.append(self.flux.get_value_matrix(1))

        o_array = []
        o_array.append(self.abs.g1)
        o_array.append(self.abs.g2)
        o_array.append(self.nfis.g1)
        o_array.append(self.nfis.g2)
        #o_array.append(self.ss1.g1)
        #o_array.append(self.ss1.g2)
        o_array.append(self.ss2.g1)
        #o_array.append(self.ss2.g2)

        flux_o_array = []
        flux_o_array.append(self.flux.g1)
        flux_o_array.append(self.flux.g2)

        input_full_matrix = []

        #fullcore expansion
        for i in range(5):

            o_matrix = input_matrix[i]

            second = np.rot90(o_matrix)
            first = np.rot90(second)
            third = np.delete(np.rot90(first), 0, 0)
            fourth = np.delete(o_matrix, 0, 0)

            second = np.array(second, dtype=float)
            first = np.array(first, dtype=float)
            third = np.array(third, dtype=float)
            fourth = np.array(fourth, dtype=float)

            half1 = np.concatenate((first, third))
            half2 = np.concatenate((second, fourth))

            full_matrix = np.concatenate((half1, half2), axis=1)
            input_full_matrix.append(full_matrix)

        self.input_tensor = np.zeros([10, 10, 5], dtype=np.float)
        self.input_tensor_full = np.zeros([19, 19, 5], dtype=np.float)

        for i in range(5):
            for j in range(10):
                for k in range(10):
                    self.input_tensor[j][k][i] = float(input_matrix[i][j][k])*float(o_array[i])

        for i in range(5):
            for j in range(19):
                for k in range(19):
                    self.input_tensor_full[j][k][i] = float(input_full_matrix[i][j][k])*float(o_array[i])

        self.output_tensor = np.zeros([100], dtype=np.float)

        for j in range(10):
            for k in range(10):
                index = j*10+k
                self.output_tensor[index] = output_matrix[0][j][k]

        self.flux_tensor = np.zeros([2, 100], dtype=np.float)

        for i in range(2):
            for j in range(10):
                for k in range(10):
                    index = j * 10 + k
                    self.flux_tensor[i, index] = float(flux_matrix[i][j][k])*float(flux_o_array[i])

        self.density_tensor_full = np.zeros([19, 19, 20], dtype=np.float)

        for i in range(20):
            density_temp = self.densities[i].get_value_matrix(0)

            o_matrix = density_temp

            second = np.rot90(o_matrix)
            first = np.rot90(second)
            third = np.delete(np.rot90(first), 0, 0)
            fourth = np.delete(o_matrix, 0, 0)

            second = np.array(second, dtype=float)
            first = np.array(first, dtype=float)
            third = np.array(third, dtype=float)
            fourth = np.array(fourth, dtype=float)

            half1 = np.concatenate((first, third))
            half2 = np.concatenate((second, fourth))

            full_matrix = np.concatenate((half1, half2), axis=1)

            for j in range(19):
                for k in range(19):
                    self.density_tensor_full[j][k][i] = float(full_matrix[j][k]) * float(self.densities[i].o1)

        self.summary_tensor = np.array(self.summary, dtype=float)


class CrossPowerSetI():

    def __init__(self,
                 summary,
                 abs,
                 nfis,
                 ss1,
                 ss2,
                 power,
                 flux,
                 densities, iabs, infis, iss1):
        self.abs = abs
        self.nfis = nfis
        self.ss1 = ss1
        self.ss2 = ss2
        self.power = power
        self.flux = flux
        self.densities = densities
        self.summary = summary
        self.iabs = iabs
        self.infis = infis
        self.iss1 = iss1

        if len(self.abs.get_value_matrix(0)) == 0:
            raise IOError

        if len(self.nfis.get_value_matrix(0)) == 0:
            raise IOError

        if len(self.power.get_value_matrix(0)) == 0:
            raise IOError

        if len(self.flux.get_value_matrix(0)) == 0:
            raise IOError

        input_matrix = []
        iinput_matrix = []
        output_matrix = []
        flux_matrix = []

        input_matrix.append(self.abs.get_value_matrix(0))
        input_matrix.append(self.abs.get_value_matrix(1))
        input_matrix.append(self.nfis.get_value_matrix(0))
        input_matrix.append(self.nfis.get_value_matrix(1))
        input_matrix.append(self.ss2.get_value_matrix(0))

        iinput_matrix.append(self.iabs.get_value_matrix(0))
        iinput_matrix.append(self.iabs.get_value_matrix(1))
        iinput_matrix.append(self.infis.get_value_matrix(0))
        iinput_matrix.append(self.infis.get_value_matrix(1))
        iinput_matrix.append(self.iss1.get_value_matrix(0))

        output_matrix.append(self.power.get_value_matrix(0))
        output_matrix.append(self.power.get_value_matrix(1))
        output_matrix.append(self.power.get_value_matrix(2))
        output_matrix.append(self.power.get_value_matrix(3))

        flux_matrix.append(self.flux.get_value_matrix(0))
        flux_matrix.append(self.flux.get_value_matrix(1))

        o_array = []
        o_array.append(self.abs.g1)
        o_array.append(self.abs.g2)
        o_array.append(self.nfis.g1)
        o_array.append(self.nfis.g2)
        o_array.append(self.ss2.g1)

        io_array = []
        io_array.append(self.iabs.g1)
        io_array.append(self.iabs.g2)
        io_array.append(self.infis.g1)
        io_array.append(self.infis.g2)
        io_array.append(self.iss1.g1)

        flux_o_array = []
        flux_o_array.append(self.flux.g1)
        flux_o_array.append(self.flux.g2)

        input_full_matrix = []

        #fullcore expansion
        for i in range(5):

            o_matrix = input_matrix[i]

            second = np.rot90(o_matrix)
            first = np.rot90(second)
            third = np.delete(np.rot90(first), 0, 0)
            fourth = np.delete(o_matrix, 0, 0)

            second = np.array(second, dtype=float)
            first = np.array(first, dtype=float)
            third = np.array(third, dtype=float)
            fourth = np.array(fourth, dtype=float)

            half1 = np.concatenate((first, third))
            half2 = np.concatenate((second, fourth))

            full_matrix = np.concatenate((half1, half2), axis=1)
            full_matrix = np.delete(full_matrix, [9,], axis=1)

            input_full_matrix.append(full_matrix)

        self.input_tensor = np.zeros([10, 10, 5], dtype=np.float)
        self.input_tensor_full = np.zeros([19, 19, 5], dtype=np.float)

        for i in range(5):
            for j in range(10):
                for k in range(10):
                    self.input_tensor[j][k][i] = float(input_matrix[i][j][k])*float(o_array[i])

        for i in range(5):
            for j in range(19):
                for k in range(19):
                    self.input_tensor_full[j][k][i] = float(input_full_matrix[i][j][k])*float(o_array[i])

        iinput_full_matrix = []
        # fullcore expansion
        for i in range(5):
            o_matrix = iinput_matrix[i]

            second = np.rot90(o_matrix)
            first = np.rot90(second)
            third = np.delete(np.rot90(first), 0, 0)
            fourth = np.delete(o_matrix, 0, 0)

            second = np.array(second, dtype=float)
            first = np.array(first, dtype=float)
            third = np.array(third, dtype=float)
            fourth = np.array(fourth, dtype=float)

            half1 = np.concatenate((first, third))
            half2 = np.concatenate((second, fourth))

            full_matrix = np.concatenate((half1, half2), axis=1)
            full_matrix = np.delete(full_matrix, [9,], axis=1)
            iinput_full_matrix.append(full_matrix)

        self.iinput_tensor = np.zeros([10, 10, 5], dtype=np.float)
        self.iinput_tensor_full = np.zeros([19, 19, 5], dtype=np.float)

        for i in range(5):
            for j in range(10):
                for k in range(10):
                    self.iinput_tensor[j][k][i] = float(iinput_matrix[i][j][k]) * float(io_array[i])

        for i in range(5):
            for j in range(19):
                for k in range(19):
                    self.iinput_tensor_full[j][k][i] = float(iinput_full_matrix[i][j][k]) * float(io_array[i])

        self.output_tensor = np.zeros([100], dtype=np.float)

        for j in range(10):
            for k in range(10):
                index = j*10+k
                self.output_tensor[index] = output_matrix[0][j][k]

        self.flux_tensor = np.zeros([2, 100], dtype=np.float)

        for i in range(2):
            for j in range(10):
                for k in range(10):
                    index = j * 10 + k
                    self.flux_tensor[i, index] = float(flux_matrix[i][j][k])*float(flux_o_array[i])

        self.density_tensor_full = np.zeros([19, 19, 20], dtype=np.float)

        for i in range(20):
            density_temp = self.densities[i].get_value_matrix(0)

            o_matrix = density_temp

            second = np.rot90(o_matrix)
            first = np.rot90(second)
            third = np.delete(np.rot90(first), 0, 0)
            fourth = np.delete(o_matrix, 0, 0)

            second = np.array(second, dtype=float)
            first = np.array(first, dtype=float)
            third = np.array(third, dtype=float)
            fourth = np.array(fourth, dtype=float)

            half1 = np.concatenate((first, third))
            half2 = np.concatenate((second, fourth))

            full_matrix = np.concatenate((half1, half2), axis=1)

            full_matrix = np.delete(full_matrix, [9,], axis=1)
            for j in range(19):
                for k in range(19):
                    self.density_tensor_full[j][k][i] = float(full_matrix[j][k]) * float(self.densities[i].o1)

        self.summary_tensor = np.array(self.summary, dtype=float)



class CrossPowerSetN():

    def __init__(self,
                 summary,
                 abs,
                 nfis,
                 ss1,
                 ss2,
                 power,
                 flux,
                 densities,
                 iabs,
                 infis,
                 iss1,
                 iabsn,
                 infisn,
                 iss1n,
                 b2d,
                 peak,
                 p2dn,
                 p3dn,
                 pp3d,
                 i3absn,
                 i3nfisn,
                 i3ss1n,
                 ):
        self.abs = abs
        self.nfis = nfis
        self.ss1 = ss1
        self.ss2 = ss2
        self.power = power
        self.flux = flux
        self.densities = densities
        self.summary = summary
        self.iabs = iabs
        self.infis = infis
        self.iss1 = iss1
        self.b2d = b2d
        self.peak = peak
        self.p2dn = p2dn
        self.p3dn = p3dn
        self.pp3d = pp3d

        self.iabsn = iabsn
        self.infisn = infisn
        self.iss1n = iss1n

        self.i3absn = i3absn
        self.i3nfisn = i3nfisn
        self.i3ss1n = i3ss1n

        if len(self.abs.get_value_matrix(0)) == 0:
            raise IOError

        if len(self.nfis.get_value_matrix(0)) == 0:
            raise IOError

        if len(self.power.get_value_matrix(0)) == 0:
            raise IOError

        if len(self.flux.get_value_matrix(0)) == 0:
            raise IOError

        input_matrix = []
        iinput_matrix = []
        iinputn_matrix = []
        binput_matrix = []
        output_matrix = []
        flux_matrix = []
        batch_matrix = []

        batch_matrix.append(self.b2d.get_batch_matrix())

        input_matrix.append(self.abs.get_value_matrix(0))
        input_matrix.append(self.abs.get_value_matrix(1))
        input_matrix.append(self.nfis.get_value_matrix(0))
        input_matrix.append(self.nfis.get_value_matrix(1))
        input_matrix.append(self.ss2.get_value_matrix(0))

        iinput_matrix.append(self.iabs.get_value_matrix(0))
        iinput_matrix.append(self.iabs.get_value_matrix(1))
        iinput_matrix.append(self.infis.get_value_matrix(0))
        iinput_matrix.append(self.infis.get_value_matrix(1))
        iinput_matrix.append(self.iss1.get_value_matrix(0))

        binput_matrix.append(self.b2d.get_value_matrix(0))

        iinputn_matrix.append(self.iabsn.get_node_matrix(0))
        iinputn_matrix.append(self.iabsn.get_node_matrix(1))
        iinputn_matrix.append(self.infisn.get_node_matrix(0))
        iinputn_matrix.append(self.infisn.get_node_matrix(1))
        iinputn_matrix.append(self.iss1n.get_node_matrix(0))

        output_matrix.append(self.power.get_value_matrix(0))
        output_matrix.append(self.power.get_value_matrix(1))
        output_matrix.append(self.power.get_value_matrix(2))
        output_matrix.append(self.power.get_value_matrix(3))

        flux_matrix.append(self.flux.get_value_matrix(0))
        flux_matrix.append(self.flux.get_value_matrix(1))

        o_array = []
        o_array.append(self.abs.g1)
        o_array.append(self.abs.g2)
        o_array.append(self.nfis.g1)
        o_array.append(self.nfis.g2)
        o_array.append(self.ss2.g1)

        io_array = []
        io_array.append(self.iabs.g1)
        io_array.append(self.iabs.g2)
        io_array.append(self.infis.g1)
        io_array.append(self.infis.g2)
        io_array.append(self.iss1.g1)

        ion_array = []
        ion_array.append(self.iabsn.g1)
        ion_array.append(self.iabsn.g2)
        ion_array.append(self.infisn.g1)
        ion_array.append(self.infisn.g2)
        ion_array.append(self.iss1n.g1)



        flux_o_array = []
        flux_o_array.append(self.flux.g1)
        flux_o_array.append(self.flux.g2)

        input_full_matrix = []

        #fullcore expansion
        for i in range(5):

            o_matrix = input_matrix[i]
            o_matrix = np.array(o_matrix)
            o_matrix[:, 0] = o_matrix[0, :]

            second = np.rot90(o_matrix)
            first = np.rot90(second)
            third = np.delete(np.rot90(first), 0, 0)
            fourth = np.delete(o_matrix, 0, 0)

            second = np.array(second, dtype=float)
            first = np.array(first, dtype=float)
            third = np.array(third, dtype=float)
            fourth = np.array(fourth, dtype=float)

            half1 = np.concatenate((first, third))
            half2 = np.concatenate((second, fourth))

            full_matrix = np.concatenate((half1, half2), axis=1)
            full_matrix = np.delete(full_matrix, [9,], axis=1)

            input_full_matrix.append(full_matrix)

        self.input_tensor_full = np.zeros([19, 19, 5], dtype=np.float)

        for i in range(5):
            for j in range(19):
                for k in range(19):
                    self.input_tensor_full[j][k][i] = float(input_full_matrix[i][j][k])*float(o_array[i])

        iinput_full_matrix = []
        # fullcore expansion
        for i in range(5):
            o_matrix = iinput_matrix[i]
            o_matrix = np.array(o_matrix)
            o_matrix[:, 0] = o_matrix[0, :]

            second = np.rot90(o_matrix)
            first = np.rot90(second)
            third = np.delete(np.rot90(first), 0, 0)
            fourth = np.delete(o_matrix, 0, 0)

            second = np.array(second, dtype=float)
            first = np.array(first, dtype=float)
            third = np.array(third, dtype=float)
            fourth = np.array(fourth, dtype=float)

            half1 = np.concatenate((first, third))
            half2 = np.concatenate((second, fourth))

            full_matrix = np.concatenate((half1, half2), axis=1)
            full_matrix = np.delete(full_matrix, [9,], axis=1)
            iinput_full_matrix.append(full_matrix)

        self.iinput_tensor_full = np.zeros([19, 19, 5], dtype=np.float)

        for i in range(5):
            for j in range(19):
                for k in range(19):
                    self.iinput_tensor_full[j][k][i] = float(iinput_full_matrix[i][j][k]) * float(io_array[i])



        peak_matrix = []
        peak_matrix.append(self.peak.get_value_matrix(1))
        peak_full_matrix = []
        # fullcore expansion

        o_matrix = peak_matrix[0]
        o_matrix = np.array(o_matrix)
        o_matrix[:, 0] = o_matrix[0, :]

        second = np.rot90(o_matrix)
        first = np.rot90(second)
        third = np.delete(np.rot90(first), 0, 0)
        fourth = np.delete(o_matrix, 0, 0)

        second = np.array(second, dtype=float)
        first = np.array(first, dtype=float)
        third = np.array(third, dtype=float)
        fourth = np.array(fourth, dtype=float)

        half1 = np.concatenate((first, third))
        half2 = np.concatenate((second, fourth))

        full_matrix = np.concatenate((half1, half2), axis=1)
        full_matrix = np.delete(full_matrix, [9,], axis=1)

        peak_full_matrix.append(full_matrix)

        self.peak_tensor_full = np.zeros([19, 19, 1], dtype=np.float)

        for i in range(1):
            for j in range(19):
                for k in range(19):
                    self.peak_tensor_full[j][k][i] = float(peak_full_matrix[i][j][k])

        fr_matrix = []
        fr_matrix.append(self.peak.get_value_matrix(0))
        fr_full_matrix = []
        # fullcore expansion

        o_matrix = fr_matrix[0]
        o_matrix = np.array(o_matrix)
        o_matrix[:, 0] = o_matrix[0, :]

        second = np.rot90(o_matrix)
        first = np.rot90(second)
        third = np.delete(np.rot90(first), 0, 0)
        fourth = np.delete(o_matrix, 0, 0)

        second = np.array(second, dtype=float)
        first = np.array(first, dtype=float)
        third = np.array(third, dtype=float)
        fourth = np.array(fourth, dtype=float)

        half1 = np.concatenate((first, third))
        half2 = np.concatenate((second, fourth))

        full_matrix = np.concatenate((half1, half2), axis=1)
        full_matrix = np.delete(full_matrix, [9,], axis=1)
        fr_full_matrix.append(full_matrix)

        self.fr_tensor_full = np.zeros([19, 19, 1], dtype=np.float)

        for i in range(1):
            for j in range(19):
                for k in range(19):
                    self.fr_tensor_full[j][k][i] = float(fr_full_matrix[i][j][k])


        # fullcore expansion

        o_matrix = self.p2dn.get_node_matrix(0, 1)

        second = np.rot90(o_matrix)
        first = np.rot90(second)
        third = np.rot90(first)
        fourth = o_matrix

        second = np.array(second, dtype=float)
        first = np.array(first, dtype=float)
        third = np.array(third, dtype=float)
        fourth = np.array(fourth, dtype=float)

        half1 = np.concatenate((first, third))
        half2 = np.concatenate((second, fourth))

        full_matrix = np.concatenate((half1, half2), axis=1)

        self.p2dn_tensor_full = np.zeros([38, 38, 1], dtype=np.float)

        for i in range(1):
            for j in range(38):
                for k in range(38):
                    self.p2dn_tensor_full[j][k][i] =float(full_matrix[j][k])


        self.p3dn_tensor_full = np.zeros([38, 38, 26, 1], dtype=np.float)

        for plane in range(26):
            o_matrix = self.p3dn[plane].get_node_matrix(0, 1)

            second = np.rot90(o_matrix)
            first = np.rot90(second)
            third = np.rot90(first)
            fourth = o_matrix

            second = np.array(second, dtype=float)
            first = np.array(first, dtype=float)
            third = np.array(third, dtype=float)
            fourth = np.array(fourth, dtype=float)

            half1 = np.concatenate((first, third))
            half2 = np.concatenate((second, fourth))

            full_matrix = np.concatenate((half1, half2), axis=1)

            for i in range(1):
                for j in range(38):
                    for k in range(38):
                        self.p3dn_tensor_full[j][k][plane][i] = float(full_matrix[j][k])

        self.iinput3n_tensor_full = np.zeros([38, 38, 26, 5], dtype=np.float)

        for plane in range(1, 27):

            i3inputn_matrix = []

            i3on_array = []
            i3on_array.append(self.i3absn[plane].g1)
            i3on_array.append(self.i3absn[plane].g2)
            i3on_array.append(self.i3nfisn[plane].g1)
            i3on_array.append(self.i3nfisn[plane].g2)
            i3on_array.append(self.i3ss1n[plane].g1)

            i3inputn_matrix.append(self.i3absn[plane].get_node_matrix(0))
            i3inputn_matrix.append(self.i3absn[plane].get_node_matrix(1))
            i3inputn_matrix.append(self.i3nfisn[plane].get_node_matrix(0))
            i3inputn_matrix.append(self.i3nfisn[plane].get_node_matrix(1))
            i3inputn_matrix.append(self.i3ss1n[plane].get_node_matrix(0))

            for i in range(5):

                o_matrix = i3inputn_matrix[i]

                second = np.rot90(o_matrix)
                first = np.rot90(second)
                third = np.rot90(first)
                fourth = o_matrix

                second = np.array(second, dtype=float)
                first = np.array(first, dtype=float)
                third = np.array(third, dtype=float)
                fourth = np.array(fourth, dtype=float)

                half1 = np.concatenate((first, third))
                half2 = np.concatenate((second, fourth))

                full_matrix = np.concatenate((half1, half2), axis=1)

                for j in range(38):
                    for k in range(38):
                        self.iinput3n_tensor_full[j][k][plane-1][i] = float(full_matrix[j][k])*float(i3on_array[i])

        self.pp3d_tensor_full = np.zeros([19, 19, 26, 1], dtype=np.float)

        for plane in range(26):
            o_matrix = self.pp3d[plane].get_value_matrix(1)
            o_matrix = np.array(o_matrix)
            o_matrix[:, 0] = o_matrix[0, :]

            second = np.rot90(o_matrix)
            first = np.rot90(second)
            third = np.delete(np.rot90(first), 0, 0)
            fourth = np.delete(o_matrix, 0, 0)

            second = np.array(second, dtype=float)
            first = np.array(first, dtype=float)
            third = np.array(third, dtype=float)
            fourth = np.array(fourth, dtype=float)

            half1 = np.concatenate((first, third))
            half2 = np.concatenate((second, fourth))

            full_matrix = np.concatenate((half1, half2), axis=1)
            full_matrix = np.delete(full_matrix, [9, ], axis=1)
            fr_full_matrix.append(full_matrix)

            for i in range(1):
                for j in range(19):
                    for k in range(19):
                        self.pp3d_tensor_full[j][k][plane][i] = float(full_matrix[j][k])


        self.p3d_tensor_full = np.zeros([19, 19, 26, 1], dtype=np.float)

        for plane in range(26):
            o_matrix = self.pp3d[plane].get_value_matrix(0)
            o_matrix = np.array(o_matrix)
            o_matrix[:, 0] = o_matrix[0, :]

            second = np.rot90(o_matrix)
            first = np.rot90(second)
            third = np.delete(np.rot90(first), 0, 0)
            fourth = np.delete(o_matrix, 0, 0)

            second = np.array(second, dtype=float)
            first = np.array(first, dtype=float)
            third = np.array(third, dtype=float)
            fourth = np.array(fourth, dtype=float)

            half1 = np.concatenate((first, third))
            half2 = np.concatenate((second, fourth))

            full_matrix = np.concatenate((half1, half2), axis=1)
            full_matrix = np.delete(full_matrix, [9, ], axis=1)
            fr_full_matrix.append(full_matrix)

            for i in range(1):
                for j in range(19):
                    for k in range(19):
                        self.p3d_tensor_full[j][k][plane][i] = float(full_matrix[j][k])


        binput_full_matrix = []
        # fullcore expansion

        o_matrix = binput_matrix[0]
        o_matrix = np.array(o_matrix)
        o_matrix[:, 0] = o_matrix[0, :]

        second = np.rot90(o_matrix)
        first = np.rot90(second)
        third = np.delete(np.rot90(first), 0, 0)
        fourth = np.delete(o_matrix, 0, 0)

        second = np.array(second, dtype=float)
        first = np.array(first, dtype=float)
        third = np.array(third, dtype=float)
        fourth = np.array(fourth, dtype=float)

        half1 = np.concatenate((first, third))
        half2 = np.concatenate((second, fourth))

        full_matrix = np.concatenate((half1, half2), axis=1)
        full_matrix = np.delete(full_matrix, [9,], axis=1)
        binput_full_matrix.append(full_matrix)

        self.binput_tensor_full = np.zeros([19, 19, 1], dtype=np.float)

        for i in range(1):
            for j in range(19):
                for k in range(19):
                    self.binput_tensor_full[j][k][i] = float(binput_full_matrix[i][j][k])


        batch_full_matrix = []
        # fullcore expansion

        o_matrix = batch_matrix[0]
        o_matrix = np.array(o_matrix)
        o_matrix[:, 0] = o_matrix[0, :]

        second = np.rot90(o_matrix)
        first = np.rot90(second)
        third = np.delete(np.rot90(first), 0, 0)
        fourth = np.delete(o_matrix, 0, 0)

        second = np.array(second)
        first = np.array(first)
        third = np.array(third)
        fourth = np.array(fourth)

        half1 = np.concatenate((first, third))
        half2 = np.concatenate((second, fourth))

        full_matrix = np.concatenate((half1, half2), axis=1)
        full_matrix = np.delete(full_matrix, [9,], axis=1)
        batch_full_matrix.append(full_matrix)

        self.batch_tensor_full = np.zeros([19, 19, 2], dtype=str)

        for i in range(2):
            for j in range(19):
                for k in range(19):
                    if 0 == len(batch_full_matrix[0][j][k]):
                        self.batch_tensor_full[j][k][i] = " "
                    else:
                        self.batch_tensor_full[j][k][i] = (batch_full_matrix[0][j][k])[i]

        iinputn_full_matrix = []
        # fullcore expansion
        for i in range(5):
            o_matrix = iinputn_matrix[i]

            second = np.rot90(o_matrix)
            first = np.rot90(second)
            third = np.rot90(first)
            fourth = o_matrix

            second = np.array(second, dtype=float)
            first = np.array(first, dtype=float)
            third = np.array(third, dtype=float)
            fourth = np.array(fourth, dtype=float)

            half1 = np.concatenate((first, third))
            half2 = np.concatenate((second, fourth))

            full_matrix = np.concatenate((half1, half2), axis=1)
            iinputn_full_matrix.append(full_matrix)

        self.iinputn_tensor_full = np.zeros([38, 38, 5], dtype=np.float)

        for i in range(5):
            for j in range(38):
                for k in range(38):
                    self.iinputn_tensor_full[j][k][i] = float(iinputn_full_matrix[i][j][k]) * float(ion_array[i])

        self.output_tensor = np.zeros([100], dtype=np.float)

        for j in range(10):
            for k in range(10):
                index = j*10+k
                self.output_tensor[index] = output_matrix[0][j][k]

        self.flux_tensor = np.zeros([2, 100], dtype=np.float)

        for i in range(2):
            for j in range(10):
                for k in range(10):
                    index = j * 10 + k
                    self.flux_tensor[i, index] = float(flux_matrix[i][j][k])*float(flux_o_array[i])

        self.summary_tensor = np.array(self.summary, dtype=float)

class CrossPowerSet3D():

    def __init__(self,summary, abs, nfis, ss1, ss2, power, flux, densities):
        self.abs = abs
        self.nfis = nfis
        self.ss1 = ss1
        self.ss2 = ss2
        self.power = power
        self.flux = flux
        self.densities = densities
        self.summary = summary

        if len(self.abs[0].get_value_matrix(0)) == 0:
            raise IOError

        if len(self.nfis[0].get_value_matrix(0)) == 0:
            raise IOError

        if len(self.power[0].get_value_matrix(0)) == 0:
            raise IOError


        output_matrix = []
        input_full_matrix_3D = []

        for _x in range(26):

            input_matrix = []
            full_matrix_3D = []

            input_matrix.append(self.abs[_x].get_value_matrix(0))
            input_matrix.append(self.abs[_x].get_value_matrix(1))
            input_matrix.append(self.nfis[_x].get_value_matrix(0))
            input_matrix.append(self.nfis[_x].get_value_matrix(1))
            input_matrix.append(self.ss1[_x].get_value_matrix(0))
            input_matrix.append(self.ss1[_x].get_value_matrix(1))
            input_matrix.append(self.ss2[_x].get_value_matrix(0))
            input_matrix.append(self.ss2[_x].get_value_matrix(1))

            output_matrix.append(self.power[_x].get_value_matrix(0))

            o_array = []
            o_array.append(self.abs[_x].g1)
            o_array.append(self.abs[_x].g2)
            o_array.append(self.nfis[_x].g1)
            o_array.append(self.nfis[_x].g2)
            o_array.append(self.ss1[_x].g1)
            o_array.append(self.ss1[_x].g2)
            o_array.append(self.ss2[_x].g1)
            o_array.append(self.ss2[_x].g2)

            #fullcore expansion
            for i in range(8):

                o_matrix = input_matrix[i]

                second = np.rot90(o_matrix)
                first = np.rot90(second)
                third = np.delete(np.rot90(first), 0, 0)
                fourth = np.delete(o_matrix, 0, 0)

                second = np.array(second, dtype=float)
                first = np.array(first, dtype=float)
                third = np.array(third, dtype=float)
                fourth = np.array(fourth, dtype=float)

                half1 = np.concatenate((first, third))
                half2 = np.concatenate((second, fourth))
                half2 = np.delete(half2, 0, 1)
                full_matrix = np.concatenate((half1, half2), axis=1)
                full_matrix = full_matrix*float(o_array[i])
                full_matrix_3D.append(full_matrix)

            input_full_matrix_3D.append(full_matrix_3D)

        input_full_matrix_3D = np.array(input_full_matrix_3D)
        input_full_matrix_3D = np.swapaxes(input_full_matrix_3D, 1, 3)
        self.input_tensor_full = np.swapaxes(input_full_matrix_3D, 0, 2)

        self.output_tensor = np.zeros([26, 100], dtype=np.float)

        for i in range(26):
            for j in range(10):
                for k in range(10):
                    index = j*10+k
                    self.output_tensor[i][index] = output_matrix[i][j][k]

        batch_matrix = self.abs[0].get_batch_matrix()

        self.batch_tensor = ['']*100

        for j in range(10):
            for k in range(10):
                index = j * 10 + k
                self.batch_tensor[index] = batch_matrix[j][k]

        self.summary_tensor = np.array(self.summary, dtype=float)


