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


