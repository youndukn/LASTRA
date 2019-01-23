import glob
from keras.layers import Lambda
import keras.backend as K


def isInt(s):
    try:
        int(s)
        return True
    except ValueError:
        return False

def get_files(excludes, shuffled=""):
    files = []
    for path in glob.glob("/media/youndukn/lastra/plants_data/*{}.inp".format(shuffled)):
        try:

            isExclude = False
            for exclude in excludes:
                if exclude in path:
                    isExclude = True
            if not isExclude:
                file_read = open(path, 'rb')
                files.append(file_read)
        except:
            pass

    return files

def get_files_with(includes, excludes, shuffled="", folders=["/media/youndukn/lastra/plants_data/"]):
    files = []
    for folder in folders:
        for path in glob.glob("{}*{}*".format(folder, shuffled)):
            try:

                isInclude = False
                for include in includes:
                    if include in path:
                        isInclude = True

                if len(includes) == 0:
                    isInclude = True

                isExclude = False
                for exclude in excludes:
                    if exclude in path:
                        isExclude = True

                if isInclude and not isExclude:
                    file_read = open(path, 'rb')
                    files.append(file_read)
            except:
                pass

    return files



def gSlicer3D_C():
    def func(x):
        return x[:,:,:8,:8, :]
    return Lambda(func)

def nodeColPermute3D_C():
    def func(x):
        x = K.permute_dimensions(x[:, :, 0:1, :, :], (0, 1, 3, 2, 4))
        return x
    return Lambda(func)

def nodeRowPermute3D_C():
    def func(x):
        x = K.permute_dimensions(x[:, :, :, 0:1, :], (0, 1, 3, 2, 4))
        return x
    return Lambda(func)

def nodeCen3D_C():
    def func(x):
        return x[:, :, 0:1, 0:1, :]
    return Lambda(func)

def assColPermute3D_C():
    def func(x):
        x = K.permute_dimensions(x[:, :, 1:2, :, :], (0, 1, 3, 2, 4))
        return x
    return Lambda(func)

def assRowPermute3D_C():
    def func(x):
        x = K.permute_dimensions(x[:, :, :, 1:2, :], (0, 1, 3, 2, 4))
        return x
    return Lambda(func)


def assCen3D_C():
    def func(x):
        return x[:, :, 1:2, 1:2, :]
    return Lambda(func)
