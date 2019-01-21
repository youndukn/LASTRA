from peewee import *
import optparse
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import pandas as pd

ad = [
    [0, 9, 9],
    [1, 9, 10],
    [2, 9, 11],
    [3, 9, 12],
    [4, 9, 13],
    [5, 9, 14],
    [6, 9, 15],
    [7, 9, 16],
    [11, 10, 10],
    [12, 10, 11],
    [13, 10, 12],
    [14, 10, 13],
    [15, 10, 14],
    [16, 10, 15],
    [17, 10, 16],
    [22, 11, 11],
    [23, 11, 12],
    [24, 11, 13],
    [25, 11, 14],
    [26, 11, 15],
    [27, 11, 16],
    [33, 12, 12],
    [34, 12, 13],
    [35, 12, 14],
    [36, 12, 15],
    [44, 13, 13],
    [45, 13, 14],
    [46, 13, 15],
    [55, 14, 14],
]

ad55 = [
    [0, 9, 9],
    [1, 9, 10],
    [2, 9, 11],
    [3, 9, 12],
    [4, 9, 13],
    [5, 9, 14],
    [6, 9, 15],
    [7, 9, 16],
    [8, 9, 17],
    [9, 9, 18],
    [11, 10, 10],
    [12, 10, 11],
    [13, 10, 12],
    [14, 10, 13],
    [15, 10, 14],
    [16, 10, 15],
    [17, 10, 16],
    [18, 10, 17],
    [19, 10, 18],
    [22, 11, 11],
    [23, 11, 12],
    [24, 11, 13],
    [25, 11, 14],
    [26, 11, 15],
    [27, 11, 16],
    [28, 11, 17],
    [29, 11, 18],
    [33, 12, 12],
    [34, 12, 13],
    [35, 12, 14],
    [36, 12, 15],
    [37, 12, 16],
    [38, 12, 17],
    [39, 12, 18],
    [44, 13, 13],
    [45, 13, 14],
    [46, 13, 15],
    [47, 13, 16],
    [48, 13, 17],
    [49, 13, 18],
    [55, 14, 14],
    [56, 14, 15],
    [57, 14, 16],
    [58, 14, 17],
    [59, 14, 18],
    [66, 15, 15],
    [67, 15, 16],
    [68, 15, 17],
    [69, 15, 18],
    [77, 16, 16],
    [78, 16, 17],
    [79, 16, 18],
    [88, 17, 17],
    [89, 17, 18],
    [99, 18, 18],
]



avg_index3 = [0.30800407, 0.21221341, 0.16709152, 0.1976553,  0.15561875, 0.16482077,
 0.26807004, 1.01336702, 0.18794173, 0.15859129, 0.14659841, 0.13182862,
 0.42721009, 0.26529106, 0.89423283, 0.25923964, 0.14511788, 0.15055152,
 0.17877154, 0.14592355, 0.45353179 ,0.18079516, 0.194133,   0.12352021,
 0.18987846, 0.16645453, 0.22849735, 0.40964016, 0.9505142 ]

avg_index = [0.29232968 , 0.27965838 , 0.16460408 , 0.27725363 , 0.1402501  , 0.24973505
 , 0.36374051 , 0.27923595 , 0.17495249 , 0.18158681 , 0.20470527 , 0.1453522
 , 0.1369815  , 0.15295642 , 0.28669556 , 0.13026588 , 0.13030828 , 0.20370394
 , 0.34184552 , 0.16909361 , 0.38661858 , 0.1632925  , 0.15197417 , 0.21991185
 , 0.1837379  , 0.32768585 , 0.25544579 , 0.31583586 , 0.43333698]

avg_index5 = [0.61394069 , 0.40181468 , 0.23825301 , 0.28287566 , 0.15907406 , 0.15982517
 , 0.19206071 , 0.43346151 , 0.31618715 , 0.18891083 , 0.15116315 , 0.13443601
 , 0.18277041 , 0.22727218 , 0.35839664 , 0.17825495 , 0.13786578 , 0.32491639
 , 0.17521039 , 0.18464096, 1.20354664 , 0.15666656 , 0.18481612 , 0.20336187
 , 0.35479143 , 0.2135847  , 0.22675557 , 0.86573847 , 0.80454506]

avg_index_depl=[1.63664015 ,0.87077278 ,1.20778947 ,0.77676505 ,0.47093103 ,0.71130695
 ,0.73775893 ,1.15716701 ,0.72676066 ,0.59873418 ,0.53774128 ,0.46125124
 ,0.58584481 ,0.53652097 ,2.31798386 ,0.62668225 ,0.67804659 ,0.49120998
 ,0.53240538 ,0.63988145 ,2.36677816 ,0.64646347 ,0.55377165 ,0.50421068
 ,0.60851401 ,0.72139966 ,0.62879033 ,1.43569583 ,1.41199373]


avg_index_depl_3=[0.99660004	 ,0.70437426	 ,0.5255726	 ,0.33858115	 ,0.41384747	 ,0.4782529
	 ,0.51887309	,1.13570799	 ,0.5413182	 ,0.37019261	 ,0.36440686	 ,0.29331005
	 ,0.39821928	 ,0.61228228	 ,0.8151648	 ,0.53124357	 ,0.37203728	 ,0.39146015
	 ,0.42204554	 ,0.67653921	,1.80133564	 ,0.58896495	 ,0.35335272	 ,0.25314721
	,1.13113673	 ,0.41843028	 ,0.6614809	,1.17990343	 ,0.87608019	]


avg_index_depl_3_3=[0.78288826  ,0.52051467  ,0.51080264  ,0.38295868  ,0.4936582   ,0.38791594
  ,0.73001342  ,0.73035423  ,0.51686835  ,0.36357911  ,0.42411975  ,0.34946822
  ,0.6242747   ,0.43663747  ,0.64312353  ,0.38838326  ,0.39693867  ,0.48183809
  ,0.57382236  ,0.51999128 ,1.04469821  ,0.52316116  ,0.57297431  ,0.41847578
  ,0.92735023  ,0.67791253  ,0.66476704 ,1.27921593  ,0.65727703]

avg_index_depl_5_3=[0.59657905  ,0.46027812  ,0.37657197  ,0.30389825  ,0.52457781  ,0.35721063
  ,0.50667839  ,0.53557078  ,0.38050172  ,0.34012625  ,0.30585557  ,0.3507391
  ,0.3948271   ,0.44977491  ,0.49976897  ,0.42285287  ,0.37014563  ,0.45675287
  ,0.35803596  ,0.55935622  ,0.47890701  ,0.36600825  ,0.24298351  ,0.3375141
  ,0.5072769   ,0.33799071  ,0.72472489  ,0.49957469  ,0.73873435]




max_dist = 1768


db = SqliteDatabase('distribution_2.db')

class Distribution(Model):

    maxError = FloatField()
    maxErrorIndex = IntegerField()

    avgError = FloatField()

    maxPower = FloatField()
    maxPowerIndex = IntegerField()

    minPower = FloatField()
    minPowerIndex = IntegerField()

    minMaxDistance = FloatField()

    file_path = CharField()

    burnup_step = FloatField()

    fxy = FloatField()

    class Meta:
        database = db

def top_error(limit):
    dist = Distribution.select().order_by(Distribution.maxError.desc()).limit(limit)
    return dist


def error_between(low, high, limit):
    dist = Distribution.select().where(Distribution.maxError.between(low, high)).limit(limit)
    return dist

def top_avg_error(limit):
    dist = Distribution.select().order_by(Distribution.avgError.desc()).limit(limit)
    return dist

def top_max(limit):
    dist = Distribution.select().order_by(Distribution.maxPower.desc()).limit(limit)
    return dist

def top_min(limit):
    dist = Distribution.select().order_by(Distribution.minPower.asc()).limit(limit)
    return dist

def low_between(low, high, limit):
    dist = Distribution.select().where(Distribution.minPower.between(low, high)).limit(limit)
    return dist

def high_between(low, high, limit):
    dist = Distribution.select().where(Distribution.maxPower.between(low, high)).limit(limit)
    return dist

def power_index_at(index, limit):
    dist = Distribution.select().where(Distribution.maxPowerIndex == index).limit(limit)
    return dist

def error_index_at(index, limit):
    dist = Distribution.select().where(Distribution.maxErrorIndex == index).limit(limit)
    return dist

def print_max_error(print_limit = 10):

    errors = top_error(max_dist)

    error = []

    for value in errors:
        error.append(value.maxError)
    print ("Maximum Error occurs")
    print_list(errors, error, print_limit, "Maximum Error")


def print_avg_error(print_limit = 3):

    errors = top_avg_error(max_dist)

    error = []

    for value in errors:
        error.append(value.avgError)

    print("Maximum Average Error occurs")

    print_list(errors, error, print_limit, "Average Error")


def print_max_power(print_limit = 10):

    low = 1.7
    high = 2.0
    highs = high_between(low, high, max_dist)

    print("Maximum Power between {} and {} count is {}".format(low, high, len(highs)))

    maximum = []
    top_highs = top_max(max_dist)
    for value in top_highs:
        maximum.append(value.maxPower)

    print_list(top_highs, maximum, print_limit, "Maximum Power")

def print_min_power(print_limit = 10):


    low = 0
    high = 0.2
    lows = low_between(low, high, max_dist)

    print("Minimum Power between {} and {} count is {}".format(low, high, len(lows)))

    minimum = []

    top_lows = top_min(max_dist)
    for value in top_lows:
        minimum.append(value.minPower)

    print_list(top_lows, minimum, print_limit, "Minimum Power")

def print_list(distributions, values ,limit, title ):
    for index in range(0, min(len(distributions), limit)):
        dist = distributions[index]
        print(
            "Max: {}, Min: {}, Error: {}, AvgError: {}, file: {}".format(dist.maxPower, dist.minPower, dist.maxError,
                                                                         dist.avgError,
                                                                         dist.file_path))
    print()
    plt.title(title)
    x = np.array(values)
    sns.distplot(x)

    plt.show()

def print_max_power_dist(print_limit = 10):

    distributions = []

    for i, value in enumerate(ad):
        distributions.append(int(len(power_index_at(i, max_dist))/10))

    print_plot_distribution(distributions, "Maximum Power( Count X 10 )", 100)


def print_max_error_dist(print_limit = 10):

    distributions = []

    for i, value in enumerate(ad):
        distributions.append(int(len(error_index_at(i, max_dist))/10))

    print_plot_distribution(distributions, "Maximum Error( Count X 10 )", 100)

def print_max_error_value_dist(print_limit = 10):
    errors = error_between(1, 100, max_dist)
    print("Max count", len(error_between(0, 1000, max_dist)))

    errors_array = []
    err_array = []
    min_array = []
    for dist in errors:
        errors_array.append(dist)
        err_array.append(dist.maxError)
        min_array.append(dist.minPower)
    plt.title("Max Error/Min Power")
    plt.plot(min_array, err_array, 'ro')
    plt.show()

    errors_array = []
    err_array = []
    min_array = []
    for dist in errors:
        errors_array.append(dist)
        err_array.append(dist.maxError)
        min_array.append(dist.fxy)
    plt.title("Avg Error/Fxy")
    plt.plot(min_array, err_array, 'bo')
    plt.show()

    distributions = []

    for i, value in enumerate(ad):
        values = []
        maxError = 0
        for xvalue in error_index_at(i, max_dist):
            if xvalue.minPower > 0.22:
                if xvalue.maxError >10:
                    print(
                        "Max: {}, Min: {}, Error: {}, AvgError: {}, file: {}".format(xvalue.maxPower, xvalue.minPower,
                                                                                     xvalue.maxError,
                                                                                     xvalue.avgError,
                                                                                     xvalue.file_path))
                values.append(xvalue.maxError)

        if len(values)> 0:
            maxError = np.max(np.array(values))
        distributions.append(maxError)

    print_plot_distribution(distributions, "Maximum Error Value", 10)

def print_avg_error_value_dist(print_limit = 10):
    print_plot_distribution(avg_index, "BOC Average Error", 0)
    print_plot_distribution(avg_index_depl_3, "Average Error Depletion 3F 1M", 0)
    print_plot_distribution(avg_index_depl_3_3, "Average Error Depletion 3,3F 2M", 0)
    print_plot_distribution(avg_index_depl_5_3, "Average Error Depletion 5,3F 2M", 0)

def print_plot_distribution(distribution, title, maximum):

    print()
    matrix = []
    for row in range(9, 17):
        array = []
        string = ""
        for column in range(9, 17):
            value_there = False
            for index, indexes in enumerate(ad):
                if indexes[1] == row and indexes[2] == column:
                    value_there = True
                    break
            if value_there:
                string += "{:4.2f} ".format(float(distribution[index]))
                array.append(distribution[index])
            else:
                string += "      "
                array.append(-1*maximum)
        matrix.append(array)

        print(string)

    dist_org = np.array(matrix)
    plt.figure(figsize=(10,7))
    plt.title(title)

    df = pd.DataFrame(dist_org)
    sns.heatmap(df, annot=True, annot_kws={"size": 10})

    plt.show()

if __name__ == '__main__':

    db.connect()
    db.create_tables([Distribution], safe=True)

    parser = optparse.OptionParser()

    parser.add_option('-q', '--question',
                      action="store", dest="question",
                      help="Question 1~4", default="all")

    parser.add_option('-l', '--limit',
                      action="store", dest="limit",
                      help="Limit 1~4", default="0")

    options, args = parser.parse_args()

    if options.question == "all":
        if options.limit != "0":
            print_avg_error_value_dist(int(options.limit))
            print_max_error_value_dist(int(options.limit))
            print_max_power_dist(int(options.limit))
            print_max_error_dist(int(options.limit))
            print_avg_error(int(options.limit))
            print_max_error(int(options.limit))
            print_max_power(int(options.limit))
            print_min_power(int(options.limit))

        else:
            print_avg_error_value_dist()
            print_max_error_value_dist()
            print_max_power_dist()
            print_max_error_dist()
            print_avg_error()
            print_max_error()
            print_max_power()
            print_min_power()
    else:
        if options.question == "maxp":
            if options.limit != "0":
                print_max_power(int(options.limit))
            else:
                print_max_power()
        if options.question == "avge":
            if options.limit != "0":
                print_max_power(int(options.limit))
            else:
                print_max_power()
        if options.question == "minp":
            if options.limit != "0":
                print_min_power(int(options.limit))
            else:
                print_min_power()
        if options.question == "maxe":
            if options.limit != "0":
                print_max_error(int(options.limit))
            else:
                print_max_error()
        if options.question == "maxdp":
            if options.limit != "0":
                print_max_power_dist(int(options.limit))
            else:
                print_max_power_dist()
        if options.question == "maxde":
            if options.limit != "0":
                print_max_error_dist(int(options.limit))
            else:
                print_max_error_dist()

    print ('Query string:', options.question)

