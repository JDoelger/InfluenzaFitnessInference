import pickle


def pickle_out_data(data, pickle_name):
    pickle_out = open(pickle_name, "wb")
    pickle.dump(data, pickle_out)
    pickle_out.close()


def pickle_in_data(pickle_name):
    pickle_in = open(pickle_name, "rb")
    sequences = pickle.load(pickle_in)
    pickle_in.close()

    return sequences


def write_results(data, i):
    pickle.dump(data, open("trajectory_" + str(i) + ".p", "wb"))
    # f = open("trajectory_{0}".format(i), "w")
    # for line in data:
    #    x = [l for l in line]
    #    for entry in x:
    #        f.write(str(entry) + " ")
    #    f.write("\n")
    # f.close()


def write_val(val, i):
    pickle.dump(val, open("valencies_" + str(i) + ".p", "wb"))
    # f = open("valencies_{0}".format(i), "w")
    # for line in val:
    #    x = [l for l in line]
    #    for entry in x:
    #        f.write(str(entry) + " ")
    #    f.write("\n")
    # f.close()


def write_LATtoSVs(LATtoSVs, i):  #
    pickle.dump(LATtoSVs, open("LATtoSVs_" + str(i) + ".p", "wb"))


def write_adj(adj, i):
    pickle.dump(adj, open("adj_matrix_" + str(i) + ".p", "wb"))
    # f = open("adj_matrix_{0}".format(i), "w")
    # for line in adj:
    #    x = [l for l in line]
    #    for entry in x:
    #        f.write(str(entry) + " ")
    #    f.write("\n")
    #f.close()
