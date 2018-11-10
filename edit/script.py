import random
def buildrawdata():
    basepath = r"D:\users\wuyu\Data\DoubanReverse37Trick"

    #@f = open(r"F:\data\conversation\mergetmp8.txt","r",encoding="utf-8")
    f1 =open(basepath+r"\Train\query.txt","r",encoding="utf-8")
    f2 =open(basepath+r"\Train\response.txt","r",encoding="utf-8")
    f3 = open(basepath+r"\Train\meta.txt","r",encoding="utf-8")
    fm = open(basepath+r"\Train\meta3.txt","w",encoding="utf-8")
    # f3 =open(basepath+r"\Dev\query.txt","r",encoding="utf-8")
    # f4 = open(basepath+r"\Dev\response.txt","r",encoding="utf-8")
    # # f3m =open(basepath+r"\Dev\meta3.txt","w",encoding="utf-8")
    # f5 = open(basepath+r"\Test\query.txt","r",encoding="utf-8")
    # f6 = open(basepath+r"\Test\response.txt","r",encoding="utf-8")
    # f5m = open(basepath+r"\Test\meta3.txt","w",encoding="utf-8")

    count = 0
    for line in f1:
        line2 = f2.readline()
        prototype = line.strip().split()
        response = line2.strip().split()

        ins = []
        dels = []

        for w in prototype:
            if w not in response and w not in dels:
                dels.append(w)
        for w in response:
            if w not in prototype and w not in ins:
                ins.append(w)

        #if random.randint(2) == 1:
        fm.write("{0}\t{1}\t{2}\n".format(line.strip()," ".join(dels)," ".join(ins)))

        # if count < 10000:
        #     f3.write(tmp[1])
        #     f3.write("\n")
        #     f4.write(tmp[3])
        #     f4.write("\n")
        #     f3m.write(u"{0}\t{1}\n".format(tmp[0],tmp[2]))
        #
        # elif count < 20000:
        #     f5.write(tmp[1])
        #     f5.write("\n")
        #     f6.write(tmp[3])
        #     f6.write("\n")
        #     f5m.write(u"{0}\t{1}\n".format(tmp[0], tmp[2]))

buildrawdata()

def buildquerymeta():
    f = open(r"D:\users\wuyu\Data\DoubanReverse37Trick\Train\query.txt","r",encoding="utf-8")
    f2 = open(r"D:\users\wuyu\Data\DoubanReverse37Trick\Train\meta.txt","r",encoding="utf-8")
    fw = open(r"D:\users\wuyu\Data\DoubanReverse37Trick\Train\query_meta.txt","w",encoding="utf-8")
    for line in f:
        line2 = f2.readline().strip()
        query,prototype = line2.split('\t')

        q_words = query.split()
        p_words = prototype.split()

        ins = []
        dele = []
        for w in q_words:
            if w not in p_words:
                dele.append(w)
        for w2 in p_words:
            if w2 not in q_words:
                ins.append(w2)
        fw.write("{0}\t{1}\t{2}\n".format(line.strip()," ".join(ins)," ".join(dele)))
#buildquerymeta()
def buildquerymeta2():
    f = open(r"D:\users\wuyu\Data\DoubanReverse37Trick\Train\query.txt","r",encoding="utf-8")
    f2 = open(r"D:\users\wuyu\Data\DoubanReverse37Trick\Train\meta.txt","r",encoding="utf-8")
    fw = open(r"D:\users\wuyu\Data\DoubanReverse37Trick\Train\query_meta2.txt","w",encoding="utf-8")
    for line in f:
        line2 = f2.readline().strip()
        query,prototype = line2.split('\t')

        q_words = query.split()
        p_words = prototype.split()


        maintain = []
        ins = []
        dele = []
        for w in q_words:
            if w not in p_words:
                dele.append(w)
            else:
                maintain.append(w)
        for w2 in p_words:
            if w2 not in q_words:
                ins.append(w2)
        fw.write("{0}\t{1}\t{2}\t{3}\n".format(line.strip()," ".join(ins)," ".join(dele)," ".join(maintain)))
#buildquerymeta2()




def buildresponsemeta():
    f = open(r"D:\users\wuyu\Data\DoubanReverse37\Train\response.txt","r",encoding="utf-8")
    f2 = open(r"D:\users\wuyu\Data\DoubanReverse37\Train\meta.txt","r",encoding="utf-8")
    fw = open(r"D:\users\wuyu\Data\DoubanReverse37\Train\response_meta.txt","w",encoding="utf-8")
    for line in f:
        line2 = f2.readline().strip()
        query,prototype = line2.split('\t')

        q_words = query.split()
        p_words = prototype.split()

        ins = []
        dele = []
        for w in q_words:
            if w not in p_words:
                dele.append(w)
        for w2 in p_words:
            if w2 not in q_words:
                ins.append(w2)
        fw.write("{0}\t{1}\t{2}\n".format(line.strip()," ".join(dele)," ".join(ins)))


def buildtestmeta():
    f = open(r"D:\users\wuyu\Data\DoubanReverseQuality\Test\1w\postrerank\query.search.10m.txt","r",encoding="utf-8")
    #f2 = open(r"D:\users\wuyu\Data\DoubanReverse37\Train\meta.txt","r",encoding="utf-8")
    fw = open(r"D:\users\wuyu\Data\DoubanReverseQuality\Test\1w\postrerank\query.search.10m.meta","w",encoding="utf-8")
    fw2 = open(r"D:\users\wuyu\Data\DoubanReverseQuality\Test\1w\postrerank\trueres","w",encoding="utf-8")
    fw3 = open(r"D:\users\wuyu\Data\DoubanReverseQuality\Test\1w\postrerank\query.search.10m.valid.txt","w",encoding="utf-8")
    for line in f:
        tmp = line.split('\t')
        query,prototype = tmp[0], tmp[2]

        if len(tmp[-1].split()) > 30:
            continue

        q_words = query.split()
        p_words = prototype.split()

        ins = []
        dele = []
        for w in q_words:
            if w not in p_words:
                dele.append(w)
        for w2 in p_words:
            if w2 not in q_words:
                ins.append(w2)
        fw.write("{0}\t{1}\t{2}\n".format(tmp[-1].strip()," ".join(ins)," ".join(dele)))
        fw2.write(tmp[1])
        fw2.write("\n")
        fw3.write(line)
#buildtestmeta()

def buildeditn():
    f = open(r"D:\users\wuyu\Data\DoubanReverseQuality\Test\1w\postrerank\query.search.10m.valid.txt","r",encoding="utf-8")
    f2 = open(r"D:\users\wuyu\pythoncode\seq2seq_pt\edit\pred.large.e5.txt","r",encoding="utf-8")
    fw = open(r"D:\users\wuyu\Data\DoubanReverseQuality\Test\1w\postrerank\genearted_results.large5.txt","w",encoding="utf-8")
    for line2 in f2:
        line = f.readline()
        tmp = line.split('\t')

        fw.write("{0}\t{1}\t{2}\t{3}\n".format(tmp[0],tmp[1],tmp[2],line2.strip()))
#buildeditn()
#buildresponsemeta()