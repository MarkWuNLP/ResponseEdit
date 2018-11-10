import logging
import torch
import s2s

try:
    import ipdb
except ImportError:
    pass

lower = True
seq_length = 50
report_every = 100000
shuffle = 1

logger = logging.getLogger(__name__)


def makeVocabulary(filenames, size):
    vocab = s2s.Dict([s2s.Constants.PAD_WORD, s2s.Constants.UNK_WORD,
                      s2s.Constants.BOS_WORD, s2s.Constants.EOS_WORD], lower=lower)
    for filename in filenames:
        with open(filename, encoding='utf-8') as f:
            for sent in f.readlines():
                for word in sent.strip().replace('\t', ' ').split(' '): # add tab for split
                    if word:
                        vocab.add(word)

    vocab.labelToIdx[''] = 0 # add null str
    originalSize = vocab.size()
    vocab = vocab.prune(size)
    logger.info('Created dictionary of size %d (pruned from %d)' %
                (vocab.size(), originalSize))

    return vocab


def initVocabulary(name, dataFiles, vocabFile, vocabSize):
    vocab = None
    if vocabFile is not None:
        # If given, load existing word dictionary.
        logger.info('Reading ' + name + ' vocabulary from \'' + vocabFile + '\'...')
        vocab = s2s.Dict(lower=lower)
        vocab.loadFile(vocabFile)
        logger.info('Loaded ' + str(vocab.size()) + ' ' + name + ' words')

    if vocab is None:
        # If a dictionary is still missing, generate it.
        logger.info('Building ' + name + ' vocabulary...')
        genWordVocab = makeVocabulary(dataFiles, vocabSize)

        vocab = genWordVocab

    return vocab


def saveVocabulary(name, vocab, file):
    logger.info('Saving ' + name + ' vocabulary to \'' + file + '\'...')
    vocab.writeFile(file)


def makeMultiSourceData(srcFile, tgtFile, srcDicts, tgtDicts):
    src, src_ins, src_del, tgt, src_query = [], [], [], [], []
    sizes = []
    count, ignored = 0, 0

    logger.info('Processing %s & %s ...' % (srcFile, tgtFile))
    srcF = open(srcFile, encoding='utf-8')
    tgtF = open(tgtFile, encoding='utf-8')

    while True:
        sline = srcF.readline()
        tline = tgtF.readline()

        # normal end of file
        if sline == "" and tline == "":
            break

        # source or target does not have same number of lines
        if sline == "" or tline == "":
            logger.info('WARNING: source and target do not have the same number of sentences')
            break

        sline_items = sline.split('\t')
        if len(sline_items) != 4:
            logger.info('WARNING: exist wrong format in source, right format: sentence\tinsert\tdelete\tmaintain')
            break

        sline = sline_items[0].strip()
        sline_ins = sline_items[1].strip()
        sline_del = sline_items[2].strip()
        sline_query = sline_items[3].strip()
        tline = tline.strip()

        # source and/or target are empty
        if sline == "" or tline == "":
            logger.info('WARNING: ignoring an empty line (' + str(count + 1) + ')')
            continue

        srcWords = sline.split(' ')
        srcInsWords = sline_ins.split(' ')
        srcDelWords = sline_del.split(' ')
        queryWords = sline_query.split(' ')
        tgtWords = tline.split(' ')

        # TODO: limit the size of insert words and delete words
        if len(srcWords) <= seq_length and len(tgtWords) <= seq_length:
            src += [srcDicts.convertToIdx(srcWords,
                                          s2s.Constants.UNK_WORD)]
            src_ins += [tgtDicts.convertToIdx(srcInsWords,
                                          s2s.Constants.UNK_WORD)]
            src_del += [tgtDicts.convertToIdx(srcDelWords,
                                          s2s.Constants.UNK_WORD)]
            src_query += [tgtDicts.convertToIdx(queryWords,
                                          s2s.Constants.UNK_WORD)]
            tgt += [tgtDicts.convertToIdx(tgtWords,
                                          s2s.Constants.UNK_WORD,
                                          s2s.Constants.BOS_WORD,
                                          s2s.Constants.EOS_WORD)]

            sizes += [len(srcWords)]
        else:
            ignored += 1

        count += 1

        if count % report_every == 0:
            logger.info('... %d sentences prepared' % count)

    srcF.close()
    tgtF.close()

    if shuffle == 1:
        logger.info('... shuffling sentences')
        perm = torch.randperm(len(src))
        src = [src[idx] for idx in perm]
        src_ins = [src_ins[idx] for idx in perm]
        src_del = [src_del[idx] for idx in perm]
        src_query = [src_query[idx] for idx in perm]
        tgt = [tgt[idx] for idx in perm]
        sizes = [sizes[idx] for idx in perm]

    logger.info('... sorting sentences by size')
    _, perm = torch.sort(torch.Tensor(sizes))
    src = [src[idx] for idx in perm]
    src_ins = [src_ins[idx] for idx in perm]
    src_del = [src_del[idx] for idx in perm]
    src_query = [src_query[idx] for idx in perm]
    tgt = [tgt[idx] for idx in perm]

    logger.info('Prepared %d sentences (%d ignored due to length == 0 or > %d)' %
                (len(src), ignored, seq_length))
    return src, src_ins, src_del,src_query, tgt

def makeData(srcFile, tgtFile, srcDicts, tgtDicts):
    src, src_ins, src_del, tgt = [], [], [], []
    sizes = []
    count, ignored = 0, 0

    logger.info('Processing %s & %s ...' % (srcFile, tgtFile))
    srcF = open(srcFile, encoding='utf-8')
    tgtF = open(tgtFile, encoding='utf-8')

    while True:
        sline = srcF.readline()
        tline = tgtF.readline()

        # normal end of file
        if sline == "" and tline == "":
            break

        # source or target does not have same number of lines
        if sline == "" or tline == "":
            logger.info('WARNING: source and target do not have the same number of sentences')
            break

        sline_items = sline.split('\t')
        if len(sline_items) != 3:
            logger.info('WARNING: exist wrong format in source, right format: sentence\tinsert\tdelete')
            break

        sline = sline_items[0].strip()
        sline_ins = sline_items[1].strip()
        sline_del = sline_items[2].strip()
        tline = tline.strip()

        # source and/or target are empty
        if sline == "" or tline == "":
            logger.info('WARNING: ignoring an empty line (' + str(count + 1) + ')')
            continue

        srcWords = sline.split(' ')
        srcInsWords = sline_ins.split(' ')
        srcDelWords = sline_del.split(' ')
        tgtWords = tline.split(' ')

        # TODO: limit the size of insert words and delete words
        if len(srcWords) <= seq_length and len(tgtWords) <= seq_length:
            src += [srcDicts.convertToIdx(srcWords,
                                          s2s.Constants.UNK_WORD)]
            src_ins += [tgtDicts.convertToIdx(srcInsWords,
                                          s2s.Constants.UNK_WORD)]
            src_del += [tgtDicts.convertToIdx(srcDelWords,
                                          s2s.Constants.UNK_WORD)]
            tgt += [tgtDicts.convertToIdx(tgtWords,
                                          s2s.Constants.UNK_WORD,
                                          s2s.Constants.BOS_WORD,
                                          s2s.Constants.EOS_WORD)]

            sizes += [len(srcWords)]
        else:
            ignored += 1

        count += 1

        if count % report_every == 0:
            logger.info('... %d sentences prepared' % count)

    srcF.close()
    tgtF.close()

    if shuffle == 1:
        logger.info('... shuffling sentences')
        perm = torch.randperm(len(src))
        src = [src[idx] for idx in perm]
        src_ins = [src_ins[idx] for idx in perm]
        src_del = [src_del[idx] for idx in perm]
        tgt = [tgt[idx] for idx in perm]
        sizes = [sizes[idx] for idx in perm]

    logger.info('... sorting sentences by size')
    _, perm = torch.sort(torch.Tensor(sizes))
    src = [src[idx] for idx in perm]
    src_ins = [src_ins[idx] for idx in perm]
    src_del = [src_del[idx] for idx in perm]
    tgt = [tgt[idx] for idx in perm]

    logger.info('Prepared %d sentences (%d ignored due to length == 0 or > %d)' %
                (len(src), ignored, seq_length))
    return src, src_ins, src_del, tgt

def prepare_multisourcedata_online(train_src, src_vocab, train_tgt, tgt_vocab):
    size = 30000
    dicts = {}
    dicts['src'] = initVocabulary('source', [train_src], src_vocab, size)
    dicts['tgt'] = initVocabulary('target', [train_tgt], tgt_vocab, size)

    logger.info('Preparing training ...')
    train = {}
    train['src'], train['ins'], train['del'],train['query'], train['tgt'] = makeMultiSourceData(train_src,
                                          train_tgt,
                                          dicts['src'],
                                          dicts['tgt'])

    dataset = {'dicts': dicts,
               'train': train,
               # 'valid': valid
               }
    return dataset

def prepare_data_online(train_src, src_vocab, train_tgt, tgt_vocab):
    size = 30000
    dicts = {}
    dicts['src'] = initVocabulary('source', [train_src], src_vocab, size)
    dicts['tgt'] = initVocabulary('target', [train_tgt], tgt_vocab, size)

    logger.info('Preparing training ...')
    train = {}
    train['src'], train['ins'], train['del'], train['tgt'] = makeData(train_src,
                                          train_tgt,
                                          dicts['src'],
                                          dicts['tgt'])

    dataset = {'dicts': dicts,
               'train': train,
               # 'valid': valid
               }
    return dataset
