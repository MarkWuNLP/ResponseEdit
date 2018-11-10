from __future__ import division

import math
import random

import torch
from torch.autograd import Variable

import s2s


class Dataset(object):
    def __init__(self, srcData, tgtData, batchSize, cuda):
        self.src = srcData
        if tgtData:
            self.tgt = tgtData
            assert (len(self.src) == len(self.tgt))
        else:
            self.tgt = None
        self.device = torch.device("cuda" if cuda else "cpu")

        self.batchSize = batchSize
        self.numBatches = math.ceil(len(self.src) / batchSize)

    def _batchify(self, data, align_right=False, include_lengths=False):
        lengths = [x.size(0) for x in data]
        max_length = max(lengths)
        out = data[0].new(len(data), max_length).fill_(s2s.Constants.PAD)
        for i in range(len(data)):
            data_length = data[i].size(0)
            offset = max_length - data_length if align_right else 0
            out[i].narrow(0, offset, data_length).copy_(data[i])

        if include_lengths:
            return out, lengths
        else:
            return out

    def __getitem__(self, index):
        assert index < self.numBatches, "%d > %d" % (index, self.numBatches)
        srcBatch, lengths = self._batchify(
            self.src[index * self.batchSize:(index + 1) * self.batchSize],
            align_right=False, include_lengths=True)

        if self.tgt:
            tgtBatch = self._batchify(
                self.tgt[index * self.batchSize:(index + 1) * self.batchSize])
        else:
            tgtBatch = None

        # within batch sorting by decreasing length for variable length rnns
        indices = range(len(srcBatch))
        if tgtBatch is None:
            batch = zip(indices, srcBatch)
        else:
            batch = zip(indices, srcBatch, tgtBatch)
        # batch = zip(indices, srcBatch) if tgtBatch is None else zip(indices, srcBatch, tgtBatch)
        batch, lengths = zip(*sorted(zip(batch, lengths), key=lambda x: -x[1]))
        if tgtBatch is None:
            indices, srcBatch = zip(*batch)
        else:
            indices, srcBatch, tgtBatch = zip(*batch)

        def wrap(b):
            if b is None:
                return b
            b = torch.stack(b, 0).t().contiguous()
            b = b.to(self.device)
            return b

        # wrap lengths in a Variable to properly split it in DataParallel
        lengths = torch.LongTensor(lengths).view(1, -1)

        return (wrap(srcBatch), lengths), (wrap(tgtBatch),), indices

    def __len__(self):
        return self.numBatches

    def shuffle(self):
        data = list(zip(self.src, self.tgt))
        self.src, self.tgt = zip(*[data[i] for i in torch.randperm(len(data))])


class MultiSourceDataSet(object):
    def __init__(self, srcData, srcInsDate, srcDelData,srcQueryData, tgtData, batchSize, cuda):
        self.src = srcData
        self.srcIns = srcInsDate
        self.srcDel = srcDelData
        self.srcQue = srcQueryData
        if tgtData:
            self.tgt = tgtData
            assert (len(self.src) == len(self.tgt))
        else:
            self.tgt = None
        self.device = torch.device("cuda" if cuda else "cpu")

        self.batchSize = batchSize
        self.numBatches = math.ceil(len(self.src) / batchSize)

    def _batchify(self, data, align_right=False, include_lengths=False):
        lengths = [x.size(0) for x in data]
        max_length = max(lengths)
        out = data[0].new(len(data), max_length).fill_(s2s.Constants.PAD)
        for i in range(len(data)):
            data_length = data[i].size(0)
            offset = max_length - data_length if align_right else 0
            out[i].narrow(0, offset, data_length).copy_(data[i])

        if include_lengths:
            return out, lengths
        else:
            return out

    def __getitem__(self, index):
        assert index < self.numBatches, "%d > %d" % (index, self.numBatches)
        srcBatch, lengths = self._batchify(
            self.src[index * self.batchSize:(index + 1) * self.batchSize],
            align_right=False, include_lengths=True)
        srcInsBatch, insertlens = self._batchify(
            self.srcIns[index * self.batchSize:(index + 1) * self.batchSize],
            align_right=False, include_lengths=True)
        srcDelBatch, deletelens = self._batchify(
            self.srcDel[index * self.batchSize:(index + 1) * self.batchSize],
            align_right=False, include_lengths=True)
        srcQueBatch, Quelens = self._batchify(
            self.srcQue[index * self.batchSize:(index + 1) * self.batchSize],
            align_right=False, include_lengths=True)

        if self.tgt:
            tgtBatch = self._batchify(
                self.tgt[index * self.batchSize:(index + 1) * self.batchSize])
        else:
            tgtBatch = None

        # within batch sorting by decreasing length for variable length rnns
        indices = range(len(srcBatch))
        if tgtBatch is None:
            batch = zip(indices, srcBatch, srcInsBatch, srcDelBatch,srcQueBatch,Quelens)
        else:
            batch = zip(indices, srcBatch, srcInsBatch, srcDelBatch,srcQueBatch, tgtBatch)
        # batch = zip(indices, srcBatch) if tgtBatch is None else zip(indices, srcBatch, tgtBatch)
        batch, lengths = zip(*sorted(zip(batch, lengths), key=lambda x: -x[1]))
        if tgtBatch is None:
            indices, srcBatch, srcInsBatch, srcDelBatch,srcQueBatch = zip(*batch)
        else:
            indices, srcBatch, srcInsBatch, srcDelBatch,srcQueBatch, tgtBatch = zip(*batch)

        def wrap(b):
            if b is None:
                return b
            b = torch.stack(b, 0).t().contiguous()
            b = b.to(self.device)
            return b

        # wrap lengths in a Variable to properly split it in DataParallel
        lengths = torch.LongTensor(lengths).view(1, -1)
       # insertlens = torch.LongTensor(insertlens).view(1,-1)
       # deletelens = torch.LongTensor(deletelens).view(1, -1)

        return (wrap(srcBatch), lengths), (wrap(srcInsBatch),), (wrap(srcDelBatch), ),(wrap(srcQueBatch),), \
               (wrap(tgtBatch),), indices

    def __len__(self):
        return self.numBatches

    def shuffle(self):
        data = list(zip(self.src, self.srcIns, self.srcDel,self.srcQue, self.tgt))
        self.src, self.srcIns, self.srcDel,self.srcQue, self.tgt = zip(*[data[i] for i in torch.randperm(len(data))])

class IDDataSet(object):
    def __init__(self, srcData, srcInsDate, srcDelData, tgtData, batchSize, cuda):
        self.src = srcData
        self.srcIns = srcInsDate
        self.srcDel = srcDelData
        if tgtData:
            self.tgt = tgtData
            assert (len(self.src) == len(self.tgt))
        else:
            self.tgt = None
        self.device = torch.device("cuda" if cuda else "cpu")

        self.batchSize = batchSize
        self.numBatches = math.ceil(len(self.src) / batchSize)

    def _batchify(self, data, align_right=False, include_lengths=False):
        lengths = [x.size(0) for x in data]
        max_length = max(lengths)
        out = data[0].new(len(data), max_length).fill_(s2s.Constants.PAD)
        for i in range(len(data)):
            data_length = data[i].size(0)
            offset = max_length - data_length if align_right else 0
            out[i].narrow(0, offset, data_length).copy_(data[i])

        if include_lengths:
            return out, lengths
        else:
            return out

    def __getitem__(self, index):
        assert index < self.numBatches, "%d > %d" % (index, self.numBatches)
        srcBatch, lengths = self._batchify(
            self.src[index * self.batchSize:(index + 1) * self.batchSize],
            align_right=False, include_lengths=True)
        srcInsBatch, insertlens = self._batchify(
            self.srcIns[index * self.batchSize:(index + 1) * self.batchSize],
            align_right=False, include_lengths=True)
        srcDelBatch, deletelens = self._batchify(
            self.srcDel[index * self.batchSize:(index + 1) * self.batchSize],
            align_right=False, include_lengths=True)

        if self.tgt:
            tgtBatch = self._batchify(
                self.tgt[index * self.batchSize:(index + 1) * self.batchSize])
        else:
            tgtBatch = None

        # within batch sorting by decreasing length for variable length rnns
        indices = range(len(srcBatch))
        if tgtBatch is None:
            batch = zip(indices, srcBatch, srcInsBatch, srcDelBatch)
        else:
            batch = zip(indices, srcBatch, srcInsBatch, srcDelBatch, tgtBatch)
        # batch = zip(indices, srcBatch) if tgtBatch is None else zip(indices, srcBatch, tgtBatch)
        batch, lengths = zip(*sorted(zip(batch, lengths), key=lambda x: -x[1]))
        if tgtBatch is None:
            indices, srcBatch, srcInsBatch, srcDelBatch = zip(*batch)
        else:
            indices, srcBatch, srcInsBatch, srcDelBatch, tgtBatch = zip(*batch)

        def wrap(b):
            if b is None:
                return b
            b = torch.stack(b, 0).t().contiguous()
            b = b.to(self.device)
            return b

        # wrap lengths in a Variable to properly split it in DataParallel
        lengths = torch.LongTensor(lengths).view(1, -1)
       # insertlens = torch.LongTensor(insertlens).view(1,-1)
       # deletelens = torch.LongTensor(deletelens).view(1, -1)

        return (wrap(srcBatch), lengths), (wrap(srcInsBatch),), (wrap(srcDelBatch), ), \
               (wrap(tgtBatch),), indices

    def __len__(self):
        return self.numBatches

    def shuffle(self):
        data = list(zip(self.src, self.srcIns, self.srcDel, self.tgt))
        self.src, self.srcIns, self.srcDel, self.tgt = zip(*[data[i] for i in torch.randperm(len(data))])